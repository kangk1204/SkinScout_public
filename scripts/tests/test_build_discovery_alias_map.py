from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build_discovery_alias_map.py"
SOURCES = ("ChEMBL", "PubChem", "BindingDB", "GtoPdb")


def load_builder():
    spec = importlib.util.spec_from_file_location("build_discovery_alias_map", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return load_builder().sha256_file(path)


def write_table(path: Path, rows: list[dict[str, str]], fmt: str) -> None:
    frame = pd.DataFrame(rows)
    if fmt == "parquet":
        frame.to_parquet(path, index=False)
    elif fmt == "tsv":
        frame.to_csv(path, sep="\t", index=False)
    else:
        frame.to_csv(path, index=False)


def write_source_manifest(
    tmp_path: Path,
    registry_path: Path,
    registry: dict,
    row_counts: dict[str, int],
) -> Path:
    provenance = tmp_path / "provenance"
    provenance.mkdir(exist_ok=True)
    inputs = {}
    for name in SOURCES:
        raw = provenance / f"{name}.raw"
        raw.write_text(f"{name} raw source\n", encoding="utf-8")
        upstream = provenance / f"{name}.manifest.json"
        upstream.write_text(json.dumps({"source": name}) + "\n", encoding="utf-8")
        inputs[name] = {
            "path": str(raw.relative_to(tmp_path)),
            "sha256": sha256(raw),
            "bytes": raw.stat().st_size,
            "manifest": {
                "path": str(upstream.relative_to(tmp_path)),
                "sha256": sha256(upstream),
                "bytes": upstream.stat().st_size,
            },
        }
    output_hashes = {}
    output_bytes = {}
    output_rows = {}
    for source in registry["sources"]:
        artifact = source["artifact"]
        name = source["canonical_name"]
        basename = Path(str(artifact["path"])).name
        output_hashes[basename] = artifact["sha256"]
        output_bytes[basename] = artifact["bytes"]
        output_rows[basename] = row_counts[name]
    builder = ROOT / "scripts/build_discovery_alias_sources.py"
    unsigned = {
        "schema_version": "skinscout.discovery-alias-sources.v1",
        "builder": {
            "path": "scripts/build_discovery_alias_sources.py",
            "sha256": sha256(builder),
        },
        "inputs": inputs,
        "output_sha256": output_hashes,
        "output_bytes": output_bytes,
        "output_rows": output_rows,
        "source_registry": registry_path.name,
        "source_registry_sha256": sha256(registry_path),
    }
    stable = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    manifest = {
        **unsigned,
        "self_binding_sha256": hashlib.sha256(stable.encode("utf-8")).hexdigest(),
    }
    path = tmp_path / "source_manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def write_registry(
    tmp_path: Path,
    *,
    rows_by_source: dict[str, list[dict[str, str]]] | None = None,
    omit: set[str] | None = None,
    mutate: Callable[[dict], None] | None = None,
) -> Path:
    rows_by_source = rows_by_source or {}
    omit = omit or set()
    formats = {"ChEMBL": "csv", "PubChem": "tsv", "BindingDB": "parquet", "GtoPdb": "csv"}
    sources = []
    row_counts: dict[str, int] = {}
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in SOURCES:
        if name in omit:
            continue
        fmt = formats[name]
        suffix = "tsv" if fmt == "tsv" else fmt
        artifact = data_dir / f"{name}.{suffix}"
        rows = rows_by_source.get(
            name,
            [{"smiles": "CCO", "name": f"{name} Ethanol", "id": f"{name}:1", "ik": ""}],
        )
        write_table(artifact, rows, fmt)
        row_counts[name] = len(rows)
        sources.append(
            {
                "canonical_name": name,
                "release": "2026.01",
                "release_date": "2026-01-31",
                "spdx_license": "CC-BY-4.0",
                "license_url": "https://example.org/license",
                "redistribution": "allowed",
                "artifact": {
                    "path": str(artifact.relative_to(tmp_path)),
                    "sha256": sha256(artifact),
                    "bytes": artifact.stat().st_size,
                    "format": fmt,
                },
                "columns": {
                    "smiles": "smiles",
                    "aliases": ["name"],
                    "inchikey": "ik",
                    "source_record_id": "id",
                },
            }
        )
    registry = {
        "schema_version": "discovery_alias_source_registry.v1",
        "sources": sources,
    }
    if mutate:
        mutate(registry)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    write_source_manifest(tmp_path, path, registry, row_counts)
    return path


def run_builder(
    tmp_path: Path,
    registry: Path,
    *,
    canonicalization_workers: int = 1,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--source-registry",
            str(registry),
            "--source-manifest",
            str(tmp_path / "source_manifest.json"),
            "--out-aliases-parquet",
            str(tmp_path / "aliases.parquet"),
            "--out-direct-reference",
            str(tmp_path / "direct.smi"),
            "--out-manifest",
            str(tmp_path / "manifest.json"),
            "--canonicalization-workers",
            str(canonicalization_workers),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_successful_four_source_build(tmp_path: Path) -> None:
    registry = write_registry(tmp_path)

    result = run_builder(tmp_path, registry)

    assert result.returncode == 0, result.stderr
    aliases = pd.read_parquet(tmp_path / "aliases.parquet")
    assert aliases["source"].tolist() == ["BindingDB", "ChEMBL", "GtoPdb", "PubChem"]
    assert aliases["alias"].tolist() == [
        "bindingdb ethanol",
        "chembl ethanol",
        "gtopdb ethanol",
        "pubchem ethanol",
    ]
    assert set(aliases["parent_canonical_smiles"]) == {"CCO"}
    assert (tmp_path / "direct.smi").read_text(encoding="utf-8").startswith("CCO ")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "discovery_alias_map.v1"
    assert manifest["source_registry_contract"] == "discovery_alias_source_registry.v1"
    assert not Path(manifest["source_registry"]).is_absolute()
    assert not Path(manifest["output_paths"]["aliases_parquet"]).is_absolute()
    assert manifest["counts"]["alias_rows"] == 4
    assert manifest["counts"]["unique_discovery_keys"] == 1
    assert manifest["counts"]["ambiguous_alias_count"] == 0
    assert manifest["canonicalization_audit"]["rows"] == 0
    assert manifest["policy"][
        "max_canonicalization_exclusion_fraction_ppm_per_source"
    ] == 10_000
    assert manifest["binding_sha256"]


def test_manifest_remains_valid_after_data_pack_relocation(tmp_path: Path) -> None:
    builder = load_builder()
    original = tmp_path / "original"
    original.mkdir()
    registry = write_registry(original)
    result = run_builder(original, registry)
    assert result.returncode == 0, result.stderr

    relocated = tmp_path / "relocated"
    shutil.move(str(original), relocated)

    builder.validate_manifest(
        relocated / "manifest.json",
        expected_registry=relocated / "registry.json",
    )


def test_canonical_collapse_across_salt_and_stereo_variants(tmp_path: Path) -> None:
    registry = write_registry(
        tmp_path,
        rows_by_source={
            "ChEMBL": [
                {"smiles": "C[C@H](O)C", "name": "Isopropanol", "id": "c1", "ik": ""},
                {"smiles": "CC(C)O.Cl", "name": "Isopropanol", "id": "c2", "ik": ""},
            ],
        },
    )

    result = run_builder(tmp_path, registry)

    assert result.returncode == 0, result.stderr
    aliases = pd.read_parquet(tmp_path / "aliases.parquet")
    chembl = aliases[aliases["source"] == "ChEMBL"]
    assert chembl["alias"].tolist() == ["isopropanol"]
    assert chembl["parent_canonical_smiles"].tolist() == ["CC(C)O"]
    assert "CC(C)O " in (tmp_path / "direct.smi").read_text(encoding="utf-8")


def test_ambiguous_alias_is_omitted(tmp_path: Path) -> None:
    registry = write_registry(
        tmp_path,
        rows_by_source={
            "ChEMBL": [{"smiles": "CCO", "name": "Shared", "id": "c1", "ik": ""}],
            "PubChem": [{"smiles": "CCC", "name": " shared ", "id": "p1", "ik": ""}],
        },
    )

    result = run_builder(tmp_path, registry)

    assert result.returncode == 0, result.stderr
    aliases = pd.read_parquet(tmp_path / "aliases.parquet")
    assert "shared" not in set(aliases["alias"])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["ambiguous_alias_key_counts"] == {"shared": 2}
    assert manifest["counts"]["omitted_ambiguous_candidate_rows"] == 2
    direct_smiles = {
        line.split(" ", 1)[0]
        for line in (tmp_path / "direct.smi").read_text(encoding="utf-8").splitlines()
    }
    assert {"CCO", "CCC"}.issubset(direct_smiles)


def test_missing_one_source_fails_closed(tmp_path: Path) -> None:
    registry = write_registry(tmp_path, omit={"GtoPdb"})

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "missing=['GtoPdb']" in result.stderr


def test_license_and_redistribution_fail_closed(tmp_path: Path) -> None:
    def mutate(registry: dict) -> None:
        registry["sources"][0]["license_url"] = "http://example.org/license"
        registry["sources"][1]["redistribution"] = "restricted"

    registry = write_registry(tmp_path, mutate=mutate)

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "license_url must be an https URL" in result.stderr


def test_source_hash_drift_fails_closed(tmp_path: Path) -> None:
    registry = write_registry(tmp_path)
    artifact = tmp_path / "data" / "ChEMBL.csv"
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "bytes drift" in result.stderr


def test_absolute_source_artifact_path_fails_closed(tmp_path: Path) -> None:
    def mutate(registry: dict) -> None:
        registry["sources"][0]["artifact"]["path"] = str(
            (tmp_path / "data" / "ChEMBL.csv").resolve()
        )

    registry = write_registry(tmp_path, mutate=mutate)

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "artifact.path must be relative" in result.stderr


def test_source_artifact_path_escape_fails_closed(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external.csv"
    external.write_text("smiles,name,id,ik\nCCO,external,e1,\n", encoding="utf-8")

    def mutate(registry: dict) -> None:
        artifact = registry["sources"][0]["artifact"]
        artifact["path"] = f"../{external.name}"
        artifact["sha256"] = sha256(external)
        artifact["bytes"] = external.stat().st_size

    registry = write_registry(tmp_path, mutate=mutate)

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "artifact.path escapes registry_root" in result.stderr


@pytest.mark.parametrize("nested_manifest", [False, True])
def test_source_manifest_nested_input_path_escape_fails_closed(
    tmp_path: Path,
    nested_manifest: bool,
) -> None:
    builder = load_builder()
    registry = write_registry(tmp_path)
    manifest_path = tmp_path / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    external = tmp_path.parent / f"{tmp_path.name}-outside-provenance"
    external.write_text("outside provenance\n", encoding="utf-8")
    record = manifest["inputs"]["ChEMBL"]
    item = record["manifest"] if nested_manifest else record
    item.update(
        {
            "path": f"../{external.name}",
            "sha256": builder.sha256_file(external),
            "bytes": external.stat().st_size,
        }
    )
    unsigned = dict(manifest)
    unsigned.pop("self_binding_sha256", None)
    manifest["self_binding_sha256"] = builder.sha256_text(
        builder.stable_json(unsigned)
    )
    manifest_path.write_text(
        builder.stable_json(manifest) + "\n",
        encoding="utf-8",
    )

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "Stage 0 provenance root" in result.stderr
    assert not (tmp_path / "aliases.parquet").exists()
    assert not (tmp_path / "direct.smi").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_output_path_escape_fails_without_touching_external_file(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    package = tmp_path / "package"
    package.mkdir()
    registry = write_registry(package)
    external = tmp_path / "outside.parquet"
    external.write_bytes(b"must remain untouched\n")

    with pytest.raises(builder.ContractError, match="alias package root"):
        builder.build(
            source_registry=registry,
            source_manifest=package / "source_manifest.json",
            out_aliases_parquet=external,
            out_direct_reference=package / "direct.smi",
            out_manifest=package / "manifest.json",
        )

    assert external.read_bytes() == b"must remain untouched\n"
    assert not (package / "direct.smi").exists()
    assert not (package / "manifest.json").exists()


def test_invalid_smiles_fails_closed(tmp_path: Path) -> None:
    registry = write_registry(
        tmp_path,
        rows_by_source={
            "GtoPdb": [{"smiles": "not_a_smiles", "name": "Bad", "id": "g1", "ik": ""}],
        },
    )

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "canonicalization exclusion fraction" in result.stderr


def test_bounded_invalid_smiles_are_audited(tmp_path: Path) -> None:
    valid_rows = [
        {
            "smiles": "CCO",
            "name": f"valid-{index}",
            "id": f"g:{index}",
            "ik": "",
        }
        for index in range(100)
    ]
    registry = write_registry(
        tmp_path,
        rows_by_source={
            "GtoPdb": [
                {"smiles": "not_a_smiles", "name": "Bad", "id": "g:bad", "ik": ""},
                *valid_rows,
            ],
        },
    )

    result = run_builder(tmp_path, registry)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counts"]["source_counts"]["GtoPdb"][
        "canonicalization_exclusions"
    ] == 1
    assert manifest["canonicalization_audit"]["rows"] == 1
    audit = (tmp_path / "canonicalization_exclusions.jsonl").read_text()
    assert '"reason_code":"rdkit_canonicalization_failed"' in audit
    assert '"source_record_id":"g:bad"' in audit


def test_parallel_canonicalization_matches_serial_bytes(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    valid_rows = [
        {
            "smiles": "CCO",
            "name": f"valid-{index}",
            "id": f"g:{index}",
            "ik": "",
        }
        for index in range(100)
    ]
    write_registry(
        fixture,
        rows_by_source={
            "GtoPdb": [
                {"smiles": "not_a_smiles", "name": "Bad", "id": "g:bad", "ik": ""},
                *valid_rows,
            ],
            "PubChem": [
                {"smiles": "CCO", "name": "shared", "id": "p:1", "ik": ""},
                {"smiles": "CCN", "name": "ambiguous", "id": "p:2", "ik": ""},
            ],
            "BindingDB": [
                {"smiles": "CCC", "name": "ambiguous", "id": "b:1", "ik": ""},
            ],
        },
    )
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    shutil.copytree(fixture, serial)
    shutil.copytree(fixture, parallel)

    serial_result = run_builder(
        serial,
        serial / "registry.json",
        canonicalization_workers=1,
    )
    parallel_result = run_builder(
        parallel,
        parallel / "registry.json",
        canonicalization_workers=2,
    )

    assert serial_result.returncode == 0, serial_result.stderr
    assert parallel_result.returncode == 0, parallel_result.stderr
    for name in (
        "aliases.parquet",
        "direct.smi",
        "canonicalization_exclusions.jsonl",
        "manifest.json",
    ):
        assert sha256(serial / name) == sha256(parallel / name)


def test_parallel_duplicate_invalid_smiles_are_audited_per_row(
    tmp_path: Path,
) -> None:
    valid_rows = [
        {
            "smiles": "CCO",
            "name": f"valid-{index}",
            "id": f"g:{index}",
            "ik": "",
        }
        for index in range(200)
    ]
    registry = write_registry(
        tmp_path,
        rows_by_source={
            "GtoPdb": [
                {"smiles": "not_a_smiles", "name": "Bad 1", "id": "g:bad:1", "ik": ""},
                {"smiles": "not_a_smiles", "name": "Bad 2", "id": "g:bad:2", "ik": ""},
                *valid_rows,
            ],
        },
    )

    result = run_builder(tmp_path, registry, canonicalization_workers=2)

    assert result.returncode == 0, result.stderr
    records = [
        json.loads(line)
        for line in (tmp_path / "canonicalization_exclusions.jsonl").read_text().splitlines()
    ]
    invalid = [record for record in records if record["source"] == "GtoPdb"]
    assert [record["row_index"] for record in invalid] == [0, 1]
    assert [record["source_record_id"] for record in invalid] == ["g:bad:1", "g:bad:2"]
    assert len({record["input_smiles_sha256"] for record in invalid}) == 1


@pytest.mark.parametrize("workers", [0, -1, 65])
def test_invalid_canonicalization_worker_count_fails_closed(
    tmp_path: Path,
    workers: int,
) -> None:
    registry = write_registry(tmp_path)

    result = run_builder(
        tmp_path,
        registry,
        canonicalization_workers=workers,
    )

    assert result.returncode != 0
    assert "canonicalization workers" in result.stderr
    assert not (tmp_path / "aliases.parquet").exists()
    assert not (tmp_path / "direct.smi").exists()
    assert not (tmp_path / "canonicalization_exclusions.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_parallel_worker_exception_removes_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = write_registry(tmp_path)
    builder = load_builder()

    def fail_canonicalization(*_args, **_kwargs):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(builder, "discovery_key", fail_canonicalization)
    with pytest.raises(RuntimeError, match="worker exploded"):
        builder.build(
            source_registry=registry,
            source_manifest=tmp_path / "source_manifest.json",
            out_aliases_parquet=tmp_path / "aliases.parquet",
            out_direct_reference=tmp_path / "direct.smi",
            out_manifest=tmp_path / "manifest.json",
            canonicalization_workers=2,
        )

    assert not (tmp_path / "aliases.parquet").exists()
    assert not (tmp_path / "direct.smi").exists()
    assert not (tmp_path / "canonicalization_exclusions.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob(".discovery-alias-map-*.sqlite"))


def test_build_self_validation_reuses_active_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = load_builder()
    registry = write_registry(tmp_path)
    original_build_stage = builder._build_stage
    build_stage_calls = 0

    def counted_build_stage(*args, **kwargs):
        nonlocal build_stage_calls
        build_stage_calls += 1
        return original_build_stage(*args, **kwargs)

    monkeypatch.setattr(builder, "_build_stage", counted_build_stage)
    builder.build(
        source_registry=registry,
        source_manifest=tmp_path / "source_manifest.json",
        out_aliases_parquet=tmp_path / "aliases.parquet",
        out_direct_reference=tmp_path / "direct.smi",
        out_manifest=tmp_path / "manifest.json",
        canonicalization_workers=1,
    )

    assert build_stage_calls == 1


def test_all_blank_aliases_fail_closed(tmp_path: Path) -> None:
    registry = write_registry(
        tmp_path,
        rows_by_source={
            source: [{"smiles": "CCO", "name": "  ", "id": f"{source}:1", "ik": ""}]
            for source in SOURCES
        },
    )

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert "no unambiguous aliases" in result.stderr


def test_stale_outputs_removed_on_failure(tmp_path: Path) -> None:
    registry = write_registry(tmp_path, omit={"PubChem"})
    for name in (
        "aliases.parquet",
        "direct.smi",
        "canonicalization_exclusions.jsonl",
        "manifest.json",
    ):
        (tmp_path / name).write_text("stale", encoding="utf-8")

    result = run_builder(tmp_path, registry)

    assert result.returncode != 0
    assert not (tmp_path / "aliases.parquet").exists()
    assert not (tmp_path / "direct.smi").exists()
    assert not (tmp_path / "canonicalization_exclusions.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_validator_rejects_output_mutation(tmp_path: Path) -> None:
    builder = load_builder()
    registry = write_registry(tmp_path)
    result = run_builder(tmp_path, registry)
    assert result.returncode == 0, result.stderr
    (tmp_path / "direct.smi").write_text("CCO mutated\n", encoding="utf-8")

    with pytest.raises(builder.ContractError, match="direct reference hash drift"):
        builder.validate_manifest(tmp_path / "manifest.json", expected_registry=registry)


def test_manifest_validator_rejects_relative_output_escape(tmp_path: Path) -> None:
    builder = load_builder()
    registry = write_registry(tmp_path)
    result = run_builder(tmp_path, registry)
    assert result.returncode == 0, result.stderr
    external = tmp_path.parent / f"{tmp_path.name}-external.smi"
    external.write_text((tmp_path / "direct.smi").read_text(), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_paths"]["direct_reference"] = f"../{external.name}"
    manifest.pop("binding_sha256")
    manifest["binding_sha256"] = builder.sha256_text(builder.stable_json(manifest))
    manifest_path.write_text(builder.stable_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(builder.ContractError, match="alias package root"):
        builder.validate_manifest(manifest_path, expected_registry=registry)


def test_manifest_validator_rejects_joint_output_and_manifest_forgery(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    registry = write_registry(tmp_path)
    result = run_builder(tmp_path, registry)
    assert result.returncode == 0, result.stderr
    direct = tmp_path / "direct.smi"
    direct.write_text("CCC forged\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["direct_reference_sha256"] = builder.sha256_file(direct)
    manifest["output_bytes"]["direct_reference"] = direct.stat().st_size
    payload = builder._canonical_payload(
        registry_hash=manifest["registry_sha256"],
        source_manifest_hash=manifest["source_manifest_sha256"],
        source_hashes=manifest["source_artifact_sha256"],
        alias_hash=manifest["outputs"]["aliases_parquet_sha256"],
        direct_reference_hash=manifest["outputs"]["direct_reference_sha256"],
        builder_hash=manifest["builder_script_sha256"],
        counts=manifest["counts"],
        canonicalization_audit=manifest["canonicalization_audit"],
        policy=manifest["policy"],
    )
    manifest["canonical_payload_sha256"] = builder.sha256_text(builder.stable_json(payload))
    manifest.pop("binding_sha256")
    manifest["binding_sha256"] = builder.sha256_text(builder.stable_json(manifest))
    manifest_path.write_text(builder.stable_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(builder.ContractError, match="active source rebuild"):
        builder.validate_manifest(manifest_path, expected_registry=registry)


def test_manifest_validator_rejects_joint_audit_and_manifest_forgery(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    registry = write_registry(tmp_path)
    result = run_builder(tmp_path, registry)
    assert result.returncode == 0, result.stderr
    audit = tmp_path / builder.CANONICALIZATION_AUDIT_FILENAME
    audit.write_text(
        builder.stable_json({
            "input_smiles_sha256": "0" * 64,
            "reason_code": "rdkit_canonicalization_failed",
            "row_index": 999,
            "source": "GtoPdb",
            "source_record_id": "forged",
        }) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonicalization_audit"] = {
        "path": audit.name,
        "sha256": builder.sha256_file(audit),
        "bytes": audit.stat().st_size,
        "rows": 1,
    }
    payload = builder._canonical_payload(
        registry_hash=manifest["registry_sha256"],
        source_manifest_hash=manifest["source_manifest_sha256"],
        source_hashes=manifest["source_artifact_sha256"],
        alias_hash=manifest["outputs"]["aliases_parquet_sha256"],
        direct_reference_hash=manifest["outputs"]["direct_reference_sha256"],
        builder_hash=manifest["builder_script_sha256"],
        counts=manifest["counts"],
        canonicalization_audit=manifest["canonicalization_audit"],
        policy=manifest["policy"],
    )
    manifest["canonical_payload_sha256"] = builder.sha256_text(
        builder.stable_json(payload)
    )
    manifest.pop("binding_sha256")
    manifest["binding_sha256"] = builder.sha256_text(builder.stable_json(manifest))
    manifest_path.write_text(builder.stable_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(builder.ContractError, match="active source rebuild"):
        builder.validate_manifest(manifest_path, expected_registry=registry)


def test_deterministic_rerun_hashes_for_same_paths(tmp_path: Path) -> None:
    registry = write_registry(tmp_path)

    first = run_builder(tmp_path, registry)
    assert first.returncode == 0, first.stderr
    first_hashes = {
        name: sha256(tmp_path / name)
        for name in (
            "aliases.parquet",
            "direct.smi",
            "canonicalization_exclusions.jsonl",
            "manifest.json",
        )
    }
    second = run_builder(tmp_path, registry)
    assert second.returncode == 0, second.stderr
    second_hashes = {
        name: sha256(tmp_path / name)
        for name in (
            "aliases.parquet",
            "direct.smi",
            "canonicalization_exclusions.jsonl",
            "manifest.json",
        )
    }
    assert second_hashes == first_hashes


def test_streams_multiple_parquet_batches_and_validator_avoids_full_table_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = load_builder()
    row_count = builder.STREAM_BATCH_SIZE * 2 + 7
    registry = write_registry(
        tmp_path,
        rows_by_source={
            "ChEMBL": [
                {
                    "smiles": "CCO",
                    "name": f"alias-{index:06d}",
                    "id": f"c:{index}",
                    "ik": "",
                }
                for index in range(row_count)
            ]
        },
    )

    result = run_builder(tmp_path, registry)

    assert result.returncode == 0, result.stderr
    parquet = builder.pq.ParquetFile(tmp_path / "aliases.parquet")
    assert parquet.metadata.num_rows == row_count + 3
    assert parquet.metadata.num_row_groups >= 3

    def reject_full_read(*_args, **_kwargs):
        raise AssertionError("validator must stream Parquet batches")

    monkeypatch.setattr(builder.pq, "read_table", reject_full_read)
    builder.validate_manifest(tmp_path / "manifest.json", expected_registry=registry)
