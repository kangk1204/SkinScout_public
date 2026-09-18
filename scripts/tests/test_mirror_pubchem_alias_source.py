from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/mirror_pubchem_alias_source.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mirror_pubchem_alias_source", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_gzip(path: Path, lines: list[str]) -> None:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as handle:
        handle.write("".join(lines).encode("utf-8"))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - source pin contract


def run_mirror(tmp_path: Path, source: Path, *, expected_md5: str | None = None):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path / "out"),
            "--release-date",
            "2026-08-14",
            "--expected-md5",
            expected_md5 or md5(source),
            "--source-file",
            str(source),
            "--source-last-modified-date",
            "2026-08-14",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_cas_source_is_sealed_deterministically(tmp_path: Path) -> None:
    source = tmp_path / "source.gz"
    write_gzip(source, ["1\tCCO\n", "2244\tCC(=O)Oc1ccccc1C(=O)O\n"])

    first = run_mirror(tmp_path, source)
    assert first.returncode == 0, first.stderr
    manifest_path = tmp_path / "out/source_manifest.json"
    first_bytes = manifest_path.read_bytes()
    payload = json.loads(first_bytes)
    assert payload["schema_version"] == "skinscout.pubchem-alias-source.v1"
    assert payload["artifact"]["rows"] == 2
    assert payload["artifact"]["first_cid"] == 1
    assert payload["artifact"]["last_cid"] == 2244
    assert payload["source"]["redistribution_scope"].startswith("PubChem-generated")

    second = run_mirror(tmp_path, source)
    assert second.returncode == 0, second.stderr
    assert manifest_path.read_bytes() == first_bytes


def test_wrong_md5_removes_stale_claim_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.gz"
    write_gzip(source, ["1\tCCO\n"])
    out = tmp_path / "out"
    out.mkdir()
    (out / "source_manifest.json").write_text("stale\n")
    (out / "CID-SMILES.gz").write_text("stale\n")

    result = run_mirror(tmp_path, source, expected_md5="0" * 32)

    assert result.returncode != 0
    assert "MD5 mismatch" in result.stderr
    assert not (out / "source_manifest.json").exists()
    assert not (out / "CID-SMILES.gz").exists()


@pytest.mark.parametrize(
    "lines,error",
    [
        (["1 CCO\n"], "CID<TAB>SMILES"),
        (["1\tCCO\n", "1\tCCC\n"], "strictly increasing"),
        (["2\tCCO"], "newline terminated"),
    ],
)
def test_malformed_content_fails_closed(
    tmp_path: Path, lines: list[str], error: str
) -> None:
    source = tmp_path / "source.gz"
    write_gzip(source, lines)

    result = run_mirror(tmp_path, source)

    assert result.returncode != 0
    assert error in result.stderr


def test_release_date_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.gz"
    write_gzip(source, ["1\tCCO\n"])
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(tmp_path / "out"),
            "--release-date",
            "2026-08-13",
            "--expected-md5",
            md5(source),
            "--source-file",
            str(source),
            "--source-last-modified-date",
            "2026-08-14",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Last-Modified date mismatch" in result.stderr


def test_symlink_source_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real.gz"
    write_gzip(real, ["1\tCCO\n"])
    source = tmp_path / "source.gz"
    source.symlink_to(real)

    result = run_mirror(tmp_path, source, expected_md5=md5(real))

    assert result.returncode != 0
    assert "unsafe" in result.stderr


def test_validator_rejects_artifact_and_manifest_forgery(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source.gz"
    write_gzip(source, ["1\tCCO\n"])
    result = run_mirror(tmp_path, source)
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    artifact = out / "CID-SMILES.gz"
    write_gzip(artifact, ["1\tCCC\n"])
    manifest_path = out / "source_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["artifact"]["bytes"] = artifact.stat().st_size
    payload["artifact"]["md5"] = md5(artifact)
    payload["artifact"]["sha256"] = module._sha256_file(artifact)
    payload["artifact"].update(module._validate_cid_smiles(artifact))
    payload["binding_sha256"] = module._binding_sha256(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.MirrorError, match="configured MD5"):
        module.validate_manifest(
            manifest_path,
            expected_release_date="2026-08-14",
            expected_md5=md5(source),
        )


@pytest.mark.parametrize(
    "field,value,error",
    [
        (("artifact", "url"), "https://example.invalid/CID-SMILES.gz", "URL mismatch"),
        (("artifact", "last_modified_date"), "2026-08-13", "last_modified_date"),
    ],
)
def test_validator_rejects_forged_release_binding(
    tmp_path: Path,
    field: tuple[str, str],
    value: str,
    error: str,
) -> None:
    module = load_module()
    source = tmp_path / "source.gz"
    write_gzip(source, ["1\tCCO\n"])
    result = run_mirror(tmp_path, source)
    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "out/source_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload[field[0]][field[1]] = value
    payload["binding_sha256"] = module._binding_sha256(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.MirrorError, match=error):
        module.validate_manifest(
            manifest_path,
            expected_release_date="2026-08-14",
            expected_md5=md5(source),
            expected_url=module.DEFAULT_URL,
        )


def test_manifest_remains_valid_after_relocation(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source.gz"
    write_gzip(source, ["1\tCCO\n"])
    result = run_mirror(tmp_path, source)
    assert result.returncode == 0, result.stderr

    relocated = tmp_path / "relocated"
    shutil.move(str(tmp_path / "out"), relocated)

    payload = module.validate_manifest(
        relocated / "source_manifest.json",
        expected_release_date="2026-08-14",
        expected_md5=md5(source),
    )
    assert payload["artifact"]["path"] == "CID-SMILES.gz"
