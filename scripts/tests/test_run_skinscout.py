"""Regression tests for the user-facing SkinScout launcher."""

from __future__ import annotations

# A real single-molecule SDF: the runner now refuses inputs outside the
# documented scope, and an unparseable placeholder is one of them.
CAFFEINE_SDF = """
     RDKit          3D

 14 15  0  0  0  0  0  0  0  0999 V2000
   -1.3276    2.7758    0.3282 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9139    1.3789    0.1374 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.3962    1.1304    0.0685 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2433    2.0523    0.1628 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.8165   -0.1834   -0.1118 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1444   -1.1648   -0.2105 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5328   -2.3350   -0.3796 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.8531   -2.0769   -0.3838 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0315   -0.7531   -0.2191 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.2648    0.0113   -0.1560 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4542   -0.8542   -0.1337 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4583   -1.9024   -0.2397 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8688    0.4349    0.0433 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0933    0.7157    0.1139 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  2  0
  3  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  1  0
  6 11  1  0
 11 12  1  0
 11 13  1  0
 13 14  2  0
 13  2  1  0
  9  5  1  0
M  END
$$$$
"""


import importlib.util
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_skinscout.py"
_TEST_RESULTS_TMP = tempfile.TemporaryDirectory(prefix="skinscout-runner-tests-")
TEST_RESULTS_ROOT = Path(_TEST_RESULTS_TMP.name)


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_skinscout_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEFAULT_RESULTS_ROOT = TEST_RESULTS_ROOT
    return module


def run_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["SKINSCOUT_RESULTS_ROOT"] = str(TEST_RESULTS_ROOT)
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_discovery_preflight(args: list[str]):
    runner = load_runner_module()
    parsed = runner.parse_args(args)
    runner._validate_evidence_mode_execution(parsed)
    return runner


def write_valid_discovery_audit(run_dir: Path, smiles: str) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "eval"))
    import leakage_check
    from leakage_check import Thresholds, audit, sealed_discovery_audit

    source_dir = run_dir / "discovery_audit_sources"
    source_dir.mkdir(parents=True)
    input_csv = source_dir / "eval_targets.csv"
    seq = source_dir / "training_cutoff_seqs.fasta"
    ligands = source_dir / "training_ligands.smi"
    pockets = source_dir / "training_holo_pockets.csv"
    direct = source_dir / "direct_exact_reference.smi"
    out_csv = source_dir / "leakage_audit.csv"
    challenge = hashlib.sha256(f"discovery:{run_dir}".encode()).hexdigest()
    (run_dir / "discovery_audit_challenge.txt").write_text(challenge + "\n")
    rows = pd.DataFrame([{
        "uniprot": "P1",
        "smiles": smiles,
        "sequence": "MABCXE",
    }])
    rows.to_csv(input_csv, index=False)
    seq.write_text(">train\nYYYYY\n")
    ligands.write_text("c1ccccc1 train\n")
    pockets.write_text("target_id,pocket_sucos\nP1,0.1\n")
    direct.write_text("CCN direct\n")
    thresholds = Thresholds(0.95, 1.0, 1.0)
    # F10: the sequence axis is an MMseqs alignment search; this fixture seals
    # an audit, so stub the search instead of depending on MMseqs/PATH.
    with patch.object(leakage_check, "mmseqs_max_seq_id", return_value=0.1):
        audited = audit(rows, seq, ligands, pockets, thresholds)
    audited.to_csv(out_csv, index=False)
    payload = sealed_discovery_audit(
        rows,
        audited,
        input_csv=input_csv,
        training_seq_db=seq,
        training_ligands=ligands,
        training_holo=pockets,
        direct_exact_reference=direct,
        out_csv=out_csv,
        thresholds=thresholds,
        execution_challenge=challenge,
    )
    (run_dir / "discovery_leakage_audit.json").write_text(
        json.dumps(payload) + "\n"
    )
    return payload


def write_valid_discovery_alias_package(
    alias_dir: Path,
    direct_smiles: str,
) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "eval"))
    import leakage_check
    from discovery_canonical import discovery_key

    alias_dir.mkdir(parents=True)
    identity = discovery_key(direct_smiles)
    direct = alias_dir / "direct_exact_reference.smi"
    direct.write_text(
        f"{identity.parent_canonical_smiles} "
        f"{identity.discovery_key_sha256}\n"
    )
    aliases = alias_dir / "aliases.parquet"
    aliases.write_bytes(b"sealed aliases\n")
    canonicalization_audit = alias_dir / "canonicalization_exclusions.jsonl"
    canonicalization_audit.write_text("")
    source_dir = alias_dir / "sources"
    upstream_dir = alias_dir / "upstream"
    source_dir.mkdir()
    upstream_dir.mkdir()
    source_names = ("ChEMBL", "BindingDB", "GtoPdb", "PubChem")
    registry_sources: list[dict[str, object]] = []
    source_inputs: dict[str, object] = {}
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    output_bytes: dict[str, int] = {}
    output_rows: dict[str, int] = {}
    for source_name in source_names:
        slug = source_name.lower()
        source_artifact = source_dir / f"{slug}_aliases.parquet"
        source_artifact.write_bytes(f"{source_name} aliases\n".encode())
        source_hash = hashlib.sha256(source_artifact.read_bytes()).hexdigest()
        source_hashes[source_name] = source_hash
        output_hashes[source_artifact.name] = source_hash
        output_bytes[source_artifact.name] = source_artifact.stat().st_size
        output_rows[source_artifact.name] = 1

        raw_input = upstream_dir / f"{slug}.dat"
        raw_input.write_bytes(f"{source_name} raw input\n".encode())
        raw_hash = hashlib.sha256(raw_input.read_bytes()).hexdigest()
        source_meta = {
            "name": source_name,
            "release": "2026.1",
            "release_date": "2026-01-01",
            "license": "CC BY 4.0",
            "license_url": "https://example.org/license",
            "redistribution": "allowed",
        }
        if source_name == "ChEMBL":
            upstream = {
                "schema_version": "chembl_activity_evidence.v1",
                "source": {
                    **source_meta,
                    "source_db": raw_input.name,
                    "source_db_sha256": raw_hash,
                    "source_db_bytes": raw_input.stat().st_size,
                },
            }
        elif source_name == "BindingDB":
            upstream = {
                "schema_version": 1,
                "source": {
                    key: value
                    for key, value in source_meta.items()
                    if key not in {"release", "release_date"}
                },
                "release": source_meta["release"],
                "release_date": source_meta["release_date"],
                "extracted": {
                    "path": raw_input.name,
                    "sha256": raw_hash,
                    "bytes": raw_input.stat().st_size,
                },
            }
        elif source_name == "GtoPdb":
            upstream = {
                "schema_version": "skinscout.gtopdb-source.v1",
                "source": source_meta,
                "artifacts": {
                    "ligands.csv": {
                        "path": raw_input.name,
                        "sha256": raw_hash,
                        "bytes": raw_input.stat().st_size,
                    }
                },
            }
        else:
            upstream = {
                "schema_version": "skinscout.pubchem-alias-source.v1",
                "source": source_meta,
                "artifact": {
                    "path": raw_input.name,
                    "sha256": raw_hash,
                    "bytes": raw_input.stat().st_size,
                },
            }
        upstream_manifest = upstream_dir / f"{slug}_manifest.json"
        upstream_manifest.write_text(json.dumps(upstream, sort_keys=True) + "\n")
        source_inputs[source_name] = {
            "path": os.path.relpath(raw_input, source_dir),
            "sha256": raw_hash,
            "bytes": raw_input.stat().st_size,
            "manifest": {
                "path": os.path.relpath(upstream_manifest, source_dir),
                "sha256": hashlib.sha256(upstream_manifest.read_bytes()).hexdigest(),
                "bytes": upstream_manifest.stat().st_size,
                **{
                    key: source_meta[key]
                    for key in ("release", "release_date", "license", "license_url")
                },
            },
        }
        registry_sources.append({
            "canonical_name": source_name,
            "release": source_meta["release"],
            "release_date": source_meta["release_date"],
            "spdx_license": "CC-BY-4.0",
            "license": source_meta["license"],
            "license_url": source_meta["license_url"],
            "redistribution": "allowed",
            "artifact": {
                "path": source_artifact.name,
                "format": "parquet",
                "sha256": source_hash,
                "bytes": source_artifact.stat().st_size,
                "rows": 1,
            },
            "columns": {
                "smiles": "smiles",
                "aliases": ["alias"],
                "inchikey": "inchikey",
                "source_record_id": "source_record_id",
            },
        })

    registry = source_dir / "source_registry.json"
    registry.write_text(json.dumps({
        "schema_version": "discovery_alias_source_registry.v1",
        "sources": registry_sources,
        "registry_root": ".",
    }, sort_keys=True) + "\n")
    missing_cids_audit = source_dir / "pubchem_missing_cids.txt"
    missing_cids_audit.write_text("", encoding="utf-8")
    source_manifest_unsigned = {
        "schema_version": "skinscout.discovery-alias-sources.v1",
        "builder": {
            "path": "scripts/build_discovery_alias_sources.py",
            "sha256": hashlib.sha256(
                (ROOT / "scripts" / "build_discovery_alias_sources.py").read_bytes()
            ).hexdigest(),
        },
        "created_at_utc": "1970-01-01T00:00:00Z",
        "inputs": source_inputs,
        "output_sha256": output_hashes,
        "output_bytes": output_bytes,
        "output_rows": output_rows,
        "audit_artifacts": {
            missing_cids_audit.name: {
                "sha256": hashlib.sha256(missing_cids_audit.read_bytes()).hexdigest(),
                "bytes": missing_cids_audit.stat().st_size,
                "rows": 0,
            }
        },
        "source_stats": {
            "BindingDB": {
                "input_rows": 1,
                "missing_smiles_rows": 0,
                "eligible_rows": 1,
            },
            "PubChem": {
                "selected_cids": 1,
                "matched_cids": 1,
                "missing_cids": 0,
                "missing_fraction_ppm": 0,
            },
        },
        "source_registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "source_registry": "source_registry.json",
        "policy": {
            "fail_closed": True,
            "PubChem_max_missing_fraction_ppm": 100_000,
        },
    }
    source_manifest = {
        **source_manifest_unsigned,
        "self_binding_sha256": hashlib.sha256(
            json.dumps(
                source_manifest_unsigned,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    source_manifest_path = source_dir / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True) + "\n"
    )
    builder = ROOT / "scripts" / "build_discovery_alias_map.py"
    outputs = {
        "aliases_parquet_sha256": hashlib.sha256(
            aliases.read_bytes()
        ).hexdigest(),
        "direct_reference_sha256": hashlib.sha256(
            direct.read_bytes()
        ).hexdigest(),
    }
    payload = {
        "schema_version": "discovery_alias_map.v1",
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            source_manifest_path.read_bytes()
        ).hexdigest(),
        "source_artifact_sha256": source_hashes,
        "outputs": outputs,
        "counts": {
            "alias_rows": 1,
            "direct_reference_rows": 1,
            "source_counts": {
                source_name: {
                    "input_rows": 1,
                    "canonicalized_rows": 1,
                    "canonicalization_exclusions": 0,
                    "canonicalization_exclusion_fraction_ppm": 0,
                }
                for source_name in source_names
            },
        },
        "canonicalization_audit": {
            "path": canonicalization_audit.name,
            "sha256": hashlib.sha256(canonicalization_audit.read_bytes()).hexdigest(),
            "bytes": canonicalization_audit.stat().st_size,
            "rows": 0,
        },
        "policy": {
            "fail_closed": True,
            "max_canonicalization_exclusion_fraction_ppm_per_source": 10_000,
        },
        "canonical_contract": {"pipeline": ["test"]},
        "builder_script_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "source_registry": os.path.relpath(registry, alias_dir),
        "source_manifest": os.path.relpath(source_manifest_path, alias_dir),
        "output_paths": {
            "aliases_parquet": aliases.name,
            "direct_reference": direct.name,
        },
        "output_bytes": {
            "aliases_parquet": aliases.stat().st_size,
            "direct_reference": direct.stat().st_size,
        },
    }
    canonical = {
        key: payload[key]
        for key in (
            "schema_version",
            "registry_sha256",
            "source_manifest_sha256",
            "source_artifact_sha256",
            "outputs",
            "counts",
            "canonicalization_audit",
            "policy",
            "canonical_contract",
            "builder_script_sha256",
        )
    }
    payload["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    payload["binding_sha256"] = leakage_check._payload_binding_sha256(payload)
    (alias_dir / "manifest.json").write_text(json.dumps(payload) + "\n")
    return payload


def verifier_stdout(
    status: str = "ok",
    *,
    run_dir: str | Path = TEST_RESULTS_ROOT / "test",
    preset: str = "target-id",
    mode: str = "comprehensive",
    checks: list[dict[str, object]] | None = None,
) -> str:
    check_status = "ok" if status == "ok" else "failed"
    run_dir = Path(run_dir)
    if checks is None:
        checks = [
            {
                "name": "user-facing run summary",
                "status": check_status,
                "path": str(run_dir / "run_summary.json"),
                "detail": "" if status == "ok" else "missing required field",
            }
        ]
    return json.dumps(
        {
            "schema_version": "skinscout.run_output_verification.v1",
            "status": status,
            "run_dir": str(run_dir),
            "preset": preset,
            "mode": mode,
            "checks": checks,
        }
    ) + "\n"


def verifier_text(
    status: str = "ok",
    *,
    run_dir: str | Path = TEST_RESULTS_ROOT / "test",
    preset: str = "target-id",
    mode: str = "comprehensive",
) -> str:
    check_status = "ok" if status == "ok" else "failed"
    detail = " - missing required field" if status != "ok" else ""
    run_dir = Path(run_dir)
    return (
        f"SkinScout run output verification: {status}\n"
        f"run_dir={run_dir} preset={preset} mode={mode}\n"
        f"[{check_status}] user-facing run summary: "
        f"{run_dir / 'run_summary.json'}{detail}\n"
    )


def write_verifier_json_out(cmd: list[str], status: str = "ok") -> None:
    run_dir = TEST_RESULTS_ROOT / "test"
    preset = "target-id"
    mode = "comprehensive"
    if "--run-dir" in cmd:
        run_dir = Path(cmd[cmd.index("--run-dir") + 1])
        preset = (
            cmd[cmd.index("--preset") + 1]
            if "--preset" in cmd
            else "target-id"
        )
        mode = cmd[cmd.index("--mode") + 1] if "--mode" in cmd else "comprehensive"
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            write_run_summary(
                summary_path,
                preset=preset,
                write_verification=False,
            )
    if "--json-out" not in cmd:
        return
    path = Path(cmd[cmd.index("--json-out") + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        verifier_stdout(
            status,
            run_dir=run_dir,
            preset=preset,
            mode=mode,
        )
    )


def run_artifact_fingerprint(path: Path, name: str) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "name": name,
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def smiles_input_provenance(
    input_smiles: str,
    canonical_smiles: str | None = None,
) -> dict[str, object]:
    return {
        "input_type": "smiles",
        "input_smiles": input_smiles,
        "input_canonical_smiles": canonical_smiles or input_smiles,
        "input_sdf": None,
    }


def sdf_input_provenance(path: Path) -> dict[str, object]:
    return {
        "input_type": "sdf",
        "input_smiles": None,
        "input_canonical_smiles": None,
        "input_sdf": str(path),
    }


def summary_input_provenance(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        compound = payload.get("compound") if isinstance(payload, dict) else None
        if isinstance(compound, dict):
            return {
                "input_type": compound.get("input_type"),
                "input_smiles": compound.get("input_smiles"),
                "input_canonical_smiles": compound.get("input_canonical_smiles"),
                "input_sdf": compound.get("input_sdf"),
            }
    return smiles_input_provenance("CCO")


def verification_command(
    run_dir: Path,
    *,
    preset: str = "target-id",
    mode: str = "fast",
    allow_degraded: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/verify_run_outputs.py",
        "--run-dir",
        str(run_dir),
        "--preset",
        preset,
        "--mode",
        mode,
        "--target-metadata",
        "data/hpa/proteinatlas.tsv",
        "--json-out",
        str(run_dir / ".run_verification_payload.json"),
    ]
    if allow_degraded:
        cmd.append("--allow-degraded")
    return cmd


def write_run_verification(
    run_dir: Path,
    *,
    preset: str = "target-id",
    mode: str = "fast",
    status: str = "ok",
    input_provenance: dict[str, object] | None = None,
    diagnostic_reasons: list[str] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if input_provenance is None:
        input_provenance = summary_input_provenance(run_dir)
    diagnostic_reasons = diagnostic_reasons or []
    allow_degraded = (
        "explicit degraded ADMET/skin-sens evidence was allowed"
        in diagnostic_reasons
    )
    checks = [
        {
            "name": "user-facing run summary",
            "status": "ok" if status == "ok" else "failed",
            "path": str(run_dir / "run_summary.json"),
            "detail": "",
        }
    ]
    command = verification_command(
        run_dir,
        preset=preset,
        mode=mode,
        allow_degraded=allow_degraded,
    )
    (run_dir / "run_verification.log").write_text(
        "\n".join([
            "$ " + shlex.join(command),
            "",
            "[stdout]",
            f"SkinScout run output verification: {status}",
            f"run_dir={run_dir} preset={preset} mode={mode}",
            (
                f"[{'ok' if status == 'ok' else 'failed'}] "
                f"user-facing run summary: {run_dir / 'run_summary.json'}"
            ),
            "",
            "[stderr]",
            "",
        ])
    )
    path = run_dir / "run_verification.json"
    path.write_text(json.dumps({
        "schema_version": "skinscout.run_verification.v1",
        "status": status,
        "verified_at_utc": "2026-06-26T00:00:00Z",
        "command": command,
        "returncode": 0 if status == "ok" else 2,
        "stdout_log": "run_verification.log",
        "preset": preset,
        "mode": mode,
        "run_dir": str(run_dir),
        "input_provenance": input_provenance,
        "diagnostic_nonclaimable_reasons": diagnostic_reasons,
        "verifier_status": status,
        "checks": checks,
        "verifier_payload": {
            "schema_version": "skinscout.run_output_verification.v1",
            "status": status,
            "run_dir": str(run_dir),
            "preset": preset,
            "mode": mode,
            "checks": checks,
        },
        "verified_artifacts": [
            run_artifact_fingerprint(
                run_dir / "run_summary.json",
                "run_summary_json",
            )
            if (run_dir / "run_summary.json").exists()
            else {
                "name": "run_summary_json",
                "path": str(run_dir / "run_summary.json"),
                "bytes": 1,
                "sha256": "0" * 64,
            },
            run_artifact_fingerprint(
                run_dir / "run_summary.md",
                "run_summary_md",
            )
            if (run_dir / "run_summary.md").exists()
            else {
                "name": "run_summary_md",
                "path": str(run_dir / "run_summary.md"),
                "bytes": 1,
                "sha256": "0" * 64,
            },
            run_artifact_fingerprint(
                run_dir / "run_verification.log",
                "run_verification_log",
            ),
            *(
                [run_artifact_fingerprint(
                    run_dir / "03_targets/ranked_targets_v3_with_efficacy.csv",
                    "target_ranking",
                )]
                if preset in {"target-id", "report"}
                and (run_dir / "03_targets/ranked_targets_v3_with_efficacy.csv").exists()
                else []
            ),
            *(
                [
                    run_artifact_fingerprint(
                        run_dir / "09_report" / "index.html",
                        "html_report",
                    )
                ]
                if preset == "report"
                else []
            ),
        ],
    }))
    return path


def write_run_summary(
    path: Path,
    *,
    action: str = "review_before_claim",
    claimable: bool | None = None,
    preset: str = "target-id",
    write_verification: bool = True,
) -> None:
    decision = {
        "proceed": "PASS",
        "review_before_claim": "FLAG_HIGH",
        "stop_before_claim": "HALT",
    }[action]
    summary_claimable = decision == "PASS" if claimable is None else claimable
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "skinscout.run_summary.v1",
        "run_id": path.parent.name,
        "preset": preset,
        "mode": "fast",
        "compound": {
            "input_type": "smiles",
            "input_smiles": "CCO",
            "input_canonical_smiles": "CCO",
            "canonical_smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        },
        "overall_decision": {
            "decision": decision,
            "requires_human_review": decision != "PASS",
            "recommended_action": action,
            "claimable": summary_claimable,
            "reasons": [
                "Skin_Reaction ADMET risk requires review: moderate",
                "top binding target lacks direct skin context support: P10275",
            ],
        },
        "safety": {
            "skin_sens_evidence": [
                {
                    "model": "husspred",
                    "status": "ok",
                    "call": "negative",
                    "probability": 0.2,
                },
                {
                    "model": "stoptox",
                    "status": "ok",
                    "call": "negative",
                    "probability": 0.2,
                },
                {
                    "model": "pred_skin",
                    "status": "ok",
                    "call": "negative",
                    "probability": 0.2,
                },
            ],
            "admet_metrics": {
                "AMES": 0.12,
                "ClinTox": 0.03,
                "DILI": 0.55,
                "Skin_Reaction": 0.31,
                "hERG": 0.04,
                "LD50_Zhu": 1.1,
                "Solubility_AqSolDB": 1.2,
                "logP": -0.0014,
                "QED": 0.4068,
                "tpsa": 20.23,
            },
            "admet_risk_assessment": {
                "high_risk_endpoints": ["DILI"],
                "moderate_risk_endpoints": ["Skin_Reaction"],
            },
        },
        "skin_toxicity": {
            "decision": "PASS",
            "toxicity_level": "low",
            "skin_reaction_risk_level": "moderate",
            "skin_reaction_value": 0.31,
            "structural_alerts_present": True,
            "structural_alert_flags": ["BRENK"],
            "degraded": False,
            "missing_models": [],
        },
        "cosmetic_drug": {
            "decision": "PROCEED",
            "drug_policy": "moderate",
            "n_warnings": 1,
        },
        "target_prediction": {
            "ranking_path": "03_targets/ranked_targets_v3_with_efficacy.csv",
            "n_targets": 20171,
            "screened_target_count": 20171,
            "screening_counts": {
                "psichic_proteome_targets": 20171,
                "daina_zoete_targets": 2943,
                "dti_rrf_candidates": 732,
                "autodock_rescored_targets": 256,
                "rerank_consensus_targets": 10,
                "skin_weighted_ranked_targets": 2,
            },
            "top_n": 2,
            "top_targets": [
                {
                    "target_id": "P10275",
                    "gene_symbol": "AR",
                    "protein_name": "Androgen receptor",
                    "final_score": 0.87,
                    "docking_rrf": 0.41,
                    "source_count": 3,
                    "sources": ["autodock", "gnina", "rtmscore"],
                    "skin_score": 0.28,
                    "skin_tier": "medium",
                    "efficacy": ["barrier repair (2 papers)"],
                },
                {
                    "target_id": "P04040",
                    "gene_symbol": "CAT",
                    "protein_name": "Catalase",
                    "final_score": 0.71,
                    "docking_rrf": 0.22,
                    "source_count": 2,
                    "sources": ["psichic", "autodock"],
                    "skin_score": 0.32,
                    "skin_tier": "medium",
                    "efficacy": ["oxidative stress (1 paper)"],
                }
            ],
        },
        "skin_specialized_binding": {
            "skin_context_decision": "skin_efficacy_literature_only",
            "skin_context_supported": False,
            "skin_expression_supported": False,
            "skin_efficacy_supported": True,
            "top_target_gene_symbol": "AR",
            "top_target_protein_name": "Androgen receptor",
            "top_target_final_score": 0.87,
            "top_target_docking_rrf": 0.41,
            "top_target_source_count": 3,
            "top_target_sources": ["autodock", "gnina", "rtmscore"],
            "top_target_skin_score": 0.28,
            "top_target_skin_tier": "medium",
            "top_target_skin_expression_supported": False,
            "top_target_skin_efficacy_supported": True,
            "top_target_skin_context_supported": False,
            "top_targets_with_skin_efficacy": 3,
            "top_target_efficacy": ["barrier repair (2 papers)"],
            "most_skin_relevant_target": {
                "target_id": "P10275",
                "gene_symbol": "AR",
                "protein_name": "Androgen receptor",
                "final_score": 0.87,
                "docking_rrf": 0.41,
                "source_count": 3,
                "sources": ["autodock", "gnina", "rtmscore"],
                "skin_score": 0.28,
                "skin_tier": "medium",
                "efficacy": ["barrier repair (2 papers)"],
            },
        },
        "artifacts": {
            "compound": "01_input/compound_canonical.json",
            "admet_report": "02_admet/admet_report.json",
            "admet_ai": "02_admet/admet_ai.json",
            "husspred": "02_admet/husspred.json",
            "stoptox": "02_admet/stoptox.json",
            "pred_skin": "02_admet/pred_skin.json",
            **({"html_report": "09_report/index.html"} if preset == "report" else {}),
            "target_ranking": "03_targets/ranked_targets_v3_with_efficacy.csv",
            "target_fast_psichic": "03_targets/mode_fast/psichic_proteome.tsv",
            "target_fast_rerank_consensus": "03_targets/mode_fast/top50.csv",
        },
    }))
    (path.parent / "run_summary.md").write_text(
        "\n".join([
            "# SkinScout Run Summary",
            "",
            f"- Run ID: {path.parent.name}",
            f"- Preset: {preset}",
            "- Mode: fast",
        ])
        + "\n"
    )
    if preset in {"target-id", "report"}:
        ranking = path.parent / "03_targets/ranked_targets_v3_with_efficacy.csv"
        ranking.parent.mkdir(parents=True, exist_ok=True)
        ranking.write_text("target_id,final_rank,final_score\nP10275,1,0.87\n")
    if write_verification:
        if preset == "report":
            report_path = path.parent / "09_report" / "index.html"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("<html><body>SkinScout report</body></html>\n")
        write_run_verification(path.parent, preset=preset)


def test_print_command_maps_smiles_to_target_id_workflow() -> None:
    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_case",
            "--preset",
            "target-id",
            "--mode",
            "comprehensive",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "snakemake -s workflow/Snakefile" in res.stdout
    assert "--use-conda" in res.stdout
    assert "kg_efficacy_label" in res.stdout
    assert "disagreement_analysis" in res.stdout
    assert "compound_smiles=CCO" in res.stdout
    assert "run_id=ethanol_case" in res.stdout
    assert "mode=comprehensive" in res.stdout


def test_print_command_target_id_sota_alias_disables_label_priors() -> None:
    parse_config = pytest.importorskip("snakemake.cli").parse_config

    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "ethanol_sota_case",
            "--preset",
            "target-id-sota",
            "--mode",
            "fast",
            "--context-profile",
            "barrier",
        ]
    )
    cmd = runner.build_snakemake_command(args, validate_conda_frontend=False)
    config_entries = cmd[cmd.index("--config") + 1 :]
    parsed = parse_config(config_entries)

    assert args.preset == "target-id"
    assert args.requested_preset == "target-id-sota"
    assert args.sota_claim is True
    assert args.run_profile.to_dict() == {
        "requested_preset": "target-id-sota",
        "normalized_preset": "target-id",
        "execution_mode": "fast",
        "analysis_profile": "target_fast",
        "evidence_mode": "evidence",
        "context_profile": "barrier",
        "sota_claim": True,
    }
    assert "kg_efficacy_label" in cmd
    assert "compound_smiles=CCO" in cmd
    assert "run_id=ethanol_sota_case" in cmd
    assert "mode=fast" in cmd
    evaluation = next(item for item in cmd if item.startswith("evaluation="))
    assert "'enabled': true" in evaluation
    assert "'default_context_profile': 'barrier'" in evaluation
    assert "'skin_known_min_case_top10': 0.80" in evaluation
    known_priors = parsed["known_target_priors"]
    assert isinstance(known_priors, dict)
    assert str(known_priors["enabled"]).lower() == "false"
    assert str(known_priors["weight"]) == "0.0"
    assert str(known_priors["similarity_enabled"]).lower() == "false"
    assert str(parsed["skin_weight"]) == "0.4"
    assert str(parsed["skin_expression"]["require_gtex_source"]).lower() == "true"
    assert str(parsed["skin_expression"]["require_proteome_source"]).lower() == "true"
    assert str(parsed["skin_min_threshold"]) == "0.06"


def test_discovery_forced_config_overrides_user_and_sota_entries(
    tmp_path: Path,
) -> None:
    parse_config = pytest.importorskip("snakemake.cli").parse_config

    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "forced_discovery_case",
            "--preset",
            "target-id-sota",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
            "--extra-config",
            "known_target_priors.enabled=true",
            "--extra-config",
            "known_target_priors.weight=1.0",
            "--extra-config",
            "known_target_priors.similarity_enabled=true",
            "--extra-config",
            "docking.daina_evidence_mode=retrieval",
            "--discovery-alias-dir",
            str(alias_dir),
        ]
    )
    cmd = runner.build_snakemake_command(args, validate_conda_frontend=False)
    config_entries = cmd[cmd.index("--config") + 1 :]
    parsed = parse_config(config_entries)

    known_priors = parsed["known_target_priors"]
    assert parsed["evidence_mode"] == "discovery"
    assert str(known_priors["enabled"]).lower() == "false"
    assert str(known_priors["weight"]) == "0.0"
    assert str(known_priors["similarity_enabled"]).lower() == "false"
    assert parsed["docking"]["daina_evidence_mode"] == "leave-query-out"


def test_launcher_writes_idempotent_run_manifest_v3(tmp_path: Path) -> None:
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "manifest_case",
            "--preset",
            "target-id-sota",
            "--mode",
            "fast",
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )
    command = runner.build_snakemake_command(
        args,
        validate_conda_frontend=False,
    )

    path = runner._ensure_run_manifest(args, command)
    first = path.read_bytes()
    assert runner._ensure_run_manifest(args, command) == path
    assert path.read_bytes() == first

    payload = json.loads(first)
    assert payload["schema_version"] == "skinscout.run_manifest.v3"
    assert payload["run_profile"]["analysis_profile"] == "target_fast"
    assert payload["run_profile"]["sota_claim"] is True
    assert payload["recipe_id"] == "daina-structural-overlay-v1"
    assert payload["target_fast"]["schema_version"] == "skinscout.target_fast.v2"
    assert (
        payload["target_fast"]["artifact_id"]
        == "target_fast_daina_structural_targets"
    )
    assert len(payload["hashes"]["config_sha256"]) == 64
    assert len(payload["hashes"]["data_sha256"]) == 64


def test_legacy_summary_migration_does_not_claim_current_data_hash(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "legacy_manifest_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )
    legacy_summary = {
        "schema_version": "skinscout.run_summary.v1",
        "run_id": "legacy_manifest_case",
        "preset": "target-id",
        "mode": "fast",
    }

    path = runner._ensure_run_manifest(args, legacy_summary=legacy_summary)
    payload = json.loads(path.read_text())

    assert payload["hashes"]["data_sha256"] is None
    assert runner._ensure_run_manifest(
        args,
        legacy_summary=legacy_summary,
    ) == path


def test_launcher_rejects_stale_data_manifest_hash(tmp_path: Path) -> None:
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "stale_data_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )
    path = runner._ensure_run_manifest(args)
    payload = json.loads(path.read_text())
    payload["hashes"]["data_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="data_sha256"):
        runner._ensure_run_manifest(args)


def test_launcher_rejects_stale_legacy_migration_hash(tmp_path: Path) -> None:
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "stale_legacy_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )
    legacy_summary = {
        "schema_version": "skinscout.run_summary.v1",
        "run_id": "stale_legacy_case",
        "preset": "target-id",
        "mode": "fast",
    }
    path = runner._ensure_run_manifest(args, legacy_summary=legacy_summary)
    payload = json.loads(path.read_text())
    payload["hashes"]["config_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="config_sha256"):
        runner._ensure_run_manifest(args, legacy_summary=legacy_summary)


def test_discovery_execution_requires_sealed_alias_package(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Direct exact manifest is missing"):
        run_discovery_preflight([
            "--smiles",
            "CCO",
            "--run-id",
            "missing_alias_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
            "--discovery-alias-dir",
            str(tmp_path / "missing_aliases"),
        ])


def test_discovery_print_command_skips_preflight_and_writes_nothing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "print_only_case"
    res = run_runner([
        "--smiles",
        "CCO",
        "--run-id",
        "print_only_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--evidence-mode",
        "discovery",
        "--extra-config",
        f"paths.results_root={tmp_path}",
        "--discovery-alias-dir",
        str(tmp_path / "missing_aliases"),
        "--print-command",
    ])

    assert res.returncode == 0, res.stderr
    assert "evidence_mode=discovery" in res.stdout
    assert not run_dir.exists()


def test_discovery_execution_accepts_unseen_compound_with_direct_preflight(
    tmp_path: Path,
) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    run_dir = tmp_path / "unseen_case"

    runner = run_discovery_preflight(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "unseen_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
            "--discovery-alias-dir",
            str(alias_dir),
        ]
    )
    payload = json.loads(
        (run_dir / "discovery_direct_preflight.json").read_text()
    )
    assert payload["schema_version"] == "skinscout.discovery-direct-preflight.v1"
    assert payload["run_id"] == "unseen_case"
    assert payload["direct_exact_match"] is False
    assert not (run_dir / "discovery_leakage_audit.json").exists()

    identity = runner.discovery_key("CCO", label="test Discovery input")
    runner._validate_existing_discovery_direct_preflight(
        args=Namespace(discovery_alias_dir=alias_dir),
        run_dir=run_dir,
        expected_input_key=identity.discovery_key_sha256,
        parent_canonical_smiles=identity.parent_canonical_smiles,
    )


def test_discovery_execution_rejects_exact_known_compound(tmp_path: Path) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCO")

    with pytest.raises(SystemExit, match="Use Evidence mode for known compounds"):
        run_discovery_preflight(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "known_case",
                "--preset",
                "target-id",
                "--mode",
                "fast",
                "--evidence-mode",
                "discovery",
                "--extra-config",
                f"paths.results_root={tmp_path}",
                "--discovery-alias-dir",
                str(alias_dir),
            ]
        )


def test_discovery_existing_preflight_rejects_alias_drift(tmp_path: Path) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    common = [
        "--smiles",
        "CCO",
        "--run-id",
        "drift_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--evidence-mode",
        "discovery",
        "--extra-config",
        f"paths.results_root={tmp_path}",
        "--discovery-alias-dir",
        str(alias_dir),
    ]
    run_discovery_preflight(common)
    (alias_dir / "direct_exact_reference.smi").write_text(
        "CCCC " + hashlib.sha256(b"CCCC").hexdigest() + "\n"
    )

    verified = run_runner([*common, "--verify-existing-run"])

    assert verified.returncode != 0
    assert "Direct exact reference" in verified.stderr
    assert "drift" in verified.stderr


def test_discovery_execution_rejects_stale_upstream_alias_manifest(
    tmp_path: Path,
) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    upstream = alias_dir / "upstream" / "chembl_manifest.json"
    upstream.write_text(upstream.read_text() + "\n")

    with pytest.raises(SystemExit, match="upstream manifest SHA-256 drift"):
        run_discovery_preflight(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "stale_alias_source_case",
                "--preset",
                "target-id",
                "--mode",
                "fast",
                "--evidence-mode",
                "discovery",
                "--extra-config",
                f"paths.results_root={tmp_path}",
                "--discovery-alias-dir",
                str(alias_dir),
            ]
        )


def test_discovery_execution_rejects_tampered_missing_cid_audit(
    tmp_path: Path,
) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    (alias_dir / "sources" / "pubchem_missing_cids.txt").write_text(
        "123\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing-CID audit hash/bytes drift"):
        run_discovery_preflight(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "tampered_missing_cid_audit_case",
                "--preset",
                "target-id",
                "--mode",
                "fast",
                "--evidence-mode",
                "discovery",
                "--extra-config",
                f"paths.results_root={tmp_path}",
                "--discovery-alias-dir",
                str(alias_dir),
            ]
        )


def test_discovery_fresh_launch_ignores_stale_post_run_audit(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "discovery_case"
    run_dir.mkdir()
    payload = write_valid_discovery_audit(run_dir, "CCO")
    direct_record = payload["inputs"]["direct_exact_reference"]
    Path(direct_record["path"]).write_text("stale changed source\n")
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")

    run_discovery_preflight(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "discovery_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
            "--discovery-alias-dir",
            str(alias_dir),
        ]
    )
    assert (run_dir / "discovery_direct_preflight.json").is_file()
    assert not (run_dir / "discovery_audit_acceptance.json").exists()


def test_discovery_existing_run_rejects_preflight_copied_from_another_run(
    tmp_path: Path,
) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    common = [
        "--smiles",
        "CCO",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--evidence-mode",
        "discovery",
        "--extra-config",
        f"paths.results_root={tmp_path}",
        "--discovery-alias-dir",
        str(alias_dir),
    ]
    run_discovery_preflight([*common, "--run-id", "old_run"])
    new_run = tmp_path / "new_run"
    new_run.mkdir()
    old_preflight = tmp_path / "old_run" / "discovery_direct_preflight.json"
    (new_run / "discovery_direct_preflight.json").write_text(
        old_preflight.read_text()
    )

    verified = run_runner(
        [*common, "--run-id", "new_run", "--verify-existing-run"]
    )

    assert verified.returncode != 0
    assert "run_id mismatch" in verified.stderr


def test_discovery_existing_run_rejects_preflight_binding_tamper(
    tmp_path: Path,
) -> None:
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")
    common = [
        "--smiles",
        "CCO",
        "--run-id",
        "tamper_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--evidence-mode",
        "discovery",
        "--extra-config",
        f"paths.results_root={tmp_path}",
        "--discovery-alias-dir",
        str(alias_dir),
    ]
    run_discovery_preflight(common)
    preflight = tmp_path / "tamper_case" / "discovery_direct_preflight.json"
    payload = json.loads(preflight.read_text())
    payload["parent_canonical_smiles"] = "tampered"
    preflight.write_text(json.dumps(payload) + "\n")

    verified = run_runner([*common, "--verify-existing-run"])

    assert verified.returncode != 0
    assert "binding mismatch" in verified.stderr


def test_discovery_existing_run_rejects_minimal_legacy_audit(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(ROOT / "eval"))
    from discovery_canonical import discovery_key

    key = discovery_key("CCO")
    run_dir = tmp_path / "legacy_case"
    run_dir.mkdir()
    (run_dir / "discovery_leakage_audit.json").write_text(json.dumps({
        "schema_version": "skinscout.discovery-leakage-audit.v1",
        "status": "ok",
        "direct_exact_count": 0,
        "counts": {"survivor_rows": 1},
        "survivors": [{"discovery_key_sha256": key.discovery_key_sha256}],
    }))

    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "legacy_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
            "--verify-existing-run",
        ]
    )

    assert res.returncode != 0
    assert "schema validation failed" in res.stderr


def test_discovery_execution_binds_current_sdf_input(tmp_path: Path) -> None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    sdf = tmp_path / "ethanol.sdf"
    sdf.write_text(Chem.MolToMolBlock(mol) + "\n$$$$\n")
    alias_dir = tmp_path / "aliases"
    write_valid_discovery_alias_package(alias_dir, "CCC")

    res = run_runner(
        [
            "--sdf",
            str(sdf),
            "--run-id",
            "sdf_discovery_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
            "--discovery-alias-dir",
            str(alias_dir),
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert f"compound_sdf={sdf}" in res.stdout


def test_discovery_daina_metadata_requires_leave_query_out(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    run_dir = tmp_path / "daina_metadata_case"
    metadata = run_dir / "03_targets" / "mode_fast" / "daina_zoete_proteome.metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({
        "schema_version": "skinscout.daina-run.v1",
        "evidence_mode": "retrieval",
    }) + "\n")
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "daina_metadata_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )

    with pytest.raises(SystemExit, match="must be leave-query-out"):
        runner._validate_discovery_daina_metadata(args)


def test_discovery_daina_metadata_accepts_leave_query_out(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    run_dir = tmp_path / "daina_metadata_ok_case"
    ligand_sdf = run_dir / "01_qm" / "optimized.sdf"
    ligand_sdf.parent.mkdir(parents=True)
    ligand_sdf.write_text("test ligand\n")
    chembl_dir = tmp_path / "chembl37"
    chembl_dir.mkdir()
    fingerprints = chembl_dir / "fp_morgan2_2048.parquet"
    activities = chembl_dir / "human_activities.parquet"
    fingerprints.write_bytes(b"fingerprints")
    activities.write_bytes(b"activities")
    fingerprints_sha256 = hashlib.sha256(fingerprints.read_bytes()).hexdigest()
    activities_sha256 = hashlib.sha256(activities.read_bytes()).hexdigest()
    source_manifest = chembl_dir / "source_manifest.json"
    source_manifest.write_text(json.dumps({
        "schema_version": "chembl_activity_evidence.v1",
        "output_sha256": {"human_activities.parquet": activities_sha256},
    }) + "\n")
    (chembl_dir / "fingerprint_manifest.json").write_text(json.dumps({
        "schema_version": "chembl_fingerprint_snapshot.v1",
        "artifact": {"sha256": fingerprints_sha256},
        "input": {"sha256": activities_sha256},
        "source_snapshot": {
            "manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        },
    }) + "\n")
    metadata = run_dir / "03_targets" / "mode_fast" / "daina_zoete_proteome.metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({
        "schema_version": "skinscout.daina-run.v1",
        "evidence_mode": "leave-query-out",
        "evidence_snapshot_id": str(source_manifest),
        "cutoff_date": None,
        "exclude_reference_similarity": 0.85,
        "score_is_calibrated_probability": False,
        "inputs": {
            "ligand_sdf": str(ligand_sdf),
            "ligand_sdf_sha256": hashlib.sha256(ligand_sdf.read_bytes()).hexdigest(),
            "fingerprints": str(fingerprints),
            "fingerprints_sha256": fingerprints_sha256,
            "activities": str(activities),
            "activities_sha256": activities_sha256,
        },
        "counts": {
            "initial_activity_rows": 12,
            "pre_similarity_filter_activity_rows": 10,
            "scored_activity_rows": 7,
            "excluded_activity_rows_total": 3,
            "ranked_targets": 2,
        },
    }) + "\n")
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "daina_metadata_ok_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--evidence-mode",
            "discovery",
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )

    runner._validate_discovery_daina_metadata(args)
    activities.write_bytes(b"tampered activities")
    with pytest.raises(SystemExit, match="activities_sha256 mismatch"):
        runner._validate_discovery_daina_metadata(args)


def test_print_command_inflammation_profile_applies_targeted_docking_tuning() -> None:
    parse_config = pytest.importorskip("snakemake.cli").parse_config

    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "inflammation_case",
            "--preset",
            "target-id-sota",
            "--mode",
            "fast",
            "--context-profile",
            "inflammation",
        ]
    )
    cmd = runner.build_snakemake_command(args, validate_conda_frontend=False)
    config_entries = cmd[cmd.index("--config") + 1 :]
    parsed = parse_config(config_entries)

    assert str(parsed["skin_weight"]) == "0.37"
    assert str(parsed["skin_min_threshold"]) == "0.05"


def test_print_command_autogenerates_run_id_from_smiles() -> None:
    runner = load_runner_module()
    expected_run_id = runner._auto_run_id_from_smiles("CCO")

    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "compound_smiles=CCO" in res.stdout
    assert f"run_id={expected_run_id}" in res.stdout


def test_print_command_preserves_input_smiles_for_stage1() -> None:
    runner = load_runner_module()
    expected_run_id = runner._auto_run_id_from_smiles("C(C)O")

    res = run_runner(
        [
            "--smiles",
            "C(C)O",
            "--preset",
            "safety",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "compound_smiles=C(C)O" in res.stdout
    assert f"run_id={expected_run_id}" in res.stdout


def test_print_command_accepts_positional_smiles() -> None:
    runner = load_runner_module()
    expected_run_id = runner._auto_run_id_from_smiles("CCO")

    res = run_runner(
        [
            "CCO",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "compound_smiles=CCO" in res.stdout
    assert f"run_id={expected_run_id}" in res.stdout


def test_print_command_rejects_positional_smiles_with_flagged_input() -> None:
    res = run_runner(
        [
            "CCO",
            "--smiles",
            "CCC",
            "--print-command",
        ]
    )

    assert res.returncode != 0
    assert "positional SMILES cannot be combined" in res.stderr


def test_snakemake_command_keeps_targets_out_of_resources() -> None:
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--run-id",
            "stage0_bootstrap",
            "--preset",
            "stage0",
            "--print-command",
        ]
    )

    cmd = runner.build_snakemake_command(args, validate_conda_frontend=False)

    assert cmd.index("stage0_complete") < cmd.index("--resources")
    assert cmd[cmd.index("--resources") + 1] == "gpu=1"
    assert cmd.index("--resources") < cmd.index("--config")


def test_print_command_safety_preset_targets_admet_and_cosmetic_gates() -> None:
    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "safety_case",
            "--preset",
            "safety",
            "--mode",
            "fast",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "skin_sens_consensus" in res.stdout
    assert "cosmetic_drug_decision" in res.stdout
    assert "kg_efficacy_label" not in res.stdout


def test_print_command_safety_degraded_adds_explicit_config() -> None:
    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "safety_degraded_case",
            "--preset",
            "safety",
            "--allow-safety-degraded",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "admet=" in res.stdout
    assert "allow_admet_ai_unavailable" in res.stdout
    assert "allow_skin_sens_unavailable" in res.stdout


def test_extra_config_dotted_keys_are_nested_for_snakemake() -> None:
    parse_config = pytest.importorskip("snakemake.cli").parse_config

    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "nested_config_case",
            "--preset",
            "target-id",
            "--mode",
            "fast",
            "--extra-config",
            "docking.fast_mode_autogrid_spacing=0.5",
            "--extra-config",
            "docking.autodock_runs=1",
            "--print-command",
        ]
    )

    cmd = runner.build_snakemake_command(args, validate_conda_frontend=False)
    config_entries = cmd[cmd.index("--config") + 1 :]

    assert not any(entry.startswith("docking.") for entry in config_entries)
    parsed = parse_config(config_entries)
    assert parsed["docking"]["autodock_runs"] == "1"
    assert parsed["docking"]["fast_mode_autogrid_spacing"] == "0.5"


@pytest.mark.parametrize("key", [
    "run_id", "mode", "compound_smiles", "compound_sdf", "evidence_mode",
    "run_dti_sanity", "target_metadata", "run_id.nested",
])
def test_extra_config_cannot_change_launcher_identity(key: str) -> None:
    runner = load_runner_module()
    args = runner.parse_args(["--smiles", "CCO", "--extra-config", f"{key}=other"])
    with pytest.raises(SystemExit, match="launcher-owned"):
        runner.build_snakemake_command(args, validate_conda_frontend=False)


@pytest.mark.parametrize("values", [
    ["paths.results_root=first", "paths.results_root=second"],
    ["docking.x=1", "docking.x=2"],
    ["docking.x=1", "docking.x.y=2"],
    ["paths={'results_root': 'other'}"],
    ["paths.results_root="],
])
def test_extra_config_rejects_ambiguous_path_and_key_overrides(values: list[str]) -> None:
    runner = load_runner_module()
    args = runner.parse_args(["--smiles", "CCO"])
    args.extra_config = values
    with pytest.raises(SystemExit, match="--extra-config"):
        runner.build_snakemake_command(args, validate_conda_frontend=False)


def test_manifest_digest_changes_with_workflow_defaults(tmp_path: Path, monkeypatch) -> None:
    runner = load_runner_module()
    (tmp_path / "workflow").mkdir()
    config = tmp_path / "workflow/config.yaml"
    config.write_text("threshold: 0.5\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    args = runner.parse_args(["--smiles", "CCO", "--preset", "safety"])
    before = runner._run_manifest_payload(args)
    config.write_text("threshold: 0.9\n")
    after = runner._run_manifest_payload(args)
    assert before["hashes"]["config_sha256"] != after["hashes"]["config_sha256"]


def test_environment_preparation_does_not_run_analysis(monkeypatch) -> None:
    runner = load_runner_module()
    captured = []
    monkeypatch.setattr(runner, "_resolve_conda_frontend", lambda *_a, **_kw: None)
    monkeypatch.setattr(runner, "_snakemake_environment", lambda _a: ({}, None))
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **_kw: captured.append(cmd) or Namespace(returncode=0))
    monkeypatch.setattr(runner, "_ensure_run_manifest", lambda *_a: pytest.fail("created an analysis manifest"))
    monkeypatch.setattr(runner, "_run_output_summary", lambda *_a: pytest.fail("attempted analysis summary"))
    assert runner.main(["--smiles", "CCO", "--mode", "fast", "--conda-create-envs-only"]) == 0
    assert len(captured) == 1
    assert "--conda-create-envs-only" in captured[0]


def test_band_rankings_are_hashed_and_tampering_is_rejected(tmp_path: Path) -> None:
    runner = load_runner_module()
    write_run_summary(tmp_path / "run_summary.json")
    fast = tmp_path / "03_targets/mode_fast"
    fast.mkdir(parents=True)
    (fast / "top50_band_reranked.csv").write_text("rank,target_id\n1,P12345\n")
    for _name, relative in runner._completed_verified_artifact_specs("target-id", tmp_path):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("verified data\n")
    artifacts, error = runner._completed_verified_artifact_snapshot(tmp_path, "target-id")
    assert error is None
    payload = {"verified_artifacts": artifacts}
    runner._validate_completed_verified_artifacts(payload, tmp_path, "target-id")
    (fast / "top50_band_reranked.csv").write_text("rank,target_id\n1,P54321\n")
    with pytest.raises(SystemExit, match="sha256 changed"):
        runner._validate_completed_verified_artifacts(payload, tmp_path, "target-id")


def test_runner_passes_safety_degraded_to_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    captured: dict[str, object] = {}

    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None

    def fake_safety_readiness(**kwargs: object) -> None:
        captured.update(kwargs)

    def fake_subprocess_run(*_args: object, **_kwargs: object) -> Namespace:
        return Namespace(returncode=0)

    runner._run_safety_readiness = fake_safety_readiness
    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    res = runner.main(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "safety_degraded_case",
            "--preset",
            "safety",
            "--allow-safety-degraded",
            "--dry-run",
            "--skip-data-readiness",
            "--skip-model-readiness",
        ]
    )

    assert res == 0
    assert captured["allow_degraded"] is True


def test_print_command_both_mode_target_id_includes_fast_and_comprehensive() -> None:
    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "both_case",
            "--preset",
            "target-id",
            "--mode",
            "both",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "kg_efficacy_label" in res.stdout
    assert "disagreement_analysis" in res.stdout
    assert "fast_rerank_consensus" in res.stdout


def test_stage0_print_command_does_not_require_compound_input() -> None:
    res = run_runner(
        [
            "--run-id",
            "stage0_bootstrap",
            "--preset",
            "stage0",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "stage0_complete" in res.stdout
    assert "compound_smiles=C" in res.stdout


def test_stage0_print_command_autogenerates_bootstrap_run_id() -> None:
    res = run_runner(
        [
            "--preset",
            "stage0",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "stage0_complete" in res.stdout
    assert "compound_smiles=C" in res.stdout
    assert "run_id=stage0_bootstrap" in res.stdout


def test_target_id_model_readiness_includes_autodock_conda_env() -> None:
    runner = load_runner_module()

    required = runner._required_model_readiness(
        "target-id",
        "comprehensive",
        True,
        use_conda=True,
    )

    assert "autodock_gpu_env" in required
    assert "meeko_env" in required
    assert "autodock_gpu" in required
    assert "meeko" not in required


def test_demo_model_readiness_uses_current_runtime_even_with_micromamba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/bin/{name}")

    command = runner._readiness_base_command(None, ["gnina", "autogrid"])

    assert command == [runner.sys.executable]


def test_advanced_model_readiness_uses_boltz_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/bin/{name}")

    command = runner._readiness_base_command(None, ["gnina", "boltz"])

    assert command == ["micromamba", "run", "-n", "cosmax-boltz2", "python"]


def test_real_target_run_requires_gpu_execution_admission() -> None:
    runner = load_runner_module()

    assert runner._requires_gpu_admission("target-id", dry_run=False) is True
    assert runner._requires_gpu_admission("report", dry_run=False) is True


def test_no_conda_advanced_readiness_checks_the_execution_interpreter(monkeypatch) -> None:
    runner = load_runner_module()
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/bin/{name}")
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: (
        calls.append(cmd) or Namespace(returncode=0, stdout="", stderr="")
    ))
    runner._run_model_readiness(["boltz", "psichic"], None, use_conda=False)
    assert calls[0][0] == runner.sys.executable
    assert "micromamba" not in calls[0]
    assert "--active-environment-only" in calls[0]
    with pytest.raises(SystemExit, match="current Python environment"):
        runner._run_model_readiness(["boltz"], "micromamba run -n other python", use_conda=False)


@pytest.mark.parametrize("returncode", [False, 0.0, "0", None, 1])
def test_completed_verification_rejects_noninteger_or_failed_execution(tmp_path, returncode) -> None:
    runner = load_runner_module()
    args = runner.parse_args(["--smiles", "CCO", "--run-id", "bad_returncode", "--preset", "target-id", "--mode", "fast"])
    write_run_summary(tmp_path / "run_summary.json")
    write_run_verification(tmp_path)
    record_path = tmp_path / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["returncode"] = returncode
    record_path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="returncode is not 0"):
        runner._read_completed_verification_record(args, tmp_path)


def test_target_dry_run_does_not_require_free_gpu_capacity() -> None:
    runner = load_runner_module()

    assert runner._requires_gpu_admission("target-id", dry_run=True) is False
    assert runner._requires_gpu_admission("safety", dry_run=False) is False


def test_gpu_admission_cannot_be_bypassed_by_unsafe_readiness_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    monkeypatch.setattr(
        runner.gpu_admission,
        "probe",
        lambda **_kwargs: {
            "available_for_analysis": False,
            "detail": "external GPU process active",
        },
    )

    with pytest.raises(SystemExit, match="우회할 수 없습니다"):
        runner._run_gpu_admission()


def test_target_id_no_conda_model_readiness_requires_active_autodock_tools() -> None:
    runner = load_runner_module()

    required = runner._required_model_readiness(
        "target-id",
        "comprehensive",
        True,
        use_conda=False,
    )

    assert "autodock_gpu" in required
    assert "meeko" in required
    assert "autodock_gpu_env" not in required
    assert "meeko_env" not in required


def test_report_model_readiness_includes_downstream_conda_envs() -> None:
    runner = load_runner_module()

    required = runner._required_model_readiness(
        "report",
        "comprehensive",
        True,
        use_conda=True,
    )

    assert "autodock_gpu_env" in required
    assert "meeko_env" in required
    assert "autodock_gpu" in required
    assert "bioemu_env" in required
    assert "md_env" in required
    assert "qm_env" in required
    assert "bioemu" not in required
    assert "gromacs" not in required
    assert "xtb" not in required


def test_report_no_conda_model_readiness_requires_active_downstream_tools() -> None:
    runner = load_runner_module()

    required = runner._required_model_readiness(
        "report",
        "comprehensive",
        True,
        use_conda=False,
    )

    assert "autodock_gpu" in required
    assert "meeko" in required
    assert "bioemu" in required
    assert "gromacs" in required
    assert "gmx_mmpbsa" in required
    assert "acpype" in required
    assert "xtb" in required
    assert "crest" in required
    assert "obabel" in required
    assert "pyscf" in required
    assert "mdtraj" in required
    assert "sklearn" in required
    assert "bioemu_env" not in required
    assert "md_env" not in required
    assert "qm_env" not in required


def test_data_readiness_reports_missing_stage0_artifacts(tmp_path: Path) -> None:
    runner = load_runner_module()
    missing = tmp_path / "missing.marker"
    runner._required_data_artifacts = lambda *_args: [("missing marker", missing)]

    with pytest.raises(SystemExit) as exc:
        runner._run_data_readiness(
            "target-id",
            "comprehensive",
            True,
            allow_stage0_build=False,
        )

    message = str(exc.value)
    assert "Data readiness preflight failed" in message
    assert str(missing) in message
    assert "--allow-stage0-build" in message


def test_data_readiness_calls_artifact_contract_with_supported_arguments(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    missing = tmp_path / "missing.marker"
    calls: list[tuple[str, str]] = []

    def required_artifacts(preset: str, mode: str) -> list[tuple[str, Path]]:
        calls.append((preset, mode))
        return [("missing marker", missing)]

    runner._required_data_artifacts = required_artifacts

    with pytest.raises(SystemExit):
        runner._run_data_readiness(
            "stage0",
            "comprehensive",
            True,
            allow_stage0_build=False,
        )

    assert calls == [("stage0", "comprehensive")]


def test_data_readiness_treats_empty_directory_as_missing(tmp_path: Path) -> None:
    runner = load_runner_module()
    empty_dir = tmp_path / "empty_boxes"
    empty_dir.mkdir()
    runner._required_data_artifacts = lambda *_args: [("empty boxes", empty_dir)]

    with pytest.raises(SystemExit) as exc:
        runner._run_data_readiness(
            "target-id",
            "comprehensive",
            True,
            allow_stage0_build=False,
        )

    assert "empty boxes" in str(exc.value)


def test_data_readiness_rejects_placeholder_reference_in_launcher(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    drugs = tmp_path / "drugs.parquet"
    pd.DataFrame(
        [
            {
                "drug_id": "PLACEHOLDER1",
                "name": "Aspirin",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "ecfp4": [0] * 32,
            }
        ]
    ).to_parquet(drugs)
    runner._required_data_artifacts = lambda *_args: [("drug reference", drugs)]

    with pytest.raises(SystemExit) as exc:
        runner._run_data_readiness(
            "safety",
            "comprehensive",
            True,
            allow_stage0_build=False,
        )

    message = str(exc.value)
    assert "drug reference [placeholder]" in message
    assert "drug_id=PLACEHOLDER1" in message


def test_stage0_build_flag_does_not_bypass_placeholder_reference(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    drugs = tmp_path / "drugs.parquet"
    pd.DataFrame(
        [
            {
                "drug_id": "PLACEHOLDER1",
                "name": "Aspirin",
                "smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "ecfp4": [0] * 32,
            }
        ]
    ).to_parquet(drugs)
    runner._required_data_artifacts = lambda *_args: [("drug reference", drugs)]

    with pytest.raises(SystemExit) as exc:
        runner._run_data_readiness(
            "target-id",
            "comprehensive",
            True,
            allow_stage0_build=True,
        )

    message = str(exc.value)
    assert "drug reference [placeholder]" in message
    assert "--allow-stage0-build only bypasses missing generated outputs" in message


def test_data_readiness_allows_explicit_stage0_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    runner._required_data_artifacts = lambda *_args: [
        ("missing marker", tmp_path / "missing.marker")
    ]
    runner._stage0_source_failures_for_build = lambda: []

    runner._run_data_readiness(
        "target-id",
        "comprehensive",
        True,
        allow_stage0_build=True,
    )
    captured = capsys.readouterr()
    assert "--allow-stage0-build was set" in captured.err


def test_stage0_build_flag_requires_source_readiness_for_generated_outputs(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    generated = tmp_path / "missing.marker"
    source = tmp_path / "cosing.csv"
    runner._required_data_artifacts = lambda *_args: [("missing marker", generated)]
    runner._stage0_source_failures_for_build = lambda: [
        ("Stage 0 source: CosIng CSV [missing]", source)
    ]

    with pytest.raises(SystemExit) as exc:
        runner._run_data_readiness(
            "target-id",
            "comprehensive",
            True,
            allow_stage0_build=True,
        )

    message = str(exc.value)
    assert "Stage 0 source readiness preflight failed" in message
    assert str(source) in message
    assert str(generated) not in message


def test_stage0_source_readiness_is_not_bypassed_by_stage0_build_flag(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    source = tmp_path / "cosing.csv"
    runner._required_data_artifacts = lambda *_args: [
        ("Stage 0 source: CosIng CSV", source)
    ]

    with pytest.raises(SystemExit) as exc:
        runner._run_data_readiness(
            "stage0",
            "comprehensive",
            True,
            allow_stage0_build=True,
        )

    message = str(exc.value)
    assert "Stage 0 source readiness preflight failed" in message
    assert str(source) in message


def test_runner_rejects_invalid_smiles() -> None:
    res = run_runner(
        [
            "--smiles",
            "not a smiles",
            "--run-id",
            "bad_smiles_case",
            "--print-command",
        ]
    )

    assert res.returncode != 0
    assert "Could not parse SMILES" in res.stderr


def test_runner_rejects_unsafe_run_id() -> None:
    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "../escape",
            "--print-command",
        ]
    )

    assert res.returncode != 0
    assert "run_id must use only letters" in res.stderr


def test_runner_requires_exactly_one_input(tmp_path: Path) -> None:
    sdf = tmp_path / "in.sdf"
    sdf.write_text(CAFFEINE_SDF)

    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--sdf",
            str(sdf),
            "--run-id",
            "duplicate_input_case",
            "--print-command",
        ]
    )

    assert res.returncode != 0
    assert "Provide exactly one of --smiles or --sdf" in res.stderr


def test_dry_run_invokes_configured_snakemake(tmp_path: Path) -> None:
    called = tmp_path / "called.txt"
    fake_snakemake = tmp_path / "snakemake"
    fake_snakemake.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(called)!r}).write_text(' '.join(sys.argv[1:]))\n"
    )
    fake_snakemake.chmod(0o755)

    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "dry_run_case",
            "--preset",
            "safety",
            "--dry-run",
            "--skip-data-readiness",
            "--skip-model-readiness",
            "--skip-safety-readiness",
            "--no-use-conda",
            "--snakemake",
            str(fake_snakemake),
        ]
    )

    assert res.returncode == 0, res.stderr
    args = called.read_text()
    assert "-n" in args
    assert "skin_sens_consensus" in args
    assert "cosmetic_drug_decision" in args
    assert "compound_smiles=CCO" in args


def test_runner_rejects_readiness_skip_for_real_execution(tmp_path: Path) -> None:
    fake_snakemake = tmp_path / "snakemake"
    fake_snakemake.write_text("#!/usr/bin/env sh\nexit 0\n")
    fake_snakemake.chmod(0o755)

    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "unsafe_skip_case",
            "--preset",
            "safety",
            "--skip-data-readiness",
            "--skip-model-readiness",
            "--skip-safety-readiness",
            "--no-use-conda",
            "--snakemake",
            str(fake_snakemake),
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )

    assert res.returncode != 0
    assert "Readiness preflight skips are only allowed for --dry-run" in res.stderr
    assert not (tmp_path / "unsafe_skip_case" / "run_manifest.json").exists()
    assert "--allow-unsafe-readiness-skip" in res.stderr


def test_runner_allows_explicit_unsafe_readiness_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    monkeypatch.setattr(runner, "_print_completed_run_result", lambda *_args: None)
    calls: list[list[str]] = []

    def fake_subprocess_run(
        cmd: list[str],
        *_args: object,
        **_kwargs: object,
    ) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    res = runner.main(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "explicit_unsafe_skip_case",
            "--preset",
            "safety",
            "--skip-data-readiness",
            "--skip-model-readiness",
            "--skip-safety-readiness",
            "--allow-unsafe-readiness-skip",
            "--no-use-conda",
        ]
    )

    assert res == 0
    assert calls[0][0] == "snakemake"
    assert calls[1][:2] == [sys.executable, "scripts/summarize_run_outputs.py"]
    assert calls[2][:2] == [sys.executable, "scripts/verify_run_outputs.py"]
    assert "--json-out" in calls[2]


def test_runner_verifies_outputs_after_successful_workflow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_stage0_claim_quality_readiness = lambda: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    runner._run_gpu_admission = lambda: None
    calls: list[list[str]] = []
    printed: list[str] = []
    monkeypatch.setattr(
        runner,
        "_print_completed_run_result",
        lambda args: printed.append(args.run_id),
    )

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    res = runner.main(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "verified_case",
            "--preset",
            "target-id",
            "--mode",
            "comprehensive",
        ]
    )

    assert res == 0
    assert "target_metadata=data/hpa/proteinatlas.tsv" in calls[0]
    summary_cmd = calls[1]
    assert summary_cmd[:2] == [sys.executable, "scripts/summarize_run_outputs.py"]
    assert "--out-json" in summary_cmd
    assert str(TEST_RESULTS_ROOT / "verified_case" / "run_summary.json") in summary_cmd
    assert "--out-md" in summary_cmd
    assert str(TEST_RESULTS_ROOT / "verified_case" / "run_summary.md") in summary_cmd
    assert ["--target-metadata", "data/hpa/proteinatlas.tsv"] == summary_cmd[
        summary_cmd.index("--target-metadata"):summary_cmd.index("--target-metadata") + 2
    ]
    verify_cmd = calls[2]
    assert verify_cmd[:2] == [sys.executable, "scripts/verify_run_outputs.py"]
    assert "--json-out" in verify_cmd
    assert "--run-dir" in verify_cmd
    assert str(TEST_RESULTS_ROOT / "verified_case") in verify_cmd
    assert ["--preset", "target-id"] == verify_cmd[
        verify_cmd.index("--preset"):verify_cmd.index("--preset") + 2
    ]
    assert ["--mode", "comprehensive"] == verify_cmd[
        verify_cmd.index("--mode"):verify_cmd.index("--mode") + 2
    ]
    assert ["--target-metadata", "data/hpa/proteinatlas.tsv"] == verify_cmd[
        verify_cmd.index("--target-metadata"):verify_cmd.index("--target-metadata") + 2
    ]
    verification_json = TEST_RESULTS_ROOT / "verified_case" / "run_verification.json"
    verification_log = TEST_RESULTS_ROOT / "verified_case" / "run_verification.log"
    record = json.loads(verification_json.read_text())
    assert record["schema_version"] == "skinscout.run_verification.v1"
    assert record["status"] == "ok"
    assert record["returncode"] == 0
    assert record["command"] == verify_cmd
    assert record["stdout_log"] == "run_verification.log"
    assert record["input_provenance"] == smiles_input_provenance("CCO")
    assert record["diagnostic_nonclaimable_reasons"] == []
    assert record["verifier_status"] == "ok"
    expected_verifier = json.loads(
        verifier_stdout(
            run_dir=TEST_RESULTS_ROOT / "verified_case",
            preset="target-id",
            mode="comprehensive",
        )
    )
    assert record["checks"] == expected_verifier["checks"]
    assert record["verifier_payload"]["checks"] == record["checks"]
    assert "user-facing run summary" in verification_log.read_text()
    assert not (
        TEST_RESULTS_ROOT / "verified_case" / ".run_verification_payload.json"
    ).exists()
    out = capsys.readouterr().out
    assert "SkinScout run output verification: ok" in out
    assert '"checks"' not in out
    assert printed == ["verified_case"]


def test_runner_verify_existing_run_regenerates_completed_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        assert cmd[0] != "snakemake"
        if "scripts/summarize_run_outputs.py" in cmd:
            out_json = Path(cmd[cmd.index("--out-json") + 1])
            preset = cmd[cmd.index("--preset") + 1]
            write_run_summary(out_json, preset=preset, write_verification=False)
            return Namespace(returncode=0, stdout="summary regenerated\n", stderr="")
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            run_dir = Path(cmd[cmd.index("--run-dir") + 1])
            preset = cmd[cmd.index("--preset") + 1]
            mode = cmd[cmd.index("--mode") + 1]
            return Namespace(
                returncode=0,
                stdout=verifier_text(run_dir=run_dir, preset=preset, mode=mode),
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    res = runner.main([
        "--smiles",
        "CCO",
        "--run-id",
        "existing_verify_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
        "--no-use-conda",
        "--verify-existing-run",
    ])

    assert res == 0
    assert len(calls) == 2
    assert calls[0][:2] == [sys.executable, "scripts/summarize_run_outputs.py"]
    assert calls[1][:2] == [sys.executable, "scripts/verify_run_outputs.py"]
    assert not any(cmd[0] == "snakemake" for cmd in calls)
    run_dir = tmp_path / "existing_verify_case"
    record = json.loads((run_dir / "run_verification.json").read_text())
    assert record["schema_version"] == "skinscout.run_verification.v1"
    assert record["status"] == "ok"
    assert record["checks"]
    assert record["verified_artifacts"]
    assert record["verifier_payload"]["checks"] == record["checks"]
    assert not (run_dir / ".run_verification_payload.json").exists()
    out = capsys.readouterr().out
    assert "summary regenerated" in out
    assert "Completed run result:" in out
    assert f"summary_json: {run_dir / 'run_summary.json'}" in out
    assert "verification_checks: 1" in out
    assert "claimable: no" in out


def test_runner_verify_existing_run_rejects_stage0() -> None:
    runner = load_runner_module()

    with pytest.raises(SystemExit) as exc:
        runner.main([
            "--preset",
            "stage0",
            "--no-use-conda",
            "--verify-existing-run",
        ])

    assert "--verify-existing-run is only for completed safety/target-id/report runs" in (
        str(exc.value)
    )


def test_runner_rejects_target_run_when_stage0_claim_quality_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    runner._run_gpu_admission = lambda: None
    calls: list[list[str]] = []

    def fail_stage0_claim_quality() -> None:
        raise SystemExit("Stage 0 claim-quality preflight failed")

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    runner._run_stage0_claim_quality_readiness = fail_stage0_claim_quality
    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "stage0_claim_failure_case",
                "--preset",
                "target-id",
            ]
        )

    assert str(exc.value) == "Stage 0 claim-quality preflight failed"
    assert calls == []


def test_runner_checks_stage0_claim_quality_after_allowed_stage0_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    runner._run_gpu_admission = lambda: None
    monkeypatch.setattr(runner, "_print_completed_run_result", lambda *_args: None)
    claim_checks: list[str] = []
    calls: list[list[str]] = []

    def fake_stage0_claim_quality() -> None:
        claim_checks.append("checked")

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    runner._run_stage0_claim_quality_readiness = fake_stage0_claim_quality
    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    res = runner.main(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "allowed_stage0_build_case",
            "--preset",
            "target-id",
            "--allow-stage0-build",
        ]
    )

    assert res == 0
    assert claim_checks == ["checked"]
    assert calls[0][0] == "snakemake"
    assert calls[1][:2] == [sys.executable, "scripts/summarize_run_outputs.py"]
    assert calls[2][:2] == [sys.executable, "scripts/verify_run_outputs.py"]
    assert "--json-out" in calls[2]


def test_runner_output_verification_failure_fails_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd, "failed")
            return Namespace(returncode=2, stdout=verifier_text("failed"), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "failed_verify_case",
                "--preset",
                "safety",
            ]
        )

    assert str(exc.value) == "Completed run output verification failed"
    assert len(calls) == 3
    verification_json = TEST_RESULTS_ROOT / "failed_verify_case" / "run_verification.json"
    verification_log = TEST_RESULTS_ROOT / "failed_verify_case" / "run_verification.log"
    record = json.loads(verification_json.read_text())
    assert record["status"] == "failed"
    assert record["returncode"] == 2
    assert record["verifier_status"] == "failed"
    assert record["checks"][0]["status"] == "failed"
    assert "missing required field" in verification_log.read_text()


def test_runner_rejects_successful_verifier_without_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            if "--json-out" in cmd:
                path = Path(cmd[cmd.index("--json-out") + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("not-json\n")
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "invalid_verify_json_case",
                "--preset",
                "safety",
            ]
        )

    assert str(exc.value) == "Completed run output verification did not produce valid JSON"
    assert "--json-out" in calls[2]
    verification_json = (
        TEST_RESULTS_ROOT / "invalid_verify_json_case" / "run_verification.json"
    )
    record = json.loads(verification_json.read_text())
    assert record["status"] == "failed"
    assert record["returncode"] == 0
    assert record["checks"] == []
    assert "verifier_json_error" in record


def test_runner_rejects_successful_verifier_with_failed_json_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd, "failed")
            return Namespace(returncode=0, stdout=verifier_text("failed"), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "failed_verify_json_status_case",
                "--preset",
                "safety",
            ]
        )

    assert str(exc.value) == "Completed run output verification JSON status was not ok"
    assert "--json-out" in calls[2]
    verification_json = (
        TEST_RESULTS_ROOT / "failed_verify_json_status_case" / "run_verification.json"
    )
    record = json.loads(verification_json.read_text())
    assert record["status"] == "failed"
    assert record["returncode"] == 0
    assert record["verifier_status"] == "failed"
    assert record["checks"][0]["status"] == "failed"
    assert "verifier_contract_error" in record


def test_runner_rejects_successful_verifier_with_mismatched_json_run_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            path = Path(cmd[cmd.index("--json-out") + 1])
            payload = json.loads(path.read_text())
            payload["run_dir"] = str(TEST_RESULTS_ROOT / "other_run")
            path.write_text(json.dumps(payload))
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "mismatched_verify_run_dir_case",
                "--preset",
                "safety",
            ]
        )

    assert str(exc.value) == "Completed run output verification JSON contract failed"
    assert "--json-out" in calls[2]
    verification_json = (
        TEST_RESULTS_ROOT / "mismatched_verify_run_dir_case" / "run_verification.json"
    )
    record = json.loads(verification_json.read_text())
    assert record["status"] == "failed"
    assert record["returncode"] == 0
    assert record["verifier_status"] == "ok"
    assert (
        record["verifier_contract_error"]
        == "verifier JSON run_dir does not match this run"
    )


def test_runner_rejects_successful_verifier_with_empty_json_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            path = Path(cmd[cmd.index("--json-out") + 1])
            payload = json.loads(path.read_text())
            payload["checks"] = []
            path.write_text(json.dumps(payload))
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "empty_verify_checks_case",
                "--preset",
                "safety",
            ]
        )

    assert str(exc.value) == "Completed run output verification JSON contract failed"
    assert "--json-out" in calls[2]
    verification_json = (
        TEST_RESULTS_ROOT / "empty_verify_checks_case" / "run_verification.json"
    )
    record = json.loads(verification_json.read_text())
    assert record["status"] == "failed"
    assert record["returncode"] == 0
    assert record["verifier_status"] == "ok"
    assert record["checks"] == []
    assert (
        record["verifier_contract_error"]
        == "verifier JSON checks must be a non-empty list"
    )


def test_runner_rejects_successful_verifier_with_invalid_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            path = Path(cmd[cmd.index("--json-out") + 1])
            payload = json.loads(path.read_text())
            del payload["schema_version"]
            path.write_text(json.dumps(payload))
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "invalid_verify_schema_case",
                "--preset",
                "safety",
            ]
        )

    assert str(exc.value) == "Completed run output verification JSON contract failed"
    assert "--json-out" in calls[2]
    verification_json = (
        TEST_RESULTS_ROOT / "invalid_verify_schema_case" / "run_verification.json"
    )
    record = json.loads(verification_json.read_text())
    assert record["status"] == "failed"
    assert record["returncode"] == 0
    assert record["verifier_status"] == "ok"
    assert (
        record["verifier_contract_error"]
        == "verifier JSON has invalid schema_version"
    )


def test_runner_output_summary_failure_fails_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/summarize_run_outputs.py" in cmd:
            return Namespace(returncode=2, stdout="", stderr="summary failed\n")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--smiles",
                "CCO",
                "--run-id",
                "failed_summary_case",
                "--preset",
                "safety",
            ]
        )

    assert str(exc.value) == "Completed run summary generation failed"
    assert len(calls) == 2


def test_runner_passes_degraded_flag_to_output_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runner._run_data_readiness = lambda *_args, **_kwargs: None
    runner._run_safety_readiness = lambda **_kwargs: None
    runner._run_model_readiness = lambda *_args, **_kwargs: None
    monkeypatch.setattr(runner, "_print_completed_run_result", lambda *_args: None)
    calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], *_args: object, **_kwargs: object) -> Namespace:
        calls.append(cmd)
        if "scripts/verify_run_outputs.py" in cmd:
            write_verifier_json_out(cmd)
            return Namespace(returncode=0, stdout=verifier_text(), stderr="")
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    res = runner.main(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "degraded_verify_case",
            "--preset",
            "safety",
            "--allow-safety-degraded",
        ]
    )

    assert res == 0
    assert "--allow-degraded" in calls[1]
    assert "--allow-degraded" in calls[2]
    assert "--json-out" in calls[2]
    record = json.loads(
        (TEST_RESULTS_ROOT / "degraded_verify_case" / "run_verification.json")
        .read_text()
    )
    assert record["diagnostic_nonclaimable_reasons"] == [
        "explicit degraded ADMET/skin-sens evidence was allowed"
    ]


def test_runner_prints_completed_result_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "printed_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    write_run_summary(tmp_path / "printed_case" / "run_summary.json")

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "Completed run result:" in out
    assert f"summary_json: {tmp_path / 'printed_case' / 'run_summary.json'}" in out
    assert f"verification_json: {tmp_path / 'printed_case' / 'run_verification.json'}" in out
    assert f"verification_log: {tmp_path / 'printed_case' / 'run_verification.log'}" in out
    assert "verification_checks: 1" in out
    assert "verified_artifacts: 4" in out
    assert "run_id: printed_case" in out
    assert "preset: target-id" in out
    assert "mode: fast" in out
    assert "input_type: smiles" in out
    assert "input_smiles: CCO" in out
    assert "input_canonical_smiles: CCO" in out
    assert "canonical_smiles: CCO" in out
    assert "inchikey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N" in out
    assert "overall_decision: FLAG_HIGH" in out
    assert "recommended_action: review_before_claim" in out
    assert "requires_human_review: yes" in out
    assert "claimable: no" in out
    assert "claim_status: human review required before claim" in out
    assert (
        "decision_reasons: Skin_Reaction ADMET risk requires review: moderate; "
        "top binding target lacks direct skin context support: P10275"
        in out
    )
    assert "skin_toxicity: PASS" in out
    assert "skin_toxicity_level: low" in out
    assert "skin_reaction_risk: moderate" in out
    assert "skin_reaction_value: 0.31" in out
    assert "structural_alerts_present: yes" in out
    assert "structural_alert_flags: BRENK" in out
    assert "safety_evidence_degraded: no" in out
    assert "missing_skin_sens_models: none" in out
    assert "skin_sens_husspred_status: ok" in out
    assert "skin_sens_husspred_call: negative" in out
    assert "skin_sens_husspred_probability: 0.2" in out
    assert "skin_sens_stoptox_status: ok" in out
    assert "skin_sens_stoptox_call: negative" in out
    assert "skin_sens_stoptox_probability: 0.2" in out
    assert "skin_sens_pred_skin_status: ok" in out
    assert "skin_sens_pred_skin_call: negative" in out
    assert "skin_sens_pred_skin_probability: 0.2" in out
    assert "high_admet_risk_endpoints: DILI" in out
    assert "moderate_admet_risk_endpoints: Skin_Reaction" in out
    assert "admet_AMES: 0.12" in out
    assert "admet_ClinTox: 0.03" in out
    assert "admet_DILI: 0.55" in out
    assert "admet_Skin_Reaction: 0.31" in out
    assert "admet_hERG: 0.04" in out
    assert "admet_LD50_Zhu: 1.1" in out
    assert "admet_Solubility_AqSolDB: 1.2" in out
    assert "admet_logP: -0.0014" in out
    assert "admet_QED: 0.4068" in out
    assert "admet_tpsa: 20.23" in out
    assert "cosmetic_drug_decision: PROCEED" in out
    assert "drug_policy: moderate" in out
    assert "drug_warnings: 1" in out
    assert "top_target: AR (P10275) - Androgen receptor" in out
    assert "target_count: 20171" in out
    assert "screened_target_count: 20171" in out
    assert "screening_autodock_rescored_targets: 256" in out
    assert "screening_daina_zoete_targets: 2943" in out
    assert "screening_dti_rrf_candidates: 732" in out
    assert "screening_psichic_proteome_targets: 20171" in out
    assert "screening_rerank_consensus_targets: 10" in out
    assert "screening_skin_weighted_ranked_targets: 2" in out
    assert "top_targets_reported: 2" in out
    assert "ranked_targets:" in out
    assert "ranked_target_1: AR (P10275) - Androgen receptor" in out
    assert "ranked_target_1_final_score: 0.87" in out
    assert "ranked_target_1_docking_rrf: 0.41" in out
    assert "ranked_target_1_source_count: 3" in out
    assert "ranked_target_1_sources: autodock, gnina, rtmscore" in out
    assert "ranked_target_1_skin_score: 0.28" in out
    assert "ranked_target_1_skin_tier: medium" in out
    assert "ranked_target_1_efficacy: barrier repair (2 papers)" in out
    assert "ranked_target_2: CAT (P04040) - Catalase" in out
    assert "ranked_target_2_final_score: 0.71" in out
    assert "ranked_target_2_docking_rrf: 0.22" in out
    assert "ranked_target_2_source_count: 2" in out
    assert "ranked_target_2_sources: psichic, autodock" in out
    assert "ranked_target_2_skin_score: 0.32" in out
    assert "ranked_target_2_skin_tier: medium" in out
    assert "ranked_target_2_efficacy: oxidative stress (1 paper)" in out
    assert "skin_context: skin_efficacy_literature_only" in out
    assert "skin_context_supported: no" in out
    assert "skin_expression_supported: no" in out
    assert "skin_efficacy_supported: yes" in out
    assert "top_target_final_score: 0.87" in out
    assert "top_target_docking_rrf: 0.41" in out
    assert "top_target_source_count: 3" in out
    assert "top_target_sources: autodock, gnina, rtmscore" in out
    assert "top_target_skin_score: 0.28" in out
    assert "top_target_skin_tier: medium" in out
    assert "top_target_gene_symbol: AR" in out
    assert "top_target_protein_name: Androgen receptor" in out
    assert "top_target_skin_expression_supported: no" in out
    assert "top_target_skin_efficacy_supported: yes" in out
    assert "top_target_skin_context_supported: no" in out
    assert "top_targets_with_skin_efficacy: 3" in out
    assert "top_target_skin_efficacy: barrier repair (2 papers)" in out
    assert "most_skin_relevant_target: AR (P10275) - Androgen receptor" in out
    assert "most_skin_relevant_gene_symbol: AR" in out
    assert "most_skin_relevant_protein_name: Androgen receptor" in out
    assert "most_skin_relevant_final_score: 0.87" in out
    assert "most_skin_relevant_docking_rrf: 0.41" in out
    assert "most_skin_relevant_source_count: 3" in out
    assert "most_skin_relevant_sources: autodock, gnina, rtmscore" in out
    assert "most_skin_relevant_skin_score: 0.28" in out
    assert "most_skin_relevant_skin_tier: medium" in out
    assert "most_skin_relevant_efficacy: barrier repair (2 papers)" in out
    assert "source_artifacts: 9 listed in run_summary.md" in out
    assert "target_source_artifacts: 3" in out
    assert "artifact_admet_report: 02_admet/admet_report.json" in out
    assert "artifact_husspred: 02_admet/husspred.json" in out
    assert "artifact_pred_skin: 02_admet/pred_skin.json" in out
    assert "artifact_stoptox: 02_admet/stoptox.json" in out
    assert "artifact_target_ranking: 03_targets/ranked_targets_v3_with_efficacy.csv" in out
    assert "artifact_target_fast_psichic: 03_targets/mode_fast/psichic_proteome.tsv" in out
    assert "artifact_target_fast_rerank_consensus: 03_targets/mode_fast/top50.csv" in out


def test_runner_prints_completed_report_html_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "printed_report_case",
        "--preset",
        "report",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    write_run_summary(
        tmp_path / "printed_report_case" / "run_summary.json",
        preset="report",
    )

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "preset: report" in out
    assert (
        f"html_report: {tmp_path / 'printed_report_case' / '09_report' / 'index.html'}"
        in out
    )
    assert "verified_artifacts: 5" in out
    assert "artifact_html_report: 09_report/index.html" in out


def test_runner_report_result_requires_html_artifact(tmp_path: Path) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_report_artifact_case",
        "--preset",
        "report",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "missing_report_artifact_case" / "run_summary.json"
    write_run_summary(summary_path, preset="report")
    payload = json.loads(summary_path.read_text())
    del payload["artifacts"]["html_report"]
    summary_path.write_text(json.dumps(payload))
    write_run_verification(summary_path.parent, preset="report")

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed report summary missing artifacts.html_report" in str(exc.value)


def test_runner_report_result_rejects_unexpected_html_artifact_path(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "unexpected_report_artifact_case",
        "--preset",
        "report",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "unexpected_report_artifact_case" / "run_summary.json"
    write_run_summary(summary_path, preset="report")
    payload = json.loads(summary_path.read_text())
    payload["artifacts"]["html_report"] = "09_report/other.html"
    summary_path.write_text(json.dumps(payload))
    write_run_verification(summary_path.parent, preset="report")

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed report summary artifacts.html_report has unexpected path" in (
        str(exc.value)
    )


def test_runner_completed_report_rejects_modified_html_after_verification(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "modified_report_html_case",
        "--preset",
        "report",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "modified_report_html_case" / "run_summary.json"
    write_run_summary(summary_path, preset="report")
    (summary_path.parent / "09_report" / "index.html").write_text(
        "<html><body>staleHTML report</body></html>\n"
    )

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification artifact sha256 changed: html_report" in (
        str(exc.value)
    )


def test_runner_completed_result_requires_ok_verification_record(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "failed_verification_record_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "failed_verification_record_case" / "run_summary.json"
    write_run_summary(summary_path)
    write_run_verification(summary_path.parent, status="failed")

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record status is not ok" in str(exc.value)


def test_runner_completed_result_rejects_stale_verification_record(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "stale_verification_record_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "stale_verification_record_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["run_dir"] = str(tmp_path / "other_run")
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record run_dir does not match this run" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_missing_verification_timestamp(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_verification_timestamp_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "missing_verification_timestamp_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    del record["verified_at_utc"]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record missing verified_at_utc" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_invalid_verification_timestamp(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "invalid_verification_timestamp_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "invalid_verification_timestamp_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["verified_at_utc"] = "2026-06-26 00:00:00"
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record verified_at_utc must be a UTC "
        "ISO-8601 timestamp"
    ) in str(exc.value)


def test_runner_completed_result_rejects_verification_stdout_log_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_stdout_log_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path / "verification_stdout_log_mismatch_case" / "run_summary.json"
    )
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["stdout_log"] = "other.log"
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record stdout_log does not match this run" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_verification_log_command_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_log_command_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path / "verification_log_command_mismatch_case" / "run_summary.json"
    )
    write_run_summary(summary_path)
    log_path = summary_path.parent / "run_verification.log"
    lines = log_path.read_text().splitlines()
    lines[0] = "$ python scripts/other_verifier.py"
    log_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification log command does not match record" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_verification_log_missing_ok_summary(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_log_missing_ok_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "verification_log_missing_ok_case" / "run_summary.json"
    write_run_summary(summary_path)
    log_path = summary_path.parent / "run_verification.log"
    log_path.write_text(
        log_path.read_text().replace(
            "SkinScout run output verification: ok",
            "SkinScout run output verification: failed",
        )
    )

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification log missing verifier ok summary" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_verification_log_metadata_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_log_metadata_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path / "verification_log_metadata_mismatch_case" / "run_summary.json"
    )
    write_run_summary(summary_path)
    log_path = summary_path.parent / "run_verification.log"
    log_path.write_text(log_path.read_text().replace("preset=target-id", "preset=safety"))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification log run metadata does not match this run" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_verification_command_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_command_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "verification_command_mismatch_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["command"][1] = "scripts/other_verifier.py"
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record command does not match this run" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_cli_target_metadata_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_cli_metadata_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--target-metadata",
        "data/hpa/different_proteinatlas.tsv",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path
        / "verification_cli_metadata_mismatch_case"
        / "run_summary.json"
    )
    write_run_summary(summary_path)

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record command does not match this run" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_degraded_reason_command_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "degraded_reason_command_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path / "degraded_reason_command_mismatch_case" / "run_summary.json"
    )
    write_run_summary(summary_path, write_verification=False)
    write_run_verification(
        summary_path.parent,
        diagnostic_reasons=[
            "explicit degraded ADMET/skin-sens evidence was allowed"
        ],
    )
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["command"] = [
        part for part in record["command"] if part != "--allow-degraded"
    ]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record degraded diagnostic reason "
        "does not match verifier command"
    ) in str(exc.value)


def test_runner_completed_result_rejects_missing_verifier_payload(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_verifier_payload_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "missing_verifier_payload_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    del record["verifier_payload"]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record missing verifier_payload" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_verifier_payload_schema_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verifier_payload_schema_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path / "verifier_payload_schema_mismatch_case" / "run_summary.json"
    )
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    del record["verifier_payload"]["schema_version"]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification payload has invalid schema_version"
        in str(exc.value)
    )


def test_runner_completed_result_rejects_verifier_payload_run_dir_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verifier_payload_run_dir_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path / "verifier_payload_run_dir_mismatch_case" / "run_summary.json"
    )
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["verifier_payload"]["run_dir"] = str(tmp_path / "other")
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification payload run_dir does not match this run"
        in str(exc.value)
    )


def test_runner_completed_result_rejects_verifier_payload_checks_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verifier_payload_checks_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path / "verifier_payload_checks_mismatch_case" / "run_summary.json"
    )
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["verifier_payload"]["checks"][0]["detail"] = "stale detail"
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification payload checks do not match record" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_ok_record_with_verifier_error_field(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "ok_record_with_error_field_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "ok_record_with_error_field_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["verifier_contract_error"] = "stale verifier error"
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record contains verifier_contract_error" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_missing_verification_input_provenance(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_verification_input_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "missing_verification_input_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    del record["input_provenance"]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record missing input provenance" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_missing_recorded_diagnostic_reasons(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_recorded_diagnostic_reasons_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path
        / "missing_recorded_diagnostic_reasons_case"
        / "run_summary.json"
    )
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    del record["diagnostic_nonclaimable_reasons"]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record missing diagnostic non-claimable reasons"
        in str(exc.value)
    )


def test_runner_completed_result_rejects_verification_input_smiles_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_input_smiles_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path
        / "verification_input_smiles_mismatch_case"
        / "run_summary.json"
    )
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["input_provenance"] = smiles_input_provenance("CCN")
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record input SMILES does not match this run"
        in str(exc.value)
    )


def test_runner_completed_result_rejects_verification_smiles_for_sdf_run(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    input_sdf = tmp_path / "compound.sdf"
    input_sdf.write_text("sdf\n")
    args = runner.parse_args([
        "--sdf",
        str(input_sdf),
        "--run-id",
        "verification_sdf_with_smiles_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "verification_sdf_with_smiles_case" / "run_summary.json"
    write_run_summary(summary_path, write_verification=False)
    payload = json.loads(summary_path.read_text())
    payload["compound"]["input_type"] = "sdf"
    payload["compound"]["input_smiles"] = None
    payload["compound"]["input_canonical_smiles"] = None
    payload["compound"]["input_sdf"] = str(input_sdf)
    summary_path.write_text(json.dumps(payload))
    write_run_verification(
        summary_path.parent,
        input_provenance={
            "input_type": "sdf",
            "input_smiles": "CCO",
            "input_canonical_smiles": None,
            "input_sdf": str(input_sdf),
        },
    )

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record input SMILES must be absent for SDF runs"
        in str(exc.value)
    )


def test_runner_completed_result_rejects_failed_verification_check(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "failed_verification_check_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "failed_verification_check_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["checks"][0]["status"] = "failed"
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record check is not ok: "
        "user-facing run summary"
    ) in str(exc.value)


def test_runner_completed_result_rejects_malformed_verification_check_name(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "malformed_verification_check_name_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "malformed_verification_check_name_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["checks"][0]["name"] = ""
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record check 1 name must be a non-empty string"
        in str(exc.value)
    )


def test_runner_completed_result_rejects_malformed_verification_check_path(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "malformed_verification_check_path_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "malformed_verification_check_path_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["checks"][0]["path"] = ""
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record check user-facing run summary "
        "path must be a non-empty string"
    ) in str(exc.value)


def test_runner_completed_result_rejects_verification_check_path_outside_run(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "outside_verification_check_path_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "outside_verification_check_path_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["checks"][0]["path"] = str(tmp_path / "outside.json")
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record check path is outside this run: "
        "user-facing run summary"
    ) in str(exc.value)


def test_runner_completed_result_rejects_verification_log_check_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "verification_log_check_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "verification_log_check_mismatch_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    record["checks"][0]["detail"] = "bytes=999"
    record["verifier_payload"]["checks"] = record["checks"]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification log missing check entry: "
        "user-facing run summary"
    ) in str(exc.value)


def test_runner_completed_result_rejects_missing_verified_artifacts(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_verified_artifacts_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "missing_verified_artifacts_case" / "run_summary.json"
    write_run_summary(summary_path)
    record_path = summary_path.parent / "run_verification.json"
    record = json.loads(record_path.read_text())
    del record["verified_artifacts"]
    record_path.write_text(json.dumps(record))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record verified_artifacts are incomplete" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_modified_summary_after_verification(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "modified_summary_after_verification_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path
        / "modified_summary_after_verification_case"
        / "run_summary.json"
    )
    write_run_summary(summary_path)
    payload = json.loads(summary_path.read_text())
    payload["compound"]["input_smiles"] = "CCN"
    summary_path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification artifact sha256 changed: run_summary_json" in (
        str(exc.value)
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("run_id", "other_run", "Completed run summary run_id does not match this run"),
        ("preset", "safety", "Completed run summary preset does not match this run"),
        ("mode", "comprehensive", "Completed run summary mode does not match this run"),
    ],
)
def test_runner_completed_result_rejects_summary_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    expected: str,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "summary_identity_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "summary_identity_mismatch_case" / "run_summary.json"
    write_run_summary(summary_path)
    payload = json.loads(summary_path.read_text())
    payload[field] = value
    summary_path.write_text(json.dumps(payload))
    write_run_verification(summary_path.parent, preset="target-id", mode="fast")

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert expected in str(exc.value)


def test_runner_completed_result_rejects_input_smiles_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCN",
        "--run-id",
        "summary_input_smiles_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "summary_input_smiles_mismatch_case" / "run_summary.json"
    write_run_summary(summary_path)
    payload = json.loads(summary_path.read_text())
    payload["compound"]["input_smiles"] = "CCN"
    summary_path.write_text(json.dumps(payload))
    write_run_verification(
        summary_path.parent,
        input_provenance=smiles_input_provenance("CCN"),
    )

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run summary input canonical SMILES does not match this run"
        in str(exc.value)
    )


def test_runner_completed_result_rejects_raw_input_smiles_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "C(C)O",
        "--run-id",
        "summary_raw_input_smiles_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path
        / "summary_raw_input_smiles_mismatch_case"
        / "run_summary.json"
    )
    write_run_summary(summary_path)
    write_run_verification(
        summary_path.parent,
        input_provenance=smiles_input_provenance("C(C)O", "CCO"),
    )

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run summary input SMILES does not match this run" in (
        str(exc.value)
    )


def test_runner_completed_result_rejects_input_sdf_mismatch(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    old_sdf = tmp_path / "old.sdf"
    new_sdf = tmp_path / "new.sdf"
    old_sdf.write_text("old sdf\n")
    new_sdf.write_text("new sdf\n")
    args = runner.parse_args([
        "--sdf",
        str(new_sdf),
        "--run-id",
        "summary_input_sdf_mismatch_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "summary_input_sdf_mismatch_case" / "run_summary.json"
    write_run_summary(summary_path)
    payload = json.loads(summary_path.read_text())
    payload["compound"]["input_type"] = "sdf"
    payload["compound"]["input_smiles"] = None
    payload["compound"]["input_canonical_smiles"] = None
    payload["compound"]["input_sdf"] = str(old_sdf)
    summary_path.write_text(json.dumps(payload))
    write_run_verification(
        summary_path.parent,
        input_provenance=sdf_input_provenance(new_sdf),
    )

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run summary input SDF does not match this run" in str(exc.value)


def test_runner_completed_result_rejects_smiles_provenance_for_sdf_run(
    tmp_path: Path,
) -> None:
    runner = load_runner_module()
    input_sdf = tmp_path / "compound.sdf"
    input_sdf.write_text("sdf\n")
    args = runner.parse_args([
        "--sdf",
        str(input_sdf),
        "--run-id",
        "summary_sdf_with_smiles_provenance_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path
        / "summary_sdf_with_smiles_provenance_case"
        / "run_summary.json"
    )
    write_run_summary(summary_path)
    payload = json.loads(summary_path.read_text())
    payload["compound"]["input_type"] = "sdf"
    payload["compound"]["input_sdf"] = str(input_sdf)
    summary_path.write_text(json.dumps(payload))
    write_run_verification(
        summary_path.parent,
        input_provenance=sdf_input_provenance(input_sdf),
    )

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run summary input SMILES must be absent for SDF runs" in (
        str(exc.value)
    )


def test_runner_completed_result_uses_summary_claimable_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "summary_claimable_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    write_run_summary(
        tmp_path / "summary_claimable_case" / "run_summary.json",
        action="proceed",
        claimable=False,
    )

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "recommended_action: proceed" in out
    assert "requires_human_review: no" in out
    assert "claimable: no" in out
    assert "claim_status: not claimable; summary claimable is not true" in out


def test_runner_completed_result_requires_review_action_to_stay_nonclaimable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "review_claimable_guard_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    write_run_summary(
        tmp_path / "review_claimable_guard_case" / "run_summary.json",
        action="review_before_claim",
        claimable=True,
    )

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "recommended_action: review_before_claim" in out
    assert "requires_human_review: yes" in out
    assert "claimable: no" in out
    assert "claim_status: human review required before claim" in out


def test_runner_completed_result_requires_no_human_review_for_claimable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "requires_human_review_claimable_guard_case",
        "--preset",
        "safety",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = (
        tmp_path
        / "requires_human_review_claimable_guard_case"
        / "run_summary.json"
    )
    write_run_summary(summary_path, action="proceed", claimable=True, preset="safety")
    payload = json.loads(summary_path.read_text())
    payload["overall_decision"]["requires_human_review"] = True
    summary_path.write_text(json.dumps(payload))
    write_run_verification(summary_path.parent, preset="safety")

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "recommended_action: proceed" in out
    assert "requires_human_review: yes" in out
    assert "claimable: no" in out
    assert "claim_status: human review required before claim" in out


def test_runner_completed_result_requires_top_target_skin_context_for_claimable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "target_context_claimable_guard_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    write_run_summary(
        tmp_path / "target_context_claimable_guard_case" / "run_summary.json",
        action="proceed",
        claimable=True,
    )
    summary_path = tmp_path / "target_context_claimable_guard_case" / "run_summary.json"
    payload = json.loads(summary_path.read_text())
    payload["skin_specialized_binding"]["skin_context_supported"] = True
    payload["skin_specialized_binding"]["skin_context_decision"] = "skin_context_supported"
    payload["skin_specialized_binding"]["skin_expression_supported"] = True
    summary_path.write_text(json.dumps(payload))
    write_run_verification(summary_path.parent)

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "recommended_action: proceed" in out
    assert "skin_context_supported: yes" in out
    assert "top_target_skin_context_supported: no" in out
    assert "claimable: no" in out
    assert (
        "claim_status: not claimable; top binding target skin context is not supported"
        in out
    )


def test_runner_completed_result_requires_aggregate_skin_context_for_claimable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "aggregate_context_claimable_guard_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "aggregate_context_claimable_guard_case" / "run_summary.json"
    write_run_summary(summary_path, action="proceed", claimable=True)
    payload = json.loads(summary_path.read_text())
    payload["skin_specialized_binding"]["top_target_skin_expression_supported"] = True
    payload["skin_specialized_binding"]["top_target_skin_context_supported"] = True
    payload["skin_specialized_binding"]["skin_context_supported"] = False
    payload["skin_specialized_binding"]["skin_context_decision"] = (
        "skin_efficacy_literature_only"
    )
    summary_path.write_text(json.dumps(payload))
    write_run_verification(summary_path.parent)

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "recommended_action: proceed" in out
    assert "skin_context_supported: no" in out
    assert "top_target_skin_context_supported: yes" in out
    assert "claimable: no" in out
    assert (
        "claim_status: not claimable; skin-specialized binding context is not supported"
        in out
    )


@pytest.mark.parametrize(
    ("extra_args", "expected_reason", "expected_action"),
    [
        (
            [
                "--skip-data-readiness",
                "--skip-safety-readiness",
                "--skip-model-readiness",
                "--allow-unsafe-readiness-skip",
            ],
            "readiness preflight skipped with diagnostic override: "
            "data readiness, safety readiness, model readiness",
            "diagnostic_next_action: rerun without diagnostic readiness skip "
            "flags before treating this run as claimable: remove "
            "--skip-data-readiness, --skip-safety-readiness, "
            "--skip-model-readiness",
        ),
        (
            ["--allow-safety-degraded"],
            "explicit degraded ADMET/skin-sens evidence was allowed",
            "diagnostic_next_action: rerun readiness and launcher without "
            "--allow-safety-degraded before treating ADMET/skin-sens output "
            "as claimable",
        ),
    ],
)
def test_runner_prints_diagnostic_runs_as_nonclaimable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    expected_reason: str,
    expected_action: str,
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "diagnostic_claim_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
        *extra_args,
    ])
    summary_path = tmp_path / "diagnostic_claim_case" / "run_summary.json"
    write_run_summary(summary_path, action="proceed", write_verification=False)
    write_run_verification(
        summary_path.parent,
        diagnostic_reasons=[expected_reason],
    )

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "recommended_action: proceed" in out
    assert "claimable: no" in out
    assert "diagnostic_nonclaimable: yes" in out
    assert f"diagnostic_reason: {expected_reason}" in out
    assert expected_action in out
    assert "claim_status: diagnostic run; not claimable" in out


def test_runner_prints_recorded_diagnostic_runs_as_nonclaimable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "recorded_diagnostic_claim_case",
        "--preset",
        "target-id",
        "--mode",
        "fast",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    summary_path = tmp_path / "recorded_diagnostic_claim_case" / "run_summary.json"
    write_run_summary(
        summary_path,
        action="proceed",
        write_verification=False,
    )
    write_run_verification(
        summary_path.parent,
        diagnostic_reasons=[
            "explicit degraded ADMET/skin-sens evidence was allowed"
        ],
    )

    runner._print_completed_run_result(args)

    out = capsys.readouterr().out
    assert "recommended_action: proceed" in out
    assert "claimable: no" in out
    assert "diagnostic_nonclaimable: yes" in out
    assert (
        "diagnostic_reason: explicit degraded ADMET/skin-sens evidence was allowed"
        in out
    )
    assert (
        "diagnostic_next_action: rerun readiness and launcher without "
        "--allow-safety-degraded before treating ADMET/skin-sens output "
        "as claimable"
    ) in out
    assert "claim_status: diagnostic run; not claimable" in out


def test_runner_completed_result_missing_verification_fails(tmp_path: Path) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_verification_case",
        "--preset",
        "target-id",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert "Completed run verification record could not be read" in str(exc.value)


def test_runner_completed_result_missing_summary_fails(tmp_path: Path) -> None:
    runner = load_runner_module()
    args = runner.parse_args([
        "--smiles",
        "CCO",
        "--run-id",
        "missing_summary_case",
        "--preset",
        "target-id",
        "--extra-config",
        f"paths.results_root={tmp_path}",
    ])
    run_dir = tmp_path / "missing_summary_case"
    run_dir.mkdir(parents=True)
    write_run_verification(run_dir, mode="comprehensive")

    with pytest.raises(SystemExit) as exc:
        runner._print_completed_run_result(args)

    assert (
        "Completed run verification record check path is missing or not a file: "
        "user-facing run summary"
    ) in str(exc.value)


def test_runner_allows_output_verification_skip_with_diagnostic_override(
    tmp_path: Path,
) -> None:
    fake_snakemake = tmp_path / "snakemake"
    fake_snakemake.write_text("#!/usr/bin/env sh\nexit 0\n")
    fake_snakemake.chmod(0o755)

    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "skip_verify_case",
            "--preset",
            "safety",
            "--skip-output-verification",
            "--skip-data-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--allow-unsafe-readiness-skip",
            "--no-use-conda",
            "--snakemake",
            str(fake_snakemake),
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "Completed diagnostic run result:" in res.stdout
    assert "run_id: skip_verify_case" in res.stdout
    assert "preset: safety" in res.stdout
    assert "mode: comprehensive" in res.stdout
    assert "input_type: smiles" in res.stdout
    assert "input_smiles: CCO" in res.stdout
    assert "input_canonical_smiles: CCO" in res.stdout
    assert "output_verification: skipped" in res.stdout
    assert "summary_json: skipped" in res.stdout
    assert "claimable: no" in res.stdout
    assert "diagnostic_nonclaimable: yes" in res.stdout
    assert "completed-run output verification was skipped" in res.stdout
    assert (
        "diagnostic_next_action: rerun without diagnostic readiness skip "
        "flags before treating this run as claimable: remove "
        "--skip-data-readiness, --skip-safety-readiness, --skip-model-readiness"
    ) in res.stdout
    assert (
        "diagnostic_next_action: rerun without --skip-output-verification "
        "and keep completed-run verification enabled before treating output "
        "as claimable"
    ) in res.stdout
    assert "claim_status: diagnostic run; not claimable" in res.stdout


def test_runner_output_verification_skip_prints_sdf_provenance(
    tmp_path: Path,
) -> None:
    fake_snakemake = tmp_path / "snakemake"
    fake_snakemake.write_text("#!/usr/bin/env sh\nexit 0\n")
    fake_snakemake.chmod(0o755)
    sdf = tmp_path / "source_ligand.sdf"
    sdf.write_text(CAFFEINE_SDF)

    res = run_runner(
        [
            "--sdf",
            str(sdf),
            "--run-id",
            "skip_verify_sdf_case",
            "--preset",
            "safety",
            "--skip-output-verification",
            "--skip-data-readiness",
            "--skip-safety-readiness",
            "--skip-model-readiness",
            "--allow-unsafe-readiness-skip",
            "--no-use-conda",
            "--snakemake",
            str(fake_snakemake),
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "Completed diagnostic run result:" in res.stdout
    assert "run_id: skip_verify_sdf_case" in res.stdout
    assert "input_type: sdf" in res.stdout
    assert f"input_sdf: {sdf}" in res.stdout
    assert "input_smiles:" not in res.stdout
    assert "input_canonical_smiles:" not in res.stdout
    assert "claimable: no" in res.stdout
    assert (
        "diagnostic_next_action: rerun without --skip-output-verification "
        "and keep completed-run verification enabled before treating output "
        "as claimable"
    ) in res.stdout
    assert "claim_status: diagnostic run; not claimable" in res.stdout


def test_runner_rejects_output_verification_skip_without_diagnostic_override(
    tmp_path: Path,
) -> None:
    fake_snakemake = tmp_path / "snakemake"
    fake_snakemake.write_text("#!/usr/bin/env sh\nexit 0\n")
    fake_snakemake.chmod(0o755)

    res = run_runner(
        [
            "--smiles",
            "CCO",
            "--run-id",
            "skip_verify_rejected_case",
            "--preset",
            "safety",
            "--skip-output-verification",
            "--no-use-conda",
            "--snakemake",
            str(fake_snakemake),
            "--extra-config",
            f"paths.results_root={tmp_path}",
        ]
    )

    assert res.returncode != 0
    assert "Completed-run output verification cannot be skipped" in res.stderr
    assert not (
        tmp_path / "skip_verify_rejected_case" / "run_manifest.json"
    ).exists()


def test_micromamba_is_exposed_as_snakemake_mamba_frontend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("#!/usr/bin/env sh\nexit 0\n")
    micromamba.chmod(0o755)

    def fake_which(name: str) -> str | None:
        return str(micromamba) if name == "micromamba" else None

    monkeypatch.setattr(runner.shutil, "which", fake_which)
    args = Namespace(use_conda=True, conda_frontend="auto")

    assert runner._resolve_conda_frontend("auto", validate=True) == "mamba"
    env, shim = runner._snakemake_environment(args)
    try:
        assert shim is not None
        mamba = Path(env["PATH"].split(os.pathsep, 1)[0]) / "mamba"
        assert mamba.is_symlink()
        assert mamba.resolve() == micromamba.resolve()
    finally:
        if shim is not None:
            shim.cleanup()


def test_snakemake_environment_exposes_active_python_bin_after_reboot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_runner_module()
    runtime_bin = tmp_path / "cosmax-base" / "bin"
    runtime_bin.mkdir(parents=True)
    python = runtime_bin / "python"
    python.write_text("#!/usr/bin/env sh\nexit 0\n")
    python.chmod(0o755)
    snakemake = runtime_bin / "snakemake"
    snakemake.write_text("#!/usr/bin/env sh\nexit 0\n")
    snakemake.chmod(0o755)

    def fake_which(name: str) -> str | None:
        return None

    monkeypatch.setattr(runner.sys, "executable", str(python))
    monkeypatch.setattr(runner.shutil, "which", fake_which)
    args = Namespace(
        use_conda=False,
        conda_frontend="auto",
        snakemake="snakemake",
    )

    env, shim = runner._snakemake_environment(args)

    assert shim is None
    assert Path(env["PATH"].split(os.pathsep, 1)[0]) == runtime_bin.resolve()


TRIPEPTIDE_SMILES = "CC(C)C[C@@H](N)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"


def test_the_cli_refuses_a_compound_outside_the_documented_scope() -> None:
    """The scope gate used to exist only in the Workbench SMILES field.

    A peptide entered on the command line built a full Snakemake invocation and
    ran to completion, producing a ranked target list indistinguishable from a
    valid one.
    """
    res = run_runner(
        [
            "--preset", "safety",
            "--mode", "fast",
            "--smiles", TRIPEPTIDE_SMILES,
            "--print-command",
        ]
    )

    assert res.returncode != 0
    assert "적용 범위 밖" in (res.stderr + res.stdout)
    assert "snakemake" not in res.stdout


def test_the_cli_refuses_an_out_of_scope_sdf(tmp_path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(TRIPEPTIDE_SMILES)
    AllChem.Compute2DCoords(mol)
    sdf = tmp_path / "peptide.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()

    res = run_runner(
        ["--preset", "safety", "--mode", "fast", "--sdf", str(sdf), "--print-command"]
    )

    assert res.returncode != 0
    assert "적용 범위 밖" in (res.stderr + res.stdout)


def test_the_cli_still_accepts_a_compound_within_scope() -> None:
    res = run_runner(
        [
            "--preset", "safety",
            "--mode", "fast",
            "--smiles", "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "--print-command",
        ]
    )

    assert res.returncode == 0, res.stderr
    assert "snakemake" in res.stdout
