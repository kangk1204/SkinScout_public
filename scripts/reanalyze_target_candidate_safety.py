#!/usr/bin/env python3
"""Re-evaluate target-first candidates with the current Stage 2 safety rules.

The tool keeps applicability, model availability, and the consensus decision
separate.  Live web responses are stored verbatim in per-compound files; the
summary only contains parsed values, hashes, and provenance pointers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import FilterCatalog

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compound_applicability import assess as assess_applicability
from stage2_consensus import (
    _call,
    _husspred_applicability_status,
    skin_sens_decision,
)
from stage2_husspred import (
    API_URL as HUSSPRED_URL,
)
from stage2_husspred import (
    MODEL_OPTIONS as HUSSPRED_OPTIONS,
)
from stage2_husspred import (
    _parse_husspred_applicability,
    _parse_husspred_response,
)
from stage2_pains_brenk import CATALOGS
from stage2_pred_skin import (
    PRED_SKIN_URL,
    _parse_pred_skin_response,
)
from stage2_pred_skin import (
    fetch as fetch_pred_skin,
)
from stage2_stoptox import (
    API_URL as STOPTOX_URL,
)
from stage2_stoptox import (
    WEB_URL as STOPTOX_WEB_URL,
)
from stage2_stoptox import (
    _skin_sens_result,
)
from stage2_stoptox import (
    via_api as fetch_stoptox,
)

SCHEMA = "skinscout.target-candidate-safety.v1"
# 비공개 워크스페이스 경로는 저장소에 두지 않는다(공개 스냅샷 가드).
PRIVATE_QUEUE = os.environ.get("SKINSCOUT_TARGET_QUEUE", "").strip()
PRIVATE_SUMMARY = os.environ.get("SKINSCOUT_TARGET_SUMMARY", "").strip()
DEFAULT_CANDIDATES = Path(PRIVATE_QUEUE) if PRIVATE_QUEUE else None
DEFAULT_HISTORY = Path(PRIVATE_SUMMARY) if PRIVATE_SUMMARY else None
DEFAULT_RUNS = ROOT / "results/runs"
DEFAULT_OUT = ROOT / "results/target_first_20260915/safety"
MODEL_NAMES = ("husspred", "stoptox", "pred_skin")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def canonical_identity(smiles: str) -> tuple[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse input SMILES")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return canonical, Chem.MolToInchiKey(molecule)


def structural_alerts(smiles: str) -> dict[str, object]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse input SMILES for structural alerts")
    matches: dict[str, list[str]] = {name: [] for name in CATALOGS}
    for name, catalog_enum in CATALOGS.items():
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(catalog_enum)
        catalog = FilterCatalog.FilterCatalog(params)
        matches[name] = [
            entry.GetDescription() for entry in catalog.GetMatches(molecule)
        ]
    return {
        "smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        "matches": matches,
        "any_pains": any(matches[name] for name in ("PAINS_A", "PAINS_B", "PAINS_C")),
        "any_brenk": bool(matches["BRENK"]),
        "any_nih": bool(matches["NIH"]),
    }


def _model_payload(name: str, smiles: str, raw: dict[str, object]) -> dict[str, object]:
    if name == "husspred":
        probability, call = _parse_husspred_response(raw)
        applicability = _parse_husspred_applicability(raw)
        return {
            "smiles": smiles,
            "status": "ok",
            "skin_sens_probability": probability,
            "skin_sens_call": call,
            "applicability_domain": applicability,
            "degraded": applicability["status"] != "inside",
            "degraded_reason": (
                ""
                if applicability["status"] == "inside"
                else f"husspred_{applicability['status']}_applicability_domain"
            ),
            "raw": raw,
        }
    if name == "stoptox":
        probability, call = _skin_sens_result(raw)
        return {
            "smiles": smiles,
            "status": "ok",
            "skin_sens_probability": probability,
            "skin_sens_call": call,
            "degraded": False,
            "degraded_reason": "",
            "raw": raw,
        }
    probability, call, result = _parse_pred_skin_response(raw)
    return {
        "smiles": smiles,
        "status": "ok",
        "pred_skin": {"raw": result, "probability": probability, "call": call},
        "consensus_call": call,
        "degraded": False,
        "degraded_reason": "",
    }


def _fetch_husspred(smiles: str) -> dict[str, object]:
    import requests

    response = requests.post(
        HUSSPRED_URL,
        json={"smiles": smiles, "options": HUSSPRED_OPTIONS},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("HuSSPred response is not a JSON object")
    return payload


def _fetch_model(name: str, smiles: str) -> dict[str, object]:
    if name == "husspred":
        return _fetch_husspred(smiles)
    if name == "stoptox":
        payload = fetch_stoptox(smiles)
    else:
        payload = fetch_pred_skin(PRED_SKIN_URL, smiles)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} response is not a JSON object")
    return payload


def fetch_with_retry(
    name: str,
    smiles: str,
    *,
    attempts: int,
    fetcher: Callable[[str, str], dict[str, object]] = _fetch_model,
) -> dict[str, object]:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            raw = fetcher(name, smiles)
            payload = _model_payload(name, smiles, raw)
            return {
                "payload": payload,
                "attempts": attempt,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - batch must preserve each failure
            errors.append(f"{type(exc).__name__}: {exc}"[:1000])
            if attempt < attempts:
                time.sleep(float(attempt))
    return {
        "payload": {
            "smiles": smiles,
            "status": "unavailable",
            "degraded": True,
            "degraded_reason": f"{name}_unavailable",
        },
        "attempts": attempts,
        "elapsed_seconds": None,
        "error": " | ".join(errors),
    }


def fetch_isolated(
    name: str,
    smiles: str,
    *,
    attempts: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """Fetch in a child process so slow streaming cannot evade a wall timeout."""
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                temporary_path = Path(handle.name)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--fetch-one",
                    name,
                    "--fetch-smiles",
                    smiles,
                    "--fetch-out",
                    str(temporary_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    f"isolated fetch exited {completed.returncode}: {detail[:700]}"
                )
            payload = json.loads(temporary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("isolated fetch output is not a JSON object")
            return {
                "payload": payload,
                "attempts": attempt,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - preserve batch failure
            errors.append(f"{type(exc).__name__}: {exc}"[:1000])
            if attempt < attempts:
                time.sleep(float(attempt))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return {
        "payload": {
            "smiles": smiles,
            "status": "unavailable",
            "degraded": True,
            "degraded_reason": f"{name}_unavailable",
        },
        "attempts": attempts,
        "elapsed_seconds": None,
        "error": " | ".join(errors),
    }


def consensus_from_payloads(
    payloads: dict[str, dict[str, object]],
) -> dict[str, object]:
    calls = {name: _call(payloads.get(name, {}), name) for name in MODEL_NAMES}
    ad_status = _husspred_applicability_status(payloads.get("husspred", {}))
    missing = sorted(name for name, call in calls.items() if call is None)
    limited = ["husspred"] if ad_status != "inside" else []
    gaps = sorted(set(missing) | set(limited))
    decision = skin_sens_decision(list(calls.values()))
    if gaps and decision == "PASS":
        decision = "FLAG_HIGH"
    return {
        "decision": decision,
        "calls": calls,
        "degraded": bool(gaps),
        "missing_models": gaps,
        "applicability_limited_models": limited,
        "husspred_applicability_domain": payloads.get("husspred", {}).get(
            "applicability_domain"
        ),
    }


def safety_consensus_is_valid(
    skin_sens: dict[str, object], *, scope: str, web_complete: bool
) -> bool:
    """A decisive two-positive HALT remains valid despite unrelated gaps."""
    if scope == "out_of_scope":
        return False
    calls = skin_sens.get("calls")
    positive_votes = (
        sum(call == "positive" for call in calls.values())
        if isinstance(calls, dict)
        else 0
    )
    if skin_sens.get("decision") == "HALT":
        return positive_votes >= 2
    return web_complete


def _historical_row_map(
    path: Path | None,
) -> dict[tuple[str, str, int], dict[str, object]]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    required = {"gene", "uniprot", "rank", "run_id", "status", "skin_sens"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Historical summary missing columns: {','.join(missing)}")
    result: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row["gene"]), str(row["uniprot"]), int(row["rank"]))
        if key in result:
            raise SystemExit(f"Historical summary repeats key: {key}")
        result[key] = row
    return result


def _input_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns:
        raise SystemExit(f"Candidate file missing smiles column: {path}")
    rows = frame.to_dict(orient="records")
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        value = row.get("smiles")
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"Candidate row {index} has blank smiles")
        canonical, inchikey = canonical_identity(value.strip())
        compound_id = "structure-sha256:" + sha256_text(canonical)
        if compound_id in seen:
            raise SystemExit(
                f"Candidate file repeats exact compound identity: {compound_id}"
            )
        seen.add(compound_id)
        row["input_smiles"] = value.strip()
        row["canonical_smiles"] = canonical
        row["compound_id"] = compound_id
        row["computed_inchikey"] = inchikey
        for key, item in list(row.items()):
            if not isinstance(item, (list, dict)) and pd.isna(item):
                row[key] = None
    return rows


def _historical_provenance(
    row: dict[str, object],
    history: dict[tuple[str, str, int], dict[str, object]],
    runs_dir: Path,
) -> dict[str, object]:
    if not all(field in row for field in ("gene", "uniprot", "rank")):
        return {"available": False, "reason": "no_historical_key"}
    key = (str(row["gene"]), str(row["uniprot"]), int(row["rank"]))
    prior = history.get(key)
    if prior is None:
        return {"available": False, "reason": "historical_key_not_found"}
    run_id = str(prior["run_id"])
    run_dir = runs_dir / run_id
    standardized = run_dir / "01_input/standardized.json"
    artifacts: dict[str, dict[str, object]] = {}
    for name, relative in {
        "standardized": "01_input/standardized.json",
        "admet_ai": "02_admet/admet_ai.json",
        "structural_alerts": "02_admet/structural_alerts.json",
        "husspred": "02_admet/husspred.json",
        "stoptox": "02_admet/stoptox.json",
        "pred_skin": "02_admet/pred_skin.json",
        "admet_report": "02_admet/admet_report.json",
    }.items():
        path = run_dir / relative
        artifacts[name] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    input_bound = False
    if standardized.is_file():
        payload = json.loads(standardized.read_text(encoding="utf-8"))
        recorded = payload.get("input_smiles")
        if isinstance(recorded, str):
            try:
                input_bound = (
                    canonical_identity(recorded)[1] == row["computed_inchikey"]
                )
            except ValueError:
                input_bound = False
    return {
        "available": True,
        "run_id": run_id,
        "old_status": prior.get("status"),
        "old_skin_sens": prior.get("skin_sens"),
        "input_identity_bound": input_bound,
        "artifacts": artifacts,
    }


def reparse_historical_model(
    name: str,
    record: dict[str, object],
    runs_dir: Path,
) -> tuple[dict[str, object], dict[str, object]] | None:
    history = record.get("historical")
    if not isinstance(history, dict) or not history.get("input_identity_bound"):
        return None
    run_id = history.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    source_path = runs_dir / run_id / "02_admet" / f"{name}.json"
    if not source_path.is_file():
        return None
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("status") != "ok":
        return None
    source_smiles = source.get("smiles")
    if not isinstance(source_smiles, str):
        return None
    try:
        if canonical_identity(source_smiles)[0] != record["canonical_smiles"]:
            return None
    except ValueError:
        return None
    if name == "pred_skin":
        raw = (source.get("pred_skin") or {}).get("raw")
    else:
        raw = source.get("raw")
    if not isinstance(raw, dict):
        return None
    payload = _model_payload(name, str(record["canonical_smiles"]), raw)
    evidence = {
        "source_url": _model_urls()[name],
        "execution": "historical_raw_reparsed",
        "reparsed_at": utc_now(),
        "request_smiles_sha256": record["canonical_smiles_sha256"],
        "source_artifact_path": str(source_path.resolve().relative_to(ROOT)),
        "source_artifact_sha256": sha256_file(source_path),
        "source_output_timestamp_from_filesystem_mtime": datetime.fromtimestamp(
            source_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "original_service_retrieval_timestamp": None,
        "timestamp_limitation": (
            "Historical artifact did not record service retrieval time; filesystem "
            "mtime is preserved only as the observed output-write timestamp."
        ),
        "source_payload_limitation": (
            "Historical STopTox payload is the adapter's extracted JSON result, "
            "not the complete HTTP response or source HTML."
            if name == "stoptox"
            else None
        ),
        "error": None,
    }
    return payload, evidence


def _model_urls() -> dict[str, str]:
    return {
        "husspred": HUSSPRED_URL,
        "stoptox": STOPTOX_URL,
        "pred_skin": PRED_SKIN_URL,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def fresh_admet(
    smiles: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    try:
        from admet_ai import ADMETModel
        from admet_ai.constants import DEFAULT_MODELS_DIR

        model = ADMETModel(num_workers=0)
        predictions = model.predict(smiles=smiles)
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("ADMET-AI batch prediction did not return a DataFrame")
        by_smiles = {
            str(index): {
                str(key): (None if pd.isna(value) else float(value))
                for key, value in values.items()
            }
            for index, values in predictions.to_dict(orient="index").items()
        }
        model_files = sorted(Path(DEFAULT_MODELS_DIR).glob("**/*.pt"))
        inventory = [
            {
                "path": str(path.relative_to(DEFAULT_MODELS_DIR)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_files
        ]
        return by_smiles, {
            "status": "ok",
            "execution": "fresh_cpu_batch",
            "package_version": _package_version("admet-ai"),
            "model_directory": str(DEFAULT_MODELS_DIR),
            "model_inventory": inventory,
            "model_inventory_sha256": sha256_text(
                json.dumps(inventory, sort_keys=True, separators=(",", ":"))
            ),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    except Exception as exc:  # noqa: BLE001 - failure belongs in the batch manifest
        return {}, {
            "status": "unavailable",
            "execution": "fresh_cpu_batch_failed",
            "package_version": _package_version("admet-ai"),
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }


def flatten_row(row: dict[str, object]) -> dict[str, object]:
    applicability = row["applicability"]
    skin = row["skin_sens"]
    alerts = row["structural_alerts"]
    calls = skin["calls"]
    old = row["historical"]
    admet = row.get("admet_predictions") or {}
    properties = applicability.get("properties") or {}
    return {
        "compound_id": row["compound_id"],
        "input_smiles": row["input_smiles"],
        "canonical_smiles": row["canonical_smiles"],
        "computed_inchikey": row["computed_inchikey"],
        "candidate_name": row.get("candidate_name") or row.get("inci_name"),
        "role": row.get("role"),
        "gene": row.get("gene"),
        "uniprot": row.get("uniprot"),
        "target_label": row.get("target_label"),
        "rank": row.get("rank"),
        "run_valid": row["run_valid"],
        "safety_consensus_valid": row["safety_consensus_valid"],
        "full_analysis_complete": row["full_analysis_complete"],
        "decision": row["decision"],
        "applicability_domain": row["applicability_domain"],
        "applicability_verdict": applicability["verdict"],
        "applicability_exclusion_codes": ";".join(
            str(item["code"]) for item in applicability.get("exclusions", [])
        ),
        "applicability_warning_codes": ";".join(
            str(item["code"]) for item in applicability.get("warnings", [])
        ),
        "molecular_weight": properties.get("molecular_weight"),
        "logp": properties.get("logp"),
        "tpsa": properties.get("tpsa"),
        "skin_sens_decision": skin["decision"],
        "husspred_call": calls.get("husspred"),
        "husspred_ad_status": (skin.get("husspred_applicability_domain") or {}).get(
            "status"
        ),
        "stoptox_call": calls.get("stoptox"),
        "pred_skin_call": calls.get("pred_skin"),
        "skin_sens_evidence_gaps": ";".join(skin.get("missing_models", [])),
        "any_pains": alerts.get("any_pains"),
        "any_brenk": alerts.get("any_brenk"),
        "any_nih": alerts.get("any_nih"),
        "admet_status": row.get("admet_status"),
        "admet_skin_reaction": admet.get("Skin_Reaction"),
        "admet_ames": admet.get("AMES"),
        "admet_clintox": admet.get("ClinTox"),
        "admet_dili": admet.get("DILI"),
        "admet_herg": admet.get("hERG"),
        "old_status": old.get("old_status"),
        "old_skin_sens": old.get("old_skin_sens"),
        "decision_changed": row.get("decision_changed"),
        "provenance_complete": row["provenance_complete"],
    }


def model_evidence_rank(
    evidence: dict[str, object], payload: dict[str, object], priority: int
) -> tuple[bool, bool, str, int]:
    """Rank model evidence without treating reparse/write time as retrieval time."""
    retrieval_timestamp = evidence.get("retrieved_at") or evidence.get(
        "original_service_retrieval_timestamp"
    )
    known_retrieval = isinstance(retrieval_timestamp, str) and bool(
        retrieval_timestamp.strip()
    )
    return (
        payload.get("status") == "ok",
        known_retrieval,
        retrieval_timestamp.strip() if known_retrieval else "",
        priority,
    )


def _batch_expected_row_count(manifest: dict[str, object]) -> int | None:
    schema = str(manifest.get("schema", ""))
    keys = (
        ("selected_unique_compound_count",)
        if schema.endswith(".reconciliation")
        else ("candidate_count", "selected_unique_compound_count")
    )
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def validated_batch_rows(path: Path) -> list[dict[str, object]]:
    """Read a safety JSONL only when its sibling manifest binds hash and rows.

    C21: reconciliation must not seal tampered or mismatched batches as a normal
    recomposition result.
    """
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Safety batch manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Safety batch manifest failed to parse: {manifest_path}: {exc}"
        ) from exc
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    artifact = (
        artifacts.get("candidate_safety_jsonl")
        if isinstance(artifacts, dict)
        else None
    )
    expected_hash = artifact.get("sha256") if isinstance(artifact, dict) else None
    if not isinstance(expected_hash, str) or not expected_hash:
        raise SystemExit(
            f"Safety batch manifest has no candidate_safety_jsonl hash: {manifest_path}"
        )
    if sha256_file(path) != expected_hash:
        raise SystemExit(f"Safety batch JSONL hash mismatch: {path}")
    expected_count = _batch_expected_row_count(manifest)
    if expected_count is None:
        raise SystemExit(f"Safety batch manifest has no row count: {manifest_path}")
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(lines) != expected_count:
        raise SystemExit(
            f"Safety batch row count mismatch: {path} "
            f"manifest={expected_count} actual={len(lines)}"
        )
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid safety JSONL row: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise SystemExit(f"Invalid safety JSONL row: {path}:{line_number}")
        rows.append(row)
    return rows


def validated_model_artifact(
    name: str,
    evidence: dict[str, object],
    row: dict[str, object],
    source_path: Path,
) -> dict[str, object]:
    """Verify one model envelope's bytes, schema, model, and request identity."""
    artifact_path = evidence.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise SystemExit(f"Model evidence lacks an artifact path: {source_path}:{name}")
    artifact = ROOT / artifact_path
    expected_hash = evidence.get("artifact_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise SystemExit(f"Model evidence lacks an artifact hash: {artifact}")
    if not artifact.is_file():
        raise SystemExit(f"Model artifact is missing: {artifact}")
    if sha256_file(artifact) != expected_hash:
        raise SystemExit(f"Model artifact hash mismatch: {artifact}")
    try:
        envelope = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Model artifact failed to parse: {artifact}: {exc}") from exc
    if not isinstance(envelope, dict):
        raise SystemExit(f"Model artifact is not an object: {artifact}")
    if envelope.get("schema") != SCHEMA + ".model-response":
        raise SystemExit(f"Model artifact schema mismatch: {artifact}")
    if envelope.get("model") != name:
        raise SystemExit(f"Model artifact model mismatch: {artifact}")
    expected_request_hash = row.get("canonical_smiles_sha256")
    if not isinstance(expected_request_hash, str) or not expected_request_hash:
        raise SystemExit(
            f"Safety row lacks canonical_smiles_sha256 for request binding: {source_path}"
        )
    if envelope.get("request_smiles_sha256") != expected_request_hash:
        raise SystemExit(f"Model artifact request hash mismatch: {artifact}")
    request_smiles = envelope.get("request_smiles")
    if (
        not isinstance(request_smiles, str)
        or sha256_text(request_smiles) != expected_request_hash
    ):
        raise SystemExit(f"Model artifact request identity mismatch: {artifact}")
    evidence_request_hash = evidence.get("request_smiles_sha256")
    if evidence_request_hash is not None and evidence_request_hash != expected_request_hash:
        raise SystemExit(f"Model evidence request hash mismatch: {artifact}")
    payload = envelope.get("parsed_payload")
    if not isinstance(payload, dict):
        raise SystemExit(f"Model artifact payload is missing: {artifact}")
    return payload


def reconcile_safety_batches(
    paths: list[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Reconcile exact structures using the newest valid evidence per model."""
    grouped: dict[str, list[tuple[int, Path, dict[str, object]]]] = {}
    for priority, path in enumerate(paths):
        for line_number, row in enumerate(validated_batch_rows(path), start=1):
            compound_id = row.get("compound_id")
            if not isinstance(compound_id, str):
                raise SystemExit(f"Invalid safety JSONL row: {path}:{line_number}")
            grouped.setdefault(compound_id, []).append((priority, path, row))

    selected: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for compound_id, candidates in grouped.items():
        canonical_values = {str(item[2].get("canonical_smiles")) for item in candidates}
        if len(canonical_values) != 1:
            raise SystemExit(
                f"Compound ID has conflicting canonical structures: {compound_id}"
            )
        decisions = {str(item[2].get("decision")) for item in candidates}
        if len(decisions) != 1:
            raise SystemExit(
                f"Compound ID has conflicting safety decisions: {compound_id}"
            )

        def quality(
            item: tuple[int, Path, dict[str, object]],
        ) -> tuple[bool, bool, bool, bool, int]:
            priority, _, row = item
            return (
                bool(row.get("safety_consensus_valid")),
                bool(row.get("full_analysis_complete")),
                bool(row.get("run_valid")),
                bool(row.get("provenance_complete")),
                priority,
            )

        chosen = max(candidates, key=quality)
        chosen_row = dict(chosen[2])
        model_selection: dict[str, dict[str, object]] = {}
        if len(candidates) > 1:
            merged_payloads: dict[str, dict[str, object]] = {}
            merged_evidence: dict[str, dict[str, object]] = {}
            for name in MODEL_NAMES:
                options = []
                for priority, source_path, row in candidates:
                    model_results = row.get("model_results", {})
                    evidence = (
                        model_results.get(name)
                        if isinstance(model_results, dict)
                        else None
                    )
                    if not isinstance(evidence, dict):
                        continue
                    payload = validated_model_artifact(name, evidence, row, source_path)
                    rank = model_evidence_rank(evidence, payload, priority)
                    options.append(
                        (
                            rank,
                            source_path,
                            evidence,
                            payload,
                        )
                    )
                if not options:
                    continue
                model_choice = max(options, key=lambda item: item[0])
                rank, source_path, evidence, payload = model_choice
                merged_payloads[name] = payload
                merged_evidence[name] = evidence
                model_selection[name] = {
                    "selected_from": str(source_path),
                    "artifact_path": evidence.get("artifact_path"),
                    "execution": evidence.get("execution"),
                    "status": payload.get("status"),
                    "evidence_timestamp": rank[2] or None,
                    "evidence_timestamp_status": "known" if rank[1] else "unknown",
                    "reparsed_at": evidence.get("reparsed_at"),
                    "reason": (
                        "valid status first, then known actual service retrieval timestamp "
                        "(unknown ranks below known), then later input"
                    ),
                    "all_attempts": [
                        {
                            "source": str(item[1]),
                            "status": item[3].get("status"),
                            "evidence_timestamp": item[0][2] or None,
                            "evidence_timestamp_status": (
                                "known" if item[0][1] else "unknown"
                            ),
                            "reparsed_at": item[2].get("reparsed_at"),
                            "execution": item[2].get("execution"),
                            "error": item[2].get("error"),
                            "artifact_path": item[2].get("artifact_path"),
                        }
                        for item in options
                    ],
                }
            skin = consensus_from_payloads(merged_payloads)
            chosen_row["model_results"] = merged_evidence
            chosen_row["skin_sens"] = skin
            chosen_row["decision"] = skin["decision"]
            web_complete = len(merged_payloads) == 3 and all(
                payload.get("status") == "ok" for payload in merged_payloads.values()
            )
            admet_complete = chosen_row.get("admet_status") == "ok"
            chosen_row["provenance_complete"] = bool(web_complete and admet_complete)
            chosen_row["full_analysis_complete"] = bool(web_complete and admet_complete)
            chosen_row["run_valid"] = chosen_row["full_analysis_complete"]
            chosen_row["safety_consensus_valid"] = safety_consensus_is_valid(
                skin,
                scope=str(chosen_row["applicability"]["verdict"]),
                web_complete=web_complete,
            )
        chosen_row["reconciliation"] = {
            "base_row_from": str(chosen[1]),
            "selection_quality": list(quality(chosen)),
            "model_selection": model_selection,
            "reason": (
                "base metadata selected by validity/completeness tuple; duplicate model "
                "evidence selected independently by valid status and timestamp; current "
                "consensus recomputed from selected payloads"
            ),
        }
        selected.append(chosen_row)
        if len(candidates) > 1:
            comparisons.append(
                {
                    "compound_id": compound_id,
                    "canonical_smiles": next(iter(canonical_values)),
                    "input_decisions": sorted(decisions),
                    "reconciled_decision": chosen_row["decision"],
                    "base_row_from": str(chosen[1]),
                    "model_selection": model_selection,
                    "inputs": [
                        {
                            "path": str(path),
                            "quality": list(quality(item)),
                            "decision": row.get("decision"),
                            "safety_consensus_valid": row.get("safety_consensus_valid"),
                            "full_analysis_complete": row.get("full_analysis_complete"),
                            "run_valid": row.get("run_valid"),
                            "provenance_complete": row.get("provenance_complete"),
                        }
                        for item in candidates
                        for _, path, row in [item]
                    ],
                }
            )
    selected.sort(key=lambda row: str(row["compound_id"]))
    comparisons.sort(key=lambda row: str(row["compound_id"]))
    return selected, comparisons


def write_reconciliation(paths: list[Path], out_dir: Path) -> None:
    rows, comparisons = reconcile_safety_batches(paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "candidate_safety.jsonl"
    csv_path = out_dir / "candidate_safety.csv"
    write_jsonl_atomic(jsonl_path, rows)
    flat = [flatten_row(row) for row in rows]
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    temporary_csv.replace(csv_path)
    manifest = {
        "schema": SCHEMA + ".reconciliation",
        "created_at": utc_now(),
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
        "input_row_count": sum(
            1 for path in paths for _ in path.open(encoding="utf-8")
        ),
        "selected_unique_compound_count": len(rows),
        "duplicate_compound_count": len(comparisons),
        "duplicate_reconciliation": comparisons,
        "code_sha256": sha256_file(Path(__file__)),
        "artifacts": {
            "candidate_safety_jsonl": {
                "path": str(jsonl_path),
                "sha256": sha256_file(jsonl_path),
            },
            "candidate_safety_csv": {
                "path": str(csv_path),
                "sha256": sha256_file(csv_path),
            },
        },
        "limitations": [
            "Historical STopTox cache stores the adapter's extracted JSON result, not full HTTP/HTML.",
            "When original service retrieval time is absent, evidence_timestamp remains unknown; reparse time is not used as retrieval time.",
        ],
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "selected_unique_compound_count": len(rows),
                "duplicate_compound_count": len(comparisons),
                "manifest": str(out_dir / "manifest.json"),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--historical-summary", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--skip-live-web", action="store_true")
    parser.add_argument("--skip-admet", action="store_true")
    parser.add_argument(
        "--reparse-historical-stoptox",
        action="store_true",
        help="Use input-bound historical STopTox raw responses with the current parser.",
    )
    parser.add_argument(
        "--stoptox-circuit-open",
        action="store_true",
        help=(
            "Do not issue uncached STopTox calls after a documented batch outage; "
            "record each remaining result as unavailable."
        ),
    )
    parser.add_argument("--fetch-one", choices=MODEL_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--fetch-smiles", help=argparse.SUPPRESS)
    parser.add_argument("--fetch-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reconcile-jsonl", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fetch_one:
        if not args.fetch_smiles or args.fetch_out is None:
            raise SystemExit("--fetch-one requires --fetch-smiles and --fetch-out")
        raw = _fetch_model(args.fetch_one, args.fetch_smiles)
        write_json_atomic(
            args.fetch_out, _model_payload(args.fetch_one, args.fetch_smiles, raw)
        )
        return
    if args.reconcile_jsonl:
        if len(args.reconcile_jsonl) < 2:
            raise SystemExit("--reconcile-jsonl requires at least two input files")
        write_reconciliation(args.reconcile_jsonl, args.out_dir)
        return
    if args.workers < 1 or args.workers > 12:
        raise SystemExit("--workers must be between 1 and 12")
    if args.attempts < 1 or args.attempts > 3:
        raise SystemExit("--attempts must be between 1 and 3")
    if args.request_timeout_seconds <= 0 or args.request_timeout_seconds > 600:
        raise SystemExit("--request-timeout-seconds must be in (0, 600]")
    if args.candidates is None:
        raise SystemExit("--candidates 또는 SKINSCOUT_TARGET_QUEUE 환경변수가 필요합니다")
    if args.historical_summary is None:
        raise SystemExit(
            "--historical-summary 또는 SKINSCOUT_TARGET_SUMMARY 환경변수가 필요합니다"
        )
    started_at = utc_now()
    rows = _input_rows(args.candidates)
    history_path = (
        args.historical_summary if args.historical_summary.is_file() else None
    )
    history = _historical_row_map(history_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"

    records: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    for source in rows:
        applicability = assess_applicability(str(source["input_smiles"]))
        alerts = structural_alerts(str(source["canonical_smiles"]))
        history_info = _historical_provenance(source, history, args.runs_dir)
        record = {
            **source,
            "schema": SCHEMA,
            "input_smiles_sha256": sha256_text(str(source["input_smiles"])),
            "canonical_smiles_sha256": sha256_text(str(source["canonical_smiles"])),
            "applicability": applicability,
            "structural_alerts": alerts,
            "historical": history_info,
            "model_results": {},
        }
        records.append(record)
        if applicability["verdict"] != "out_of_scope":
            eligible.append(record)

    if not args.skip_live_web:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for record in eligible:
                for name in MODEL_NAMES:
                    raw_path = raw_dir / str(record["compound_id"]) / f"{name}.json"
                    if name == "stoptox" and args.reparse_historical_stoptox:
                        replay = reparse_historical_model(name, record, args.runs_dir)
                        if replay is not None:
                            payload, evidence = replay
                            envelope = {
                                "schema": SCHEMA + ".model-response",
                                "model": name,
                                **evidence,
                                "request_smiles": record["canonical_smiles"],
                                "parsed_payload": payload,
                            }
                            write_json_atomic(raw_path, envelope)
                            evidence["artifact_path"] = str(
                                raw_path.resolve().relative_to(ROOT)
                            )
                            evidence["artifact_sha256"] = sha256_file(raw_path)
                            record["model_results"][name] = evidence
                            record.setdefault("_model_payloads", {})[name] = payload
                            continue
                    if raw_path.is_file():
                        try:
                            envelope = json.loads(raw_path.read_text(encoding="utf-8"))
                            payload = envelope["parsed_payload"]
                            if (
                                envelope.get("schema") == SCHEMA + ".model-response"
                                and envelope.get("model") == name
                                and envelope.get("request_smiles_sha256")
                                == record["canonical_smiles_sha256"]
                                and isinstance(payload, dict)
                            ):
                                record["model_results"][name] = {
                                    "source_url": _model_urls()[name],
                                    "execution": "resumed_live_refresh",
                                    "retrieved_at": envelope.get("retrieved_at"),
                                    "request_smiles_sha256": record[
                                        "canonical_smiles_sha256"
                                    ],
                                    "artifact_path": str(
                                        raw_path.resolve().relative_to(ROOT)
                                    ),
                                    "artifact_sha256": sha256_file(raw_path),
                                    "attempts": envelope.get("attempts"),
                                    "elapsed_seconds": envelope.get("elapsed_seconds"),
                                    "error": envelope.get("error"),
                                }
                                record.setdefault("_model_payloads", {})[name] = payload
                                continue
                        except (KeyError, TypeError, json.JSONDecodeError):
                            pass
                    if name == "stoptox" and args.stoptox_circuit_open:
                        payload = {
                            "smiles": record["canonical_smiles"],
                            "status": "unavailable",
                            "degraded": True,
                            "degraded_reason": "stoptox_current_batch_service_outage",
                        }
                        evidence = {
                            "source_url": STOPTOX_URL,
                            "execution": "not_called_current_batch_service_outage",
                            "recorded_at": utc_now(),
                            "request_smiles_sha256": record["canonical_smiles_sha256"],
                            "error": (
                                "Circuit opened after six distinct candidates each failed "
                                "two bounded live attempts (API 404 followed by web fallback "
                                "failure/timeout); no equivalent historical raw response was "
                                "bound to this exact structure."
                            ),
                        }
                        envelope = {
                            "schema": SCHEMA + ".model-response",
                            "model": name,
                            **evidence,
                            "request_smiles": record["canonical_smiles"],
                            "parsed_payload": payload,
                        }
                        write_json_atomic(raw_path, envelope)
                        evidence["artifact_path"] = str(
                            raw_path.resolve().relative_to(ROOT)
                        )
                        evidence["artifact_sha256"] = sha256_file(raw_path)
                        record["model_results"][name] = evidence
                        record.setdefault("_model_payloads", {})[name] = payload
                        continue
                    future = executor.submit(
                        fetch_isolated,
                        name,
                        str(record["canonical_smiles"]),
                        attempts=args.attempts,
                        timeout_seconds=args.request_timeout_seconds,
                    )
                    futures[future] = (record, name)
            for future in as_completed(futures):
                record, name = futures[future]
                result = future.result()
                payload = result.pop("payload")
                raw_path = raw_dir / str(record["compound_id"]) / f"{name}.json"
                retrieved_at = utc_now()
                source_urls = (
                    [STOPTOX_URL, STOPTOX_WEB_URL]
                    if name == "stoptox"
                    else [_model_urls()[name]]
                )
                envelope = {
                    "schema": SCHEMA + ".model-response",
                    "model": name,
                    "execution": "live_refresh",
                    "retrieved_at": retrieved_at,
                    "source_urls": source_urls,
                    "request_smiles": record["canonical_smiles"],
                    "request_smiles_sha256": record["canonical_smiles_sha256"],
                    "attempts": result["attempts"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "error": result["error"],
                    "parsed_payload": payload,
                }
                write_json_atomic(raw_path, envelope)
                result.update(
                    {
                        "source_url": _model_urls()[name],
                        "execution": "live_refresh",
                        "retrieved_at": retrieved_at,
                        "request_smiles_sha256": record["canonical_smiles_sha256"],
                        "artifact_path": str(raw_path.resolve().relative_to(ROOT)),
                        "artifact_sha256": sha256_file(raw_path),
                    }
                )
                record["model_results"][name] = result
                record.setdefault("_model_payloads", {})[name] = payload

    admet_by_smiles: dict[str, dict[str, object]] = {}
    if args.skip_admet:
        admet_provenance = {"status": "skipped", "execution": "operator_requested_skip"}
    else:
        admet_by_smiles, admet_provenance = fresh_admet(
            [str(record["canonical_smiles"]) for record in eligible]
        )

    for record in records:
        scope = str(record["applicability"]["verdict"])
        if scope == "out_of_scope":
            skin = {
                "decision": "UNAVAILABLE",
                "calls": {name: None for name in MODEL_NAMES},
                "degraded": True,
                "missing_models": list(MODEL_NAMES),
                "applicability_limited_models": [],
                "husspred_applicability_domain": None,
                "reason": "models_not_run_for_out_of_scope_input",
            }
        elif args.skip_live_web:
            skin = {
                "decision": "UNAVAILABLE",
                "calls": {name: None for name in MODEL_NAMES},
                "degraded": True,
                "missing_models": list(MODEL_NAMES),
                "applicability_limited_models": [],
                "husspred_applicability_domain": None,
                "reason": "live_web_skipped",
            }
        else:
            skin = consensus_from_payloads(
                {
                    name: payload
                    for name, payload in record.get("_model_payloads", {}).items()
                }
            )
        record["skin_sens"] = skin
        record["decision"] = skin["decision"]
        record["applicability_domain"] = {
            "in_scope": "inside",
            "review": "review",
            "out_of_scope": "outside",
        }[scope]
        record["admet_predictions"] = admet_by_smiles.get(
            str(record["canonical_smiles"]), {}
        )
        if scope == "out_of_scope":
            record["admet_status"] = "not_run_out_of_scope"
        elif str(record["canonical_smiles"]) in admet_by_smiles:
            record["admet_status"] = "ok"
        else:
            record["admet_status"] = admet_provenance["status"]
        model_payloads = record.get("_model_payloads", {})
        web_complete = scope == "out_of_scope" or (
            len(model_payloads) == 3
            and all(
                payload.get("status") == "ok" for payload in model_payloads.values()
            )
        )
        admet_complete = scope == "out_of_scope" or record["admet_status"] == "ok"
        record["provenance_complete"] = bool(web_complete and admet_complete)
        record["full_analysis_complete"] = bool(
            scope != "out_of_scope" and web_complete and admet_complete
        )
        record["safety_consensus_valid"] = safety_consensus_is_valid(
            skin, scope=scope, web_complete=web_complete
        )
        record["run_valid"] = record["full_analysis_complete"]
        old_decision = record["historical"].get("old_skin_sens")
        record["decision_changed"] = (
            old_decision != record["decision"] if old_decision is not None else None
        )
        record.pop("_model_payloads", None)

    code_paths = [
        Path(__file__),
        SCRIPTS / "compound_applicability.py",
        SCRIPTS / "stage2_consensus.py",
        SCRIPTS / "stage2_husspred.py",
        SCRIPTS / "stage2_stoptox.py",
        SCRIPTS / "stage2_pred_skin.py",
        SCRIPTS / "stage2_pains_brenk.py",
    ]
    counts: dict[str, dict[str, int]] = {}
    for field in ("applicability_domain", "decision", "admet_status"):
        values: dict[str, int] = {}
        for record in records:
            key = str(record[field])
            values[key] = values.get(key, 0) + 1
        counts[field] = values
    model_execution_counts: dict[str, dict[str, int]] = {
        name: {} for name in MODEL_NAMES
    }
    for record in records:
        for name, result in record["model_results"].items():
            execution = str(result.get("execution", "unknown"))
            current = model_execution_counts[name]
            current[execution] = current.get(execution, 0) + 1
    manifest = {
        "schema": SCHEMA,
        "started_at": started_at,
        "completed_at": utc_now(),
        "candidate_file": str(args.candidates),
        "candidate_file_sha256": sha256_file(args.candidates),
        "historical_summary": str(history_path) if history_path else None,
        "historical_summary_sha256": sha256_file(history_path)
        if history_path
        else None,
        "candidate_count": len(records),
        "web_eligible_count": len(eligible),
        "web_execution": "skipped" if args.skip_live_web else "live_refresh",
        "stoptox_execution": (
            "historical_raw_reparsed_when_input_bound_else_live"
            if args.reparse_historical_stoptox
            else "live_refresh"
        ),
        "stoptox_circuit_open": args.stoptox_circuit_open,
        "workers": args.workers,
        "max_attempts": args.attempts,
        "hard_request_timeout_seconds": args.request_timeout_seconds,
        "counts": counts,
        "model_execution_counts": model_execution_counts,
        "valid_run_count": sum(bool(record["run_valid"]) for record in records),
        "valid_safety_consensus_count": sum(
            bool(record["safety_consensus_valid"]) for record in records
        ),
        "provenance_complete_count": sum(
            bool(record["provenance_complete"]) for record in records
        ),
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
        },
        "runtime": {
            "python": sys.version,
            "rdkit": rdBase.rdkitVersion,
            "pandas": pd.__version__,
            "requests": _package_version("requests"),
        },
        "model_sources": {
            **_model_urls(),
            "stoptox_fallback": STOPTOX_WEB_URL,
        },
        "admet": admet_provenance,
        "interpretation": {
            "PASS": "current computational screen; not experimental safety confirmation",
            "FLAG_HIGH": "additional review/testing required",
            "HALT": "excluded by current sensitization consensus policy",
            "UNAVAILABLE": "no valid consensus because analysis was skipped or unsupported",
        },
    }
    jsonl_path = args.out_dir / "candidate_safety.jsonl"
    csv_path = args.out_dir / "candidate_safety.csv"
    manifest_path = args.out_dir / "manifest.json"
    write_jsonl_atomic(jsonl_path, records)
    flat = [flatten_row(record) for record in records]
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    temporary_csv.replace(csv_path)
    manifest["artifacts"] = {
        "candidate_safety_jsonl": {
            "path": str(jsonl_path),
            "sha256": sha256_file(jsonl_path),
        },
        "candidate_safety_csv": {
            "path": str(csv_path),
            "sha256": sha256_file(csv_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "candidate_count": len(records),
                "counts": counts,
                "valid_run_count": manifest["valid_run_count"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
