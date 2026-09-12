#!/usr/bin/env bash
# eval/run_all.sh
# Drive every benchmark + leakage audit referenced in INSTRUCTIONS §14 + §18.
# Inputs default to results/eval/ — override via env vars.

set -euo pipefail

ROOT="${COSMAX_ROOT:-$(pwd)}"
OUT="${OUT:-results/eval}"
LOG="${LOG:-results/logs/eval}"
RANKINGS="${RANKINGS:-${OUT}/rankings}"
ALLOW_PARTIAL="${EVAL_ALLOW_PARTIAL:-0}"
ALLOW_INCOMPLETE_LEAKAGE="${EVAL_ALLOW_INCOMPLETE_LEAKAGE:-0}"
ALLOW_THRESHOLD_FAILURE="${EVAL_ALLOW_THRESHOLD_FAILURE:-0}"
CHEMBL_DIR="${CHEMBL_DIR:-data/chembl37}"
TARGET_CLASSES="${TARGET_CLASSES:-${CHEMBL_DIR}/target_classes.parquet}"
CHEMBL_FP_PARQUET="${CHEMBL_FP_PARQUET:-${CHEMBL_DIR}/fp_morgan2_2048.parquet}"
MMSEQS_TRAINING_DB="${MMSEQS_TRAINING_DB:-data/mmseqs/training_cutoff_db}"
TRAINING_LIGANDS="${TRAINING_LIGANDS:-${CHEMBL_DIR}/training_ligands.smi}"
TRAINING_HOLO="${TRAINING_HOLO:-data/plinder/training_holo_pockets.csv}"
COLLECTED_RUNS_MANIFEST="${COLLECTED_RUNS_MANIFEST:-${OUT}/collected_runs.json}"
SKIN_KNOWN_CASES="${SKIN_KNOWN_CASES:-data/validation/skin_known_target_panel.csv}"
SKIN_KNOWN_RANKINGS="${SKIN_KNOWN_RANKINGS:-${RANKINGS}/skin_known_target}"
SKIN_KNOWN_RUN_LEDGER="${SKIN_KNOWN_RUN_LEDGER:-${OUT}/skin_known_target_run_ledger.csv}"
SOTA_BASELINES="${SOTA_BASELINES:-${OUT}/sota_baselines.csv}"
SOTA_ABLATIONS="${SOTA_ABLATIONS:-${OUT}/sota_ablations.csv}"
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-workflow/config.yaml}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT}"
if [ ! -e "${WORKFLOW_CONFIG}" ] && [ "${ALLOW_PARTIAL}" = "1" ]; then
    echo "[eval][WARN] missing workflow config; using repo defaults for partial diagnostics: ${WORKFLOW_CONFIG}" >&2
    WORKFLOW_CONFIG="${SCRIPT_ROOT}/workflow/config.yaml"
fi

config_value() {
    local dotted="$1"
    local default="$2"
    "${PYTHON_BIN}" - "$dotted" "$default" "$WORKFLOW_CONFIG" <<'PY'
import sys
from pathlib import Path

key, default, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
path = Path(config_path)
if not path.exists():
    print(default)
    raise SystemExit(0)
try:
    import yaml
    payload = yaml.safe_load(path.read_text())
except Exception as exc:
    raise SystemExit(f"invalid evaluation workflow config {path}: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit(f"evaluation workflow config root must be a mapping: {path}")
value = payload
try:
    for part in key.split("."):
        value = value[part]
except (KeyError, TypeError):
    print(default)
else:
    print(value)
PY
}

EVAL_SEQ_ID_THRESHOLD="${EVAL_SEQ_ID_THRESHOLD:-$(config_value evaluation.leakage.seq_id_threshold 0.30)}"
EVAL_LIGAND_TANIMOTO_THRESHOLD="${EVAL_LIGAND_TANIMOTO_THRESHOLD:-$(config_value evaluation.leakage.ligand_tanimoto_threshold 0.50)}"
EVAL_POCKET_SUCOS_THRESHOLD="${EVAL_POCKET_SUCOS_THRESHOLD:-$(config_value evaluation.leakage.pocket_sucos_threshold 0.50)}"
EVAL_COLD_START_MIN_RECALL_AT_50="${EVAL_COLD_START_MIN_RECALL_AT_50:-$(config_value evaluation.thresholds.cold_start_min_recall_at_50 0.01)}"
EVAL_COSMETIC_RETRO_MIN_MEAN_TOP10="${EVAL_COSMETIC_RETRO_MIN_MEAN_TOP10:-$(config_value evaluation.thresholds.cosmetic_retro_min_mean_top10 0.01)}"
EVAL_SKIN_KNOWN_MIN_CASE_TOP10="${EVAL_SKIN_KNOWN_MIN_CASE_TOP10:-$(config_value evaluation.sota.skin_known_min_case_top10 0.80)}"
EVAL_SKIN_KNOWN_MIN_TARGET_TOP10="${EVAL_SKIN_KNOWN_MIN_TARGET_TOP10:-$(config_value evaluation.sota.skin_known_min_target_top10 0.50)}"
EVAL_SKIN_KNOWN_MIN_TARGET_TOP30="${EVAL_SKIN_KNOWN_MIN_TARGET_TOP30:-$(config_value evaluation.sota.skin_known_min_target_top30 0.60)}"
EVAL_SKIN_KNOWN_CONTEXT_PROFILE="${EVAL_SKIN_KNOWN_CONTEXT_PROFILE:-$(config_value evaluation.sota.default_context_profile auto)}"
EVAL_SKIN_EFFICACY_MIN_MEAN_PRECISION="${EVAL_SKIN_EFFICACY_MIN_MEAN_PRECISION:-$(config_value evaluation.thresholds.skin_efficacy_min_mean_precision 0.01)}"
EVAL_SKIN_EFFICACY_MIN_MEAN_RECALL="${EVAL_SKIN_EFFICACY_MIN_MEAN_RECALL:-$(config_value evaluation.thresholds.skin_efficacy_min_mean_recall 0.01)}"
EVAL_ANALOG_MIN_RECOVERY="${EVAL_ANALOG_MIN_RECOVERY:-$(config_value evaluation.thresholds.analog_min_recovery 0.01)}"
EVAL_ANALOG_MIN_NOVELTY="${EVAL_ANALOG_MIN_NOVELTY:-$(config_value evaluation.thresholds.analog_min_novelty 0.80)}"
EVAL_ANALOG_MIN_MEAN_RA_SCORE="${EVAL_ANALOG_MIN_MEAN_RA_SCORE:-$(config_value evaluation.thresholds.analog_min_mean_ra_score 0.70)}"
EVAL_PHARMACOPHORE_MIN_PRESERVED_FRACTION="${EVAL_PHARMACOPHORE_MIN_PRESERVED_FRACTION:-$(config_value evaluation.thresholds.pharmacophore_min_preserved_fraction 0.80)}"
ACTIVITY_RETRIEVAL_REQUIRED="${ACTIVITY_RETRIEVAL_REQUIRED:-$(config_value evaluation.activity_retrieval.required true)}"
ACTIVITY_BENCHMARK_MANIFEST="${ACTIVITY_BENCHMARK_MANIFEST:-$(config_value evaluation.activity_retrieval.benchmark_dir data/activity_benchmark_202608)/manifest.json}"
ACTIVITY_RETRIEVAL_INDEX_MANIFEST="${ACTIVITY_RETRIEVAL_INDEX_MANIFEST:-$(config_value evaluation.activity_retrieval.retrieval_index_dir data/activity_retrieval_202608)/manifest.json}"
ACTIVITY_RECOVERY_PANELS_MANIFEST="${ACTIVITY_RECOVERY_PANELS_MANIFEST:-$(config_value evaluation.activity_retrieval.panels_dir data/activity_recovery_panels_202608)/manifest.json}"
ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST="${ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST:-$(config_value evaluation.activity_retrieval.dev_selection_dir results/eval/activity_retrieval_202608/dev_selection)/manifest.json}"
ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST="${ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST:-$(config_value evaluation.activity_retrieval.final_evaluation_dir results/eval/activity_retrieval_202608/test_evaluation)/manifest.json}"
ACTIVITY_RCSB_CONTACT_POCKET_LEAKAGE_MANIFEST="${ACTIVITY_RCSB_CONTACT_POCKET_LEAKAGE_MANIFEST:-$(config_value evaluation.activity_retrieval.rcsb_contact_pocket_leakage_manifest results/audits/rcsb_contact_pocket_leakage.manifest.json)}"
ACTIVITY_RETRIEVAL_GATE="${ACTIVITY_RETRIEVAL_GATE:-data/manifests/activity_retrieval_final_gate.flag}"

abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "${ROOT}/$1" ;;
    esac
}

OUT="$(abspath "${OUT}")"
LOG="$(abspath "${LOG}")"
RANKINGS="$(abspath "${RANKINGS}")"
CHEMBL_DIR="$(abspath "${CHEMBL_DIR}")"
TARGET_CLASSES="$(abspath "${TARGET_CLASSES}")"
CHEMBL_FP_PARQUET="$(abspath "${CHEMBL_FP_PARQUET}")"
MMSEQS_TRAINING_DB="$(abspath "${MMSEQS_TRAINING_DB}")"
TRAINING_LIGANDS="$(abspath "${TRAINING_LIGANDS}")"
TRAINING_HOLO="$(abspath "${TRAINING_HOLO}")"
COLLECTED_RUNS_MANIFEST="$(abspath "${COLLECTED_RUNS_MANIFEST}")"
SKIN_KNOWN_CASES="$(abspath "${SKIN_KNOWN_CASES}")"
SKIN_KNOWN_RANKINGS="$(abspath "${SKIN_KNOWN_RANKINGS}")"
SKIN_KNOWN_RUN_LEDGER="$(abspath "${SKIN_KNOWN_RUN_LEDGER}")"
SOTA_BASELINES="$(abspath "${SOTA_BASELINES}")"
SOTA_ABLATIONS="$(abspath "${SOTA_ABLATIONS}")"
WORKFLOW_CONFIG="$(abspath "${WORKFLOW_CONFIG}")"
ACTIVITY_BENCHMARK_MANIFEST="$(abspath "${ACTIVITY_BENCHMARK_MANIFEST}")"
ACTIVITY_RETRIEVAL_INDEX_MANIFEST="$(abspath "${ACTIVITY_RETRIEVAL_INDEX_MANIFEST}")"
ACTIVITY_RECOVERY_PANELS_MANIFEST="$(abspath "${ACTIVITY_RECOVERY_PANELS_MANIFEST}")"
ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST="$(abspath "${ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST}")"
ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST="$(abspath "${ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST}")"
ACTIVITY_RCSB_CONTACT_POCKET_LEAKAGE_MANIFEST="$(abspath "${ACTIVITY_RCSB_CONTACT_POCKET_LEAKAGE_MANIFEST}")"
ACTIVITY_RETRIEVAL_GATE="$(abspath "${ACTIVITY_RETRIEVAL_GATE}")"

mkdir -p "${OUT}" "${LOG}"

require_input() {
    local path="$1"
    local label="$2"
    if [ -e "${path}" ]; then
        return 0
    fi
    if [ "${ALLOW_PARTIAL}" = "1" ]; then
        echo "[eval][WARN] missing ${label}; skipping diagnostic step: ${path}" >&2
        return 1
    fi
    echo "[eval][FATAL] missing ${label}: ${path}" >&2
    echo "[eval][FATAL] Set EVAL_ALLOW_PARTIAL=1 only for explicit partial diagnostics." >&2
    exit 1
}

remove_output() {
    local path="$1"
    if [ -f "${path}" ]; then
        rm -f "${path}"
    fi
}

validate_activity_retrieval_gate() {
    "${PYTHON_BIN}" - \
        "${ACTIVITY_BENCHMARK_MANIFEST}" \
        "${ACTIVITY_RETRIEVAL_INDEX_MANIFEST}" \
        "${ACTIVITY_RECOVERY_PANELS_MANIFEST}" \
        "${ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST}" \
        "${ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST}" \
        "${ACTIVITY_RCSB_CONTACT_POCKET_LEAKAGE_MANIFEST}" \
        "${SCRIPT_ROOT}" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMAS = {
    "benchmark": "activity_benchmark.v1",
    "index": "skinscout.activity-retrieval-index.v4",
    "panels": "skinscout.activity-recovery-panels.v5",
    "selection": "skinscout.activity-retrieval-selection.v1",
    "final": "skinscout.activity-retrieval-evaluation.v1",
    "rcsb_leakage": "skinscout.rcsb-contact-pocket-leakage-audit.v1",
    "rcsb_fragments": "skinscout.rcsb-contact-pocket-fragments.v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str, label: str) -> dict:
    manifest = Path(path)
    if not manifest.exists() or manifest.stat().st_size == 0:
        raise SystemExit(f"missing activity retrieval {label} manifest: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid activity retrieval {label} manifest JSON: {manifest}") from exc
    expected = SCHEMAS[label]
    if payload.get("schema_version") != expected:
        raise SystemExit(
            f"activity retrieval {label} manifest schema_version must be {expected}: {manifest}"
        )
    return payload


def resolve_record_path(record: dict, manifest_path: Path) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise SystemExit(f"manifest output record missing path near {manifest_path}")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(f"pyarrow is required to validate parquet row counts: {path}") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def table_rows(path: Path) -> int:
    if path.suffix == ".parquet":
        return parquet_rows(path)
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return max(0, sum(1 for _ in csv.reader(handle)) - 1)
    return -1


def require_record(payload: dict, section: str, key: str, *, label: str) -> dict:
    mapping = payload.get(section)
    if not isinstance(mapping, dict):
        raise SystemExit(f"activity retrieval {label} manifest missing {section}")
    record = mapping.get(key)
    if not isinstance(record, dict):
        raise SystemExit(f"activity retrieval {label} manifest missing {section}.{key}")
    return record


def validate_outputs(payload: dict, manifest_path: Path, *, label: str) -> None:
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise SystemExit(f"activity retrieval {label} manifest missing outputs")
    for name, record in outputs.items():
        if name == "manifest.json":
            continue
        if not isinstance(record, dict):
            raise SystemExit(f"activity retrieval {label} output {name!r} must be an object")
        path = resolve_record_path(record, manifest_path)
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"activity retrieval {label} output is missing or empty: {path}")
        expected_sha = record.get("sha256")
        if isinstance(expected_sha, str) and expected_sha and sha256(path) != expected_sha:
            raise SystemExit(f"activity retrieval {label} output sha256 mismatch: {path}")
        if "rows" in record:
            actual_rows = table_rows(path)
            if actual_rows >= 0 and int(record["rows"]) != actual_rows:
                raise SystemExit(f"activity retrieval {label} output row count mismatch: {path}")


def require_artifact_record(record: dict, *, label: str, rows: bool = False) -> Path:
    path = resolve_record_path(record, Path.cwd() / "manifest.json")
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"activity retrieval {label} artifact is missing or empty: {path}")
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise SystemExit(f"activity retrieval {label} artifact sha256 binding is invalid")
    if sha256(path) != expected_sha:
        raise SystemExit(f"activity retrieval {label} artifact sha256 mismatch: {path}")
    if rows:
        expected_rows = record.get("rows")
        if not isinstance(expected_rows, int) or isinstance(expected_rows, bool) or expected_rows < 0:
            raise SystemExit(f"activity retrieval {label} artifact row binding is invalid")
        actual_rows = table_rows(path)
        if actual_rows >= 0 and actual_rows != expected_rows:
            raise SystemExit(f"activity retrieval {label} artifact row count mismatch: {path}")
    return path.resolve()


def require_count(value: object, *, label: str, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(f"activity retrieval {label} count must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise SystemExit(f"activity retrieval {label} count exceeds denominator")
    return value


def require_rate(value: object, numerator: int, denominator: int, *, label: str) -> None:
    if denominator == 0:
        if value is not None:
            raise SystemExit(f"activity retrieval {label} rate must be null for zero denominator")
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(f"activity retrieval {label} rate must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
        raise SystemExit(f"activity retrieval {label} rate must be finite in [0,1]")
    if not math.isclose(float(value), numerator / denominator, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"activity retrieval {label} rate does not match count/total")


def validate_rcsb_contact_pocket_leakage(payload: dict, manifest_path: Path, benchmark_path: Path) -> None:
    contract = payload.get("contract")
    required_contract = {
        "evaluation_only": True,
        "never_training": True,
        "never_calibration": True,
        "never_model_selection": True,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) is not expected
        for key, expected in required_contract.items()
    ):
        raise SystemExit(
            "activity retrieval RCSB contact-pocket leakage contract must be "
            "evaluation-only and never training/calibration/model-selection"
        )

    benchmark_record = require_record(
        payload, "inputs", "activity_benchmark_manifest", label="RCSB contact-pocket leakage"
    )
    if Path(benchmark_record.get("path", "")).resolve() != benchmark_path.resolve():
        raise SystemExit("activity retrieval RCSB contact-pocket leakage benchmark path binding is stale")
    if benchmark_record.get("sha256") != sha256(benchmark_path):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage benchmark sha256 binding is stale")
    if benchmark_record.get("schema_version") != SCHEMAS["benchmark"]:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage benchmark schema binding is invalid")

    fragments = require_record(
        payload, "inputs", "rcsb_contact_fragments", label="RCSB contact-pocket leakage"
    )
    fragment_manifest = fragments.get("manifest")
    if not isinstance(fragment_manifest, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment manifest binding is missing")
    fragment_manifest_path = require_artifact_record(
        fragment_manifest, label="RCSB contact-pocket leakage fragment manifest"
    )
    fragment_payload = load(str(fragment_manifest_path), "rcsb_fragments")
    if fragment_manifest.get("schema_version") != SCHEMAS["rcsb_fragments"]:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment manifest schema binding is invalid")
    index_record = fragments.get("index")
    if not isinstance(index_record, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment index binding is missing")
    index_path = require_artifact_record(
        index_record, label="RCSB contact-pocket leakage fragment index", rows=True
    )
    fragment_artifacts = fragment_payload.get("artifacts")
    if not isinstance(fragment_artifacts, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment manifest artifacts are missing")
    manifest_index = fragment_artifacts.get("index_csv") or fragment_artifacts.get("index_parquet")
    if not isinstance(manifest_index, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment manifest index binding is missing")
    if (
        Path(manifest_index.get("path", "")).resolve() != index_path
        or manifest_index.get("sha256") != index_record.get("sha256")
        or manifest_index.get("rows") != index_record.get("rows")
    ):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment index binding is stale")
    exclusions_record = fragments.get("exclusions")
    if not isinstance(exclusions_record, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment exclusions binding is missing")
    exclusions_path = require_artifact_record(
        exclusions_record, label="RCSB contact-pocket leakage fragment exclusions", rows=True
    )
    manifest_exclusions = fragment_artifacts.get("exclusions_csv")
    if not isinstance(manifest_exclusions, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment manifest exclusions binding is missing")
    if (
        Path(manifest_exclusions.get("path", "")).resolve() != exclusions_path
        or manifest_exclusions.get("sha256") != exclusions_record.get("sha256")
        or manifest_exclusions.get("rows") != exclusions_record.get("rows")
    ):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage fragment exclusions binding is stale")

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage manifest missing outputs")
    detailed_record = outputs.get("detailed_csv")
    if not isinstance(detailed_record, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage missing detailed CSV output")
    detailed_path = require_artifact_record(
        detailed_record, label="RCSB contact-pocket leakage detailed CSV", rows=True
    )
    with detailed_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required_columns = {
        "ligand_key",
        "fragment_extracted",
        "foldseek_covered",
        "leakage_tm40",
        "leakage_tm50",
        "leakage_tm60",
    }
    if not required_columns.issubset(set(reader.fieldnames or [])):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage detailed CSV missing required columns")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage manifest missing summary")
    pairs = summary.get("pairs")
    ligands = summary.get("ligand_queries")
    if not isinstance(pairs, dict) or not isinstance(ligands, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage summary totals are missing")
    total = len(rows)
    if require_count(pairs.get("total"), label="RCSB contact-pocket leakage pair total") != total:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage pair total does not match CSV rows")
    extracted = sum(1 for row in rows if row["fragment_extracted"] == "true")
    excluded = sum(1 for row in rows if row["fragment_extracted"] == "false")
    covered = sum(1 for row in rows if row["foldseek_covered"] == "true")
    if require_count(pairs.get("fragment_extracted"), label="RCSB contact-pocket leakage extracted pairs", maximum=total) != extracted:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage extracted pair total mismatch")
    if require_count(pairs.get("fragment_extraction_excluded"), label="RCSB contact-pocket leakage excluded pairs", maximum=total) != excluded:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage excluded pair total mismatch")
    if extracted + excluded != total:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage extracted/excluded totals do not cover CSV")
    if require_count(pairs.get("covered"), label="RCSB contact-pocket leakage covered pairs", maximum=total) != covered:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage covered pair total mismatch")
    if require_count(pairs.get("uncovered"), label="RCSB contact-pocket leakage uncovered pairs", maximum=total) != total - covered:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage uncovered pair total mismatch")
    require_rate(pairs.get("coverage_rate"), covered, total, label="RCSB contact-pocket leakage coverage")

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage thresholds are missing")
    if [float(value) for value in thresholds.get("max_directional_tm", [])] != [0.4, 0.5, 0.6]:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage thresholds must be 0.4/0.5/0.6")
    expected_columns = {"0.4": "leakage_tm40", "0.5": "leakage_tm50", "0.6": "leakage_tm60"}
    if thresholds.get("columns") != expected_columns:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage threshold columns are invalid")

    pair_thresholds = pairs.get("leakage_by_max_directional_tm_threshold")
    ligand_thresholds = ligands.get("leakage_by_max_directional_tm_threshold")
    if not isinstance(pair_thresholds, dict) or not isinstance(ligand_thresholds, dict):
        raise SystemExit("activity retrieval RCSB contact-pocket leakage threshold summaries are missing")
    by_ligand: dict[str, list[dict]] = {}
    for row in rows:
        by_ligand.setdefault(row["ligand_key"], []).append(row)
    ligand_total = len(by_ligand)
    ligand_covered = sum(
        1 for grouped in by_ligand.values()
        if any(row["foldseek_covered"] == "true" for row in grouped)
    )
    if require_count(ligands.get("total"), label="RCSB contact-pocket leakage ligand total") != ligand_total:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage ligand total mismatch")
    if require_count(ligands.get("covered"), label="RCSB contact-pocket leakage covered ligands", maximum=ligand_total) != ligand_covered:
        raise SystemExit("activity retrieval RCSB contact-pocket leakage covered ligand total mismatch")

    for threshold, column in expected_columns.items():
        pair_record = pair_thresholds.get(threshold)
        ligand_record = ligand_thresholds.get(threshold)
        if not isinstance(pair_record, dict) or not isinstance(ligand_record, dict):
            raise SystemExit(f"activity retrieval RCSB contact-pocket leakage threshold {threshold} summary is missing")
        pair_count = sum(1 for row in rows if row[column] == "true")
        if require_count(pair_record.get("count"), label=f"RCSB contact-pocket leakage threshold {threshold} pairs", maximum=total) != pair_count:
            raise SystemExit(f"activity retrieval RCSB contact-pocket leakage threshold {threshold} pair count mismatch")
        require_rate(
            pair_record.get("rate_all_pairs"),
            pair_count,
            total,
            label=f"RCSB contact-pocket leakage threshold {threshold} all-pair",
        )
        require_rate(
            pair_record.get("rate_covered_pairs"),
            pair_count,
            covered,
            label=f"RCSB contact-pocket leakage threshold {threshold} covered-pair",
        )
        ligand_count = sum(
            1 for grouped in by_ligand.values()
            if any(row[column] == "true" for row in grouped)
        )
        if require_count(ligand_record.get("leaked_ligand_queries"), label=f"RCSB contact-pocket leakage threshold {threshold} ligands", maximum=ligand_total) != ligand_count:
            raise SystemExit(f"activity retrieval RCSB contact-pocket leakage threshold {threshold} ligand count mismatch")
        require_rate(
            ligand_record.get("leakage_rate"),
            ligand_count,
            ligand_total,
            label=f"RCSB contact-pocket leakage threshold {threshold} ligand",
        )


benchmark_path, index_path, panels_path, selection_path, final_path, rcsb_leakage_path = map(Path, sys.argv[1:7])
script_root = Path(sys.argv[7]).resolve()
benchmark = load(str(benchmark_path), "benchmark")
index = load(str(index_path), "index")
panels = load(str(panels_path), "panels")
selection = load(str(selection_path), "selection")
final = load(str(final_path), "final")
rcsb_leakage = load(str(rcsb_leakage_path), "rcsb_leakage")

validate_outputs(index, index_path, label="index")
validate_outputs(panels, panels_path, label="panels")
validate_outputs(selection, selection_path, label="dev-selection")
validate_outputs(final, final_path, label="final-evaluation")
validate_rcsb_contact_pocket_leakage(rcsb_leakage, rcsb_leakage_path, benchmark_path)

if selection.get("passes_dev_gate") is not True:
    raise SystemExit("activity retrieval dev-selection gate did not pass")
if final.get("passes_frozen_test_gate") is not True:
    raise SystemExit("activity retrieval final-evaluation gate did not pass")
if panels.get("passes_panel_adequacy_gate") is not True:
    raise SystemExit("activity retrieval dual-cold panel adequacy gate did not pass")
rcsb_record = panels.get("inputs", {}).get("rcsb_holo_direct_contact_panel")
if not isinstance(rcsb_record, dict):
    raise SystemExit("activity retrieval panels missing rcsb_holo_direct_contact_panel input")
rcsb_contract = rcsb_record.get("contract")
required_rcsb_contract = {
    "positive_only": True,
    "no_affinities_or_calibration": True,
    "never_training_or_model_selection": True,
}
if not isinstance(rcsb_contract, dict) or any(
    rcsb_contract.get(key) is not expected
    for key, expected in required_rcsb_contract.items()
):
    raise SystemExit(
        "activity retrieval RCSB panel contract must be positive-only, "
        "no-calibration, and never-model-selection"
    )
rcsb_path = Path(rcsb_record.get("path", ""))
if not rcsb_path.exists() or sha256(rcsb_path) != rcsb_record.get("sha256"):
    raise SystemExit("activity retrieval RCSB ranking binding is stale")
rcsb_manifest = rcsb_record.get("manifest")
if not isinstance(rcsb_manifest, dict):
    raise SystemExit("activity retrieval RCSB panel manifest binding is missing")
rcsb_manifest_path = Path(rcsb_manifest.get("path", ""))
if (
    rcsb_manifest.get("schema_version") != "skinscout.rcsb-holo-direct-contact-panel.v1"
    or not rcsb_manifest_path.exists()
    or sha256(rcsb_manifest_path) != rcsb_manifest.get("sha256")
):
    raise SystemExit("activity retrieval RCSB panel manifest binding is stale")

if panels["inputs"]["benchmark_manifest"]["sha256"] != sha256(benchmark_path):
    raise SystemExit("activity retrieval panels are stale relative to benchmark manifest")
if panels["inputs"]["retrieval_index_manifest"]["sha256"] != sha256(index_path):
    raise SystemExit("activity retrieval panels are stale relative to index manifest")
if selection["inputs"]["index_manifest"]["sha256"] != sha256(index_path):
    raise SystemExit("activity retrieval dev-selection is stale relative to index manifest")
if selection["inputs"]["panels_manifest"]["sha256"] != sha256(panels_path):
    raise SystemExit("activity retrieval dev-selection is stale relative to panels manifest")
if final["inputs"]["index_manifest"]["sha256"] != sha256(index_path):
    raise SystemExit("activity retrieval final-evaluation is stale relative to index manifest")
if final["inputs"]["panels_manifest"]["sha256"] != sha256(panels_path):
    raise SystemExit("activity retrieval final-evaluation is stale relative to panels manifest")

recipe_record = selection.get("outputs", {}).get("recipe.json", {})
final_recipe = final.get("inputs", {}).get("recipe", {})
if not isinstance(recipe_record, dict) or final_recipe.get("sha256") != recipe_record.get("sha256"):
    raise SystemExit("activity retrieval final-evaluation recipe does not match dev-selected recipe")

# Keep this shell entrypoint aligned with the production gate's stronger
# provenance, pocket-tree, detailed-row, and cross-manifest validation.
sys.path.insert(0, str(script_root / "scripts"))
from validate_activity_retrieval_gate import validate_manifest_contract

validate_manifest_contract(
    benchmark_manifest=benchmark_path,
    index_manifest=index_path,
    panels_manifest=panels_path,
    selection_manifest=selection_path,
    final_evaluation_manifest=final_path,
    pocket_leakage_manifest=rcsb_leakage_path,
)
PY
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/scripts/validate_activity_retrieval_gate.py" create \
        --benchmark-manifest "${ACTIVITY_BENCHMARK_MANIFEST}" \
        --index-manifest "${ACTIVITY_RETRIEVAL_INDEX_MANIFEST}" \
        --panels-manifest "${ACTIVITY_RECOVERY_PANELS_MANIFEST}" \
        --selection-manifest "${ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST}" \
        --final-evaluation-manifest "${ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST}" \
        --pocket-leakage-manifest "${ACTIVITY_RCSB_CONTACT_POCKET_LEAKAGE_MANIFEST}" \
        --out-gate "${ACTIVITY_RETRIEVAL_GATE}"
}

remove_output "${OUT}/iteration_manifest.json"
THRESHOLD_FAILURE_ARGS=()
if [ "${ALLOW_THRESHOLD_FAILURE}" = "1" ]; then
    THRESHOLD_FAILURE_ARGS+=(--allow-threshold-failure)
fi

# --- v2 leakage audit + cold-start ------------------------------------------
if require_input "${OUT}/eval_targets.csv" "leakage audit targets" \
   && require_input "${MMSEQS_TRAINING_DB}" "leakage sequence reference" \
   && require_input "${TRAINING_LIGANDS}" "leakage ligand reference" \
   && require_input "${TRAINING_HOLO}" "leakage pocket reference"; then
    LEAKAGE_ARGS=()
    if [ "${ALLOW_INCOMPLETE_LEAKAGE}" = "1" ]; then
        LEAKAGE_ARGS+=(--allow-incomplete)
    fi
    remove_output "${OUT}/leakage_audit.csv"
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/leakage_check.py" \
        --input-csv "${OUT}/eval_targets.csv" \
        --training-seq-db "${MMSEQS_TRAINING_DB}" \
        --training-ligands "${TRAINING_LIGANDS}" \
        --training-holo "${TRAINING_HOLO}" \
        --seq-id-threshold "${EVAL_SEQ_ID_THRESHOLD}" \
        --ligand-tanimoto-threshold "${EVAL_LIGAND_TANIMOTO_THRESHOLD}" \
        --pocket-sucos-threshold "${EVAL_POCKET_SUCOS_THRESHOLD}" \
        --out-csv "${OUT}/leakage_audit.csv" \
        "${LEAKAGE_ARGS[@]}" \
        > "${LOG}/leakage.log" 2>&1
fi

# --- Cold-start (3 modes) ---------------------------------------------------
for MODE in comprehensive fast dti_only; do
    R="${RANKINGS}/cold_start__${MODE}.csv"
    GT="${OUT}/cold_start_truth.csv"
    if require_input "${GT}" "cold-start ground truth" \
       && require_input "${R}" "cold-start ${MODE} ranking"; then
        remove_output "${OUT}/cold_start_${MODE}.csv"
        "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/cold_start_eval.py" \
            --mode "${MODE}" \
            --ranking-csv "${R}" \
            --ground-truth-csv "${GT}" \
            --chembl-dir "${CHEMBL_DIR}" \
            --min-recall-at-50 "${EVAL_COLD_START_MIN_RECALL_AT_50}" \
            "${THRESHOLD_FAILURE_ARGS[@]}" \
            --out-csv "${OUT}/cold_start_${MODE}.csv" \
            > "${LOG}/cold_start_${MODE}.log" 2>&1
    fi
done

# --- v2 DTI-vs-Docking disagreement -----------------------------------------
if require_input "${OUT}/disagreement_analysis.json" "disagreement analysis JSON" \
   && require_input "${TARGET_CLASSES}" "target class reference"; then
    remove_output "${OUT}/disagreement_per_class.csv"
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/disagreement_eval.py" \
        --disagreement-json "${OUT}/disagreement_analysis.json" \
        --target-classes "${TARGET_CLASSES}" \
        --out-csv "${OUT}/disagreement_per_class.csv" \
        > "${LOG}/disagreement.log" 2>&1
fi

if [ "${ALLOW_PARTIAL}" != "1" ]; then
    require_input "${COLLECTED_RUNS_MANIFEST}" "collected run provenance manifest"
fi
case "${ACTIVITY_RETRIEVAL_REQUIRED}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        if require_input "${ACTIVITY_BENCHMARK_MANIFEST}" "activity retrieval benchmark manifest" \
           && require_input "${ACTIVITY_RETRIEVAL_INDEX_MANIFEST}" "activity retrieval index manifest" \
           && require_input "${ACTIVITY_RECOVERY_PANELS_MANIFEST}" "activity retrieval recovery panels manifest" \
           && require_input "${ACTIVITY_RETRIEVAL_DEV_SELECTION_MANIFEST}" "activity retrieval dev-selection manifest" \
           && require_input "${ACTIVITY_RETRIEVAL_FINAL_EVAL_MANIFEST}" "activity retrieval final-evaluation manifest" \
           && require_input "${ACTIVITY_RCSB_CONTACT_POCKET_LEAKAGE_MANIFEST}" "activity retrieval RCSB contact-pocket leakage manifest"; then
            validate_activity_retrieval_gate > "${LOG}/activity_retrieval_gate.log" 2>&1
        fi
        ;;
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
        echo "[eval][WARN] activity retrieval final gate disabled by configuration" >&2
        ;;
    *)
        echo "[eval][FATAL] ACTIVITY_RETRIEVAL_REQUIRED must be boolean-like, got: ${ACTIVITY_RETRIEVAL_REQUIRED}" >&2
        exit 1
        ;;
esac
if [ "${ALLOW_PARTIAL}" != "1" ] && [ -e "${SKIN_KNOWN_RANKINGS}" ]; then
    require_input "${SKIN_KNOWN_RUN_LEDGER}" "known skin compound run ledger"
    require_input "${SOTA_BASELINES}" "SOTA baseline fairness table"
    require_input "${SOTA_ABLATIONS}" "SOTA ablation freeze table"
fi

# --- SOTA known skin compound target recovery -------------------------------
if require_input "${SKIN_KNOWN_CASES}" "known skin compound target panel" \
   && require_input "${SKIN_KNOWN_RANKINGS}" "known skin compound target rankings"; then
    remove_output "${OUT}/skin_known_target_recovery.csv"
    remove_output "${OUT}/skin_known_target_recovery_targets.csv"
    remove_output "${OUT}/skin_known_target_recovery_summary.json"
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/skin_known_target_recovery_eval.py" \
        --cases-csv "${SKIN_KNOWN_CASES}" \
        --rankings-dir "${SKIN_KNOWN_RANKINGS}" \
        --out-csv "${OUT}/skin_known_target_recovery.csv" \
        --out-target-csv "${OUT}/skin_known_target_recovery_targets.csv" \
        --out-summary-json "${OUT}/skin_known_target_recovery_summary.json" \
        --context-profile "${EVAL_SKIN_KNOWN_CONTEXT_PROFILE}" \
        --min-case-top10 "${EVAL_SKIN_KNOWN_MIN_CASE_TOP10}" \
        --min-target-top10 "${EVAL_SKIN_KNOWN_MIN_TARGET_TOP10}" \
        --min-target-top30 "${EVAL_SKIN_KNOWN_MIN_TARGET_TOP30}" \
        "${THRESHOLD_FAILURE_ARGS[@]}" \
        > "${LOG}/skin_known_target_recovery.log" 2>&1
fi

# --- v3 Cosmetic retrospective ----------------------------------------------
if require_input "${RANKINGS}/cosmetic_retro" "cosmetic retrospective rankings directory"; then
    remove_output "${OUT}/cosmetic_retrospective.csv"
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/cosmetic_retrospective_eval.py" \
        --rankings-dir "${RANKINGS}/cosmetic_retro" \
        --min-mean-top10 "${EVAL_COSMETIC_RETRO_MIN_MEAN_TOP10}" \
        "${THRESHOLD_FAILURE_ARGS[@]}" \
        --out-csv "${OUT}/cosmetic_retrospective.csv" \
        > "${LOG}/cosmetic_retro.log" 2>&1
fi

# --- v3 Skin-efficacy KG recovery -------------------------------------------
if require_input "${RANKINGS}/cosmetic_retro" "skin-efficacy rankings directory"; then
    remove_output "${OUT}/skin_efficacy_recovery.csv"
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/skin_efficacy_recovery_eval.py" \
        --ranked-dir "${RANKINGS}/cosmetic_retro" \
        --min-mean-precision "${EVAL_SKIN_EFFICACY_MIN_MEAN_PRECISION}" \
        --min-mean-recall "${EVAL_SKIN_EFFICACY_MIN_MEAN_RECALL}" \
        "${THRESHOLD_FAILURE_ARGS[@]}" \
        --out-csv "${OUT}/skin_efficacy_recovery.csv" \
        > "${LOG}/skin_efficacy.log" 2>&1
fi

# --- v3 Analog quality ------------------------------------------------------
if require_input "${OUT}/generated_analogs.csv" "generated analogs CSV" \
   && require_input "${OUT}/other_cosing.csv" "other CosIng CSV" \
   && require_input "${CHEMBL_FP_PARQUET}" "ChEMBL fingerprint parquet"; then
    remove_output "${OUT}/analog_quality.csv"
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/analog_quality_eval.py" \
        --analogs-csv "${OUT}/generated_analogs.csv" \
        --other-cosing-csv "${OUT}/other_cosing.csv" \
        --chembl-fp-parquet "${CHEMBL_FP_PARQUET}" \
        --min-recovery "${EVAL_ANALOG_MIN_RECOVERY}" \
        --min-novelty "${EVAL_ANALOG_MIN_NOVELTY}" \
        --min-mean-ra-score "${EVAL_ANALOG_MIN_MEAN_RA_SCORE}" \
        "${THRESHOLD_FAILURE_ARGS[@]}" \
        --out-csv "${OUT}/analog_quality.csv" \
        > "${LOG}/analog_quality.log" 2>&1
fi

# --- v3 Pharmacophore conservation ------------------------------------------
if require_input "${OUT}/parent.sdf" "pharmacophore parent SDF" \
   && require_input "${OUT}/consensus.json" "pharmacophore consensus JSON" \
   && require_input "${OUT}/interaction_anchor_map.json" "pharmacophore interaction-anchor map" \
   && require_input "${OUT}/generated_analogs.csv" "generated analogs CSV"; then
    remove_output "${OUT}/pharmacophore_conservation.csv"
    remove_output "${OUT}/pharmacophore_conservation_detail.csv"
    "${PYTHON_BIN}" "${SCRIPT_ROOT}/eval/pharmacophore_conservation_eval.py" \
        --parent-sdf "${OUT}/parent.sdf" \
        --consensus-json "${OUT}/consensus.json" \
        --interaction-anchor-map "${OUT}/interaction_anchor_map.json" \
        --analogs-csv "${OUT}/generated_analogs.csv" \
        --min-preserved-fraction "${EVAL_PHARMACOPHORE_MIN_PRESERVED_FRACTION}" \
        "${THRESHOLD_FAILURE_ARGS[@]}" \
        --out-csv "${OUT}/pharmacophore_conservation.csv" \
        --out-detail-csv "${OUT}/pharmacophore_conservation_detail.csv" \
        > "${LOG}/pharmacophore_conservation.log" 2>&1
fi

ITERATION_ARGS=(
    --eval-dir "${OUT}"
    --rankings-dir "${RANKINGS}"
    --out-manifest "${OUT}/iteration_manifest.json"
    --workflow-config "${WORKFLOW_CONFIG}"
    --activity-retrieval-gate "${ACTIVITY_RETRIEVAL_GATE}"
    --target-classes "${TARGET_CLASSES}"
    --chembl-dir "${CHEMBL_DIR}"
    --chembl-fp-parquet "${CHEMBL_FP_PARQUET}"
    --training-seq-db "${MMSEQS_TRAINING_DB}"
    --training-ligands "${TRAINING_LIGANDS}"
    --training-holo "${TRAINING_HOLO}"
    --seq-id-threshold "${EVAL_SEQ_ID_THRESHOLD}"
    --ligand-tanimoto-threshold "${EVAL_LIGAND_TANIMOTO_THRESHOLD}"
    --pocket-sucos-threshold "${EVAL_POCKET_SUCOS_THRESHOLD}"
    --cold-start-min-recall-at-50 "${EVAL_COLD_START_MIN_RECALL_AT_50}"
    --cosmetic-retro-min-mean-top10 "${EVAL_COSMETIC_RETRO_MIN_MEAN_TOP10}"
    --skin-known-cases-csv "${SKIN_KNOWN_CASES}"
    --skin-known-rankings-dir "${SKIN_KNOWN_RANKINGS}"
    --skin-known-run-ledger-csv "${SKIN_KNOWN_RUN_LEDGER}"
    --sota-baselines-csv "${SOTA_BASELINES}"
    --sota-ablations-csv "${SOTA_ABLATIONS}"
    --skin-known-min-case-top10 "${EVAL_SKIN_KNOWN_MIN_CASE_TOP10}"
    --skin-known-min-target-top10 "${EVAL_SKIN_KNOWN_MIN_TARGET_TOP10}"
    --skin-known-min-target-top30 "${EVAL_SKIN_KNOWN_MIN_TARGET_TOP30}"
    --skin-known-context-profile "${EVAL_SKIN_KNOWN_CONTEXT_PROFILE}"
    --skin-efficacy-min-mean-precision "${EVAL_SKIN_EFFICACY_MIN_MEAN_PRECISION}"
    --skin-efficacy-min-mean-recall "${EVAL_SKIN_EFFICACY_MIN_MEAN_RECALL}"
    --analog-min-recovery "${EVAL_ANALOG_MIN_RECOVERY}"
    --analog-min-novelty "${EVAL_ANALOG_MIN_NOVELTY}"
    --analog-min-mean-ra-score "${EVAL_ANALOG_MIN_MEAN_RA_SCORE}"
    --pharmacophore-min-preserved-fraction "${EVAL_PHARMACOPHORE_MIN_PRESERVED_FRACTION}"
    --collected-runs-manifest "${COLLECTED_RUNS_MANIFEST}"
)
if [ "${ALLOW_PARTIAL}" = "1" ]; then
    ITERATION_ARGS+=(
        --allow-empty-iteration
        --allow-incomplete-eval-inputs
        --allow-incomplete-leakage
        --allow-missing-input-runs
    )
elif [ "${ALLOW_INCOMPLETE_LEAKAGE}" = "1" ]; then
    ITERATION_ARGS+=(--allow-incomplete-leakage)
fi
if [ "${ALLOW_THRESHOLD_FAILURE}" = "1" ]; then
    ITERATION_ARGS+=(--allow-threshold-failure)
fi
(
    cd "${SCRIPT_ROOT}"
    "${PYTHON_BIN}" eval/run_iteration.py "${ITERATION_ARGS[@]}"
) > "${LOG}/iteration_manifest.log" 2>&1

echo "[eval] all available benchmarks run — manifest in ${OUT}/iteration_manifest.json"
ls -la "${OUT}"
