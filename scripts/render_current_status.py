#!/usr/bin/env python3
"""Render the current operational status block in `docs/IMPLEMENTATION_STATUS.md`.

The block used to be hand-written, so it kept claiming a promoted
`union_any_consensus`, operational PASS, and Top30 35/46 after the active gate
had moved to `union_discount_light` and a diagnostic runtime binding, and the
hash-bound summary said 34/46. The numbers live only in artifacts, so the
document is generated from them: the active gate names the bound recipe and
summary; the recipe and summary files are read back and their fractions turned
into integer pair counts. `--check` fails when the document drifts.

R02 follow-up: reading the stored decisions was not enough. When the runtime
index manifest was regenerated without rebinding the gate, `check-operational`
failed while this generator still printed the stored PASS. The block now runs
the official checker (`validate_activity_retrieval_gate.check_operational_gate`)
and, when it fails, marks the current verification state as failed/unverified
while keeping the stored evaluation numbers as explicitly historical.

Run:
    python scripts/render_current_status.py --write   # regenerate the block
    python scripts/render_current_status.py --check   # verify the document
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from validate_activity_retrieval_gate import check_operational_gate

ROOT = Path(__file__).resolve().parents[1]
STATUS_DOC = ROOT / "docs" / "IMPLEMENTATION_STATUS.md"
GATE = ROOT / "data" / "manifests" / "activity_retrieval_operational_gate.flag"
FINAL_GATE = ROOT / "data" / "manifests" / "activity_retrieval_final_gate.flag"
CONFIG = ROOT / "workflow" / "config.yaml"
BEGIN_MARKER = "<!-- BEGIN GENERATED: current-operational-state -->"
END_MARKER = "<!-- END GENERATED: current-operational-state -->"
KNOWN_PANEL = "full_known_skin_panel_15_cases"
TEMPORAL_PANEL = "temporal_test_2025"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _hits(metrics: dict, metric: str) -> tuple[int, int]:
    pairs = int(metrics["n_truth_pairs"])
    value = metrics.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(f"summary metric {metric} is not numeric")
    return int(round(float(value) * pairs)), pairs


def _operational_validation(gate_path: Path, gate: dict) -> dict:
    """Run the checker the run path runs, and keep the binding hashes for display."""
    try:
        check_operational_gate(gate_path)
    except SystemExit as exc:
        status = "fail"
        detail = str(exc) or "check-operational exited non-zero"
    except Exception as exc:  # missing or malformed bound artifacts
        status = "fail"
        detail = f"{type(exc).__name__}: {exc}"
    else:
        status = "pass"
        detail = None
    record = gate.get("runtime_index_manifest")
    bound = record.get("sha256") if isinstance(record, dict) else None
    actual = None
    if isinstance(record, dict) and record.get("path"):
        manifest_path = Path(str(record["path"]))
        if manifest_path.is_file():
            actual = _sha256(manifest_path)
    return {
        "status": status,
        "detail": detail,
        "runtime_index_bound_sha256": bound,
        "runtime_index_actual_sha256": actual,
    }


def current_state(root: Path = ROOT) -> dict:
    config = yaml.safe_load((root / CONFIG.relative_to(ROOT)).read_text(encoding="utf-8"))[
        "docking"
    ]
    if config.get("daina_recipe_scoring") is not True:
        raise SystemExit("the active config does not use recipe scoring")
    recipe_path = root / config["daina_recipe_path"]
    recipe = _load_json(recipe_path)
    gate_path = root / GATE.relative_to(ROOT)
    gate = _load_json(gate_path)
    final_gate_path = root / FINAL_GATE.relative_to(ROOT)
    final_gate = _load_json(final_gate_path) if final_gate_path.exists() else {}
    operational_validation = _operational_validation(gate_path, gate)
    index_dir = root / config["daina_recipe_index_dir"]
    summary_record = gate.get("final_summary")
    if not isinstance(summary_record, dict) or not summary_record.get("path"):
        raise SystemExit("the operational gate does not bind a final summary")
    summary_path = Path(str(summary_record["path"]))
    if _sha256(summary_path) != summary_record.get("sha256"):
        raise SystemExit("the bound final summary is stale")
    summary = _load_json(summary_path)
    ranking = summary["ranking"]

    def arm(panel: str, recipe_id: str) -> dict:
        metrics = ranking[panel][recipe_id]
        top10, pairs = _hits(metrics, "top10")
        top30, _ = _hits(metrics, "top30")
        return {
            "top10": top10,
            "top30": top30,
            "pairs": pairs,
            "mrr": float(metrics["mrr"]),
        }

    selected_id = recipe["selected_recipe"]["recipe_id"]
    baseline_id = recipe["baseline_recipe"]["recipe_id"]
    operational_id = str(
        (gate.get("evaluation_decision") or {}).get("operational_recipe_id")
        or selected_id
    )
    return {
        "recipe_id": selected_id,
        "operational_recipe_id": operational_id,
        "baseline_recipe_id": baseline_id,
        "recipe_path": _display(recipe_path),
        "recipe_sha256": _sha256(recipe_path),
        "index_dir": _display(index_dir),
        "gate_path": _display(gate_path),
        "gate_sha256": _sha256(gate_path),
        "runtime_decision": dict(gate.get("runtime_decision") or {}),
        "evaluation_decision": dict(gate.get("evaluation_decision") or {}),
        "operational_validation": operational_validation,
        "claim_gate_status": final_gate.get("status", "missing"),
        "runtime_claim_ready": gate.get("runtime_decision", {}).get("claim_ready"),
        "summary_path": _display(summary_path),
        "summary_sha256": str(summary_record.get("sha256")),
        "known_selected": arm(KNOWN_PANEL, selected_id),
        "known_baseline": arm(KNOWN_PANEL, baseline_id),
        "temporal_selected": arm(TEMPORAL_PANEL, selected_id),
        "temporal_baseline": arm(TEMPORAL_PANEL, baseline_id),
        "dual_cold_selected": {
            "mrr": float(ranking["dual_cold"][selected_id]["mrr"])
        },
        "dual_cold_baseline": {
            "mrr": float(ranking["dual_cold"][baseline_id]["mrr"])
        },
        "rerank_enabled": bool(config.get("fast_mode_rerank_enabled", False)),
    }


def render_block(state: dict) -> str:
    known_selected = state["known_selected"]
    known_baseline = state["known_baseline"]
    temporal = state["temporal_selected"]
    temporal_baseline = state["temporal_baseline"]
    runtime = state["runtime_decision"]
    evaluation = state["evaluation_decision"]
    validation = state["operational_validation"]
    verified = validation["status"] == "pass"
    history = "" if verified else "저장된(역사) "
    lines = [BEGIN_MARKER]
    if verified:
        lines.append(
            "- 현재 운영 검증: **통과** — `validate_activity_retrieval_gate.py "
            "check-operational`이 현재 runtime index·recipe·manifest 결속을 승인했다."
        )
    else:
        lines.append(
            "- 현재 운영 검증: **실패(미검증)** — `validate_activity_retrieval_gate.py "
            f"check-operational` 실패: `{validation['detail']}`. 아래 런타임·평가 수치는 "
            "게이트에 저장된 역사 기록이며 현재 효력이 검증되지 않았다."
        )
    if state["operational_recipe_id"] == state["recipe_id"]:
        lines.append(
            f"- 활성 recipe: `{state['recipe_id']}` — `{state['recipe_path']}` "
            f"(sha256 `{state['recipe_sha256']}`)"
        )
    else:
        lines.append(
            f"- 활성 recipe: 운영 `{state['operational_recipe_id']}` (게이트 decision), "
            f"dev-selected `{state['recipe_id']}` — `{state['recipe_path']}` "
            f"(sha256 `{state['recipe_sha256']}`); 실행 경로는 게이트의 operational "
            "recipe를 적용한다."
        )
    if verified:
        lines.append(
            f"- 런타임 검색 인덱스: `{state['index_dir']}` (운영 게이트가 sha256으로 결속)"
        )
    else:
        lines.append(
            f"- 런타임 검색 인덱스: `{state['index_dir']}` — 게이트 결속 sha256 "
            f"`{validation['runtime_index_bound_sha256']}` vs 현재 manifest "
            f"`{validation['runtime_index_actual_sha256']}` (불일치; 재바인딩 전까지 "
            "기본 MODE-FAST 표적 분석은 차단된다)"
        )
    lines.append(
        f"- operational gate `{state['gate_path']}` (sha256 `{state['gate_sha256']}`): "
        f"{history}runtime decision `{runtime.get('promotion_decision')}`, "
        f"`claim_ready={str(runtime.get('claim_ready')).lower()}` — 런타임 인덱스는 동결 "
        "평가 인덱스가 아니므로 동결 평가 수치를 운영 성능 주장으로 쓸 수 없다."
    )
    lines.append(
        f"- 같은 게이트의 {history}evaluation decision: `{evaluation.get('promotion_decision')}`, "
        f"operational recipe `{evaluation.get('operational_recipe_id')}`, "
        f"`claim_ready={str(evaluation.get('claim_ready')).lower()}`; claim gate "
        f"`status={state['claim_gate_status']}`."
        + ("" if verified else " 현재 유효성은 위 운영 검증 실패로 보류된다.")
    )
    lines.append(
        f"- 해시로 결속된 `{state['summary_path']}` (sha256 `{state['summary_sha256']}`)의 "
        f"{known_selected['pairs']}쌍 known panel{'' if verified else ' (역사 기록)'}: baseline "
        f"`{state['baseline_recipe_id']}` Top10 {known_baseline['top10']}/{known_baseline['pairs']} · "
        f"Top30 {known_baseline['top30']}/{known_baseline['pairs']}; selected "
        f"`{state['recipe_id']}` Top10 {known_selected['top10']}/{known_selected['pairs']} · "
        f"Top30 {known_selected['top30']}/{known_selected['pairs']}."
    )
    lines.append(
        f"- temporal 2025{'' if verified else ' (역사 기록)'}: baseline "
        f"Top10 {temporal_baseline['top10']}/{temporal_baseline['pairs']} · "
        f"Top30 {temporal_baseline['top30']}/{temporal_baseline['pairs']} · "
        f"MRR {temporal_baseline['mrr']:.5f}; selected "
        f"Top10 {temporal['top10']}/{temporal['pairs']} · "
        f"Top30 {temporal['top30']}/{temporal['pairs']} · MRR {temporal['mrr']:.5f}."
    )
    lines.append(
        f"- dual-cold: baseline MRR {state['dual_cold_baseline']['mrr']:.8f}, selected "
        f"MRR {state['dual_cold_selected']['mrr']:.8f}; 회수 주장 없음."
    )
    lines.append(
        "- 밴드 재정렬은 현재 기본값에서 "
        + ("켜져 있다." if state["rerank_enabled"] else "꺼져 있다.")
        + " (구조 주석은 11–50위 밴드를 그대로 사용)"
    )
    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def document_with_block(document: str, block: str) -> str:
    if BEGIN_MARKER not in document or END_MARKER not in document:
        raise SystemExit(
            f"{STATUS_DOC} has no generated block markers ({BEGIN_MARKER})"
        )
    before, rest = document.split(BEGIN_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    return before + block.rstrip("\n") + after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    state = current_state()
    block = render_block(state)
    document = STATUS_DOC.read_text(encoding="utf-8")
    updated = document_with_block(document, block)
    if args.write:
        STATUS_DOC.write_text(updated, encoding="utf-8")
        print(f"updated {STATUS_DOC}")
        return 0
    if updated != document:
        raise SystemExit(
            f"{STATUS_DOC} current-state block is stale; run "
            "python scripts/render_current_status.py --write"
        )
    print("current-state block is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
