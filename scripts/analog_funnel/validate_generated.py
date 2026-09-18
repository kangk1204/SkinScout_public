#!/usr/bin/env python3
"""B2 검증: BRICS 생성 후보를 안전성(ADMET) 컷 → 의도표적 쌍 도킹으로 검증한다."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENB = Path("/tmp/opencode/genb")
OUT = GENB / "dock_results.csv"
LOGDIR = GENB / "logs"
LOCK = GENB / ".b2.lock"
MM = os.environ.get("MICROMAMBA_BIN", "micromamba")
DOCK_ENV = "cosmax-autodock-gpu"
TOP_PER_TARGET = 10


sys.path.insert(0, str(ROOT / "scripts"))
import run_skinscout as rs  # noqa: E402
import safety_states  # noqa: E402

# 미커밋 HuSSPred AD 게이트가 unknown 상태에서 음성 투표를 버려 consensus가
# fail-closed로 죽는다(다른 세션 WIP). 명시적 degraded 모드로 계산한다.
DEGRADED_CONFIG = rs._snakemake_config_entries(rs.SAFETY_DEGRADED_CONFIG)


def first_line(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except Exception:
        return ""


def docking_box_dir() -> Path:
    """workflow config의 paths.docking_boxes로 박스 디렉터리를 해석한다."""
    import yaml

    config_path = ROOT / "workflow" / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"workflow config를 읽을 수 없습니다: {config_path}: {exc}") from exc
    try:
        configured = str(config["paths"]["docking_boxes"]).strip()
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"paths.docking_boxes가 없습니다: {config_path}") from exc
    if not configured:
        raise SystemExit(f"paths.docking_boxes가 비어 있습니다: {config_path}")
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


SAFE_DECISIONS = safety_states.SAFE_DECISIONS


def safety_state_for_run(run_dir: Path) -> tuple[str, dict]:
    """Read the run's decision AND its nested evidence gaps.

    The availability/applicability/decision separation lives in
    ``safety_states``: a degraded run stays docking-eligible as a diagnostic
    but can never be recorded as a safety PASS or a claimable result.
    """
    skin = first_line(run_dir / "02_admet/skin_sens_decision.txt")
    payload: dict = {}
    report = run_dir / "02_admet/admet_report.json"
    if report.is_file():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    skin_sens = safety_states.nested_skin_sens(payload)
    state = safety_states.safety_state(
        skin or skin_sens.get("decision", ""),
        missing_models=skin_sens.get("missing_models"),
        applicability_limited_models=skin_sens.get("applicability_limited_models"),
        degraded=skin_sens.get("degraded"),
    )
    return skin, state


def run(cmd: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def pair_dock(run_id: str, ligand: Path, uniprot: str, tag: str) -> tuple[str, str, str]:
    run_dir = ROOT / "results/runs" / run_id
    fast_dir = run_dir / "03_targets/mode_fast"
    pair_dir = run_dir / "03_targets/funnel_pair"
    daina = fast_dir / "daina_top256.csv"
    if not daina.is_file():
        return "", "", "fast_output_missing"
    rows = list(csv.DictReader(daina.open(encoding="utf-8-sig")))
    target_row = next((r for r in rows if r.get("target_id") == uniprot), None)
    note = ""
    if target_row is None:
        target_row = {**rows[0], "target_id": uniprot, "daina_rank": 1}
        note = "synthesized_rank_outside_daina_top256"
    target_row = {**target_row, "daina_rank": 1}
    pair_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("autodock_pair.tsv", "gnina_pair.tsv",
                  "autodock_pair_status.json", "gnina_pair_status.json"):
        (pair_dir / stale).unlink(missing_ok=True)
    with (pair_dir / "selected.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerow(target_row)
    steps = [
        (["python", "scripts/stage3_autogrid_maps.py",
          "--selected-csv", str(pair_dir / "selected.csv"), "--ligand-pdbqt", str(ligand),
          "--receptor-dir", "data/human_pdbqt", "--box-dir", str(docking_box_dir()),
          "--no-pocket-list", "data/no_pocket_targets.list", "--spacing", "0.375",
          "--cache-dir", str(Path.home() / ".cache/skinscout/autogrid"),
          "--dock-rank-from", "0", "--dock-rank-to", "0",
          "--out-map-dir", str(pair_dir / "autogrid_maps"),
          "--out-manifest", str(pair_dir / "autogrid_map_manifest.json")],
         f"{tag}_autogrid.log"),
        (["python", "scripts/stage3_autodock_run.py",
          "--ligand-pdbqt", str(ligand), "--receptor-dir", "data/human_pdbqt",
          "--box-dir", str(docking_box_dir()),
          "--map-manifest", str(pair_dir / "autogrid_map_manifest.json"),
          "--allow-partial-structure", "--engine", "autodock_gpu",
          "--nrun", "4", "--search-volume-advisory", "27000",
          "--high-volume-min-nrun", "32", "--ls-method", "ad",
          "--max-receptors", "256", "--vina-cpu", "16",
          "--out-scores", str(pair_dir / "autodock_pair.tsv"),
          "--out-pose-dir", str(pair_dir / "autodock_pair_poses"),
          "--out-status-manifest", str(pair_dir / "autodock_pair_status.json")],
         f"{tag}_autodock.log"),
        (["python", "scripts/stage3_gnina_rescore.py",
          "--top-csv", str(pair_dir / "autodock_pair.tsv"),
          "--pose-manifest", str(pair_dir / "autodock_pair_poses/pose_manifest.json"),
          "--pose-dir", str(pair_dir / "autodock_pair_poses"),
          "--clean-dir", "data/human_clean", "--use-gpu", "--allow-partial-structure",
          "--out-scores", str(pair_dir / "gnina_pair.tsv"),
          "--out-status-manifest", str(pair_dir / "gnina_pair_status.json")],
         f"{tag}_gnina.log"),
    ]
    for script_cmd, log_name in steps:
        if run([MM, "run", "-n", DOCK_ENV, *script_cmd], LOGDIR / log_name) != 0:
            return "", "", f"step_failed:{log_name}"
    autodock = gnina = ""
    scores = pair_dir / "autodock_pair.tsv"
    if scores.is_file():
        for r in csv.DictReader(scores.open(encoding="utf-8-sig"), delimiter="\t"):
            if r.get("target_id") == uniprot:
                autodock = r.get("vina_score", "")
    gnina_scores = pair_dir / "gnina_pair.tsv"
    if gnina_scores.is_file():
        for r in csv.DictReader(gnina_scores.open(encoding="utf-8-sig"), delimiter="\t"):
            if r.get("target_id") == uniprot:
                gnina = r.get("cnn_affinity", "")
    return autodock, gnina, note


def main() -> None:
    if LOCK.exists():
        raise SystemExit(f"lock exists: {LOCK}")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        LOGDIR.mkdir(parents=True, exist_ok=True)
        candidates: list[dict] = []
        for path in sorted(GENB.glob("*_candidates.csv")):
            rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
            for row in rows[:TOP_PER_TARGET]:
                if not row.get("gene") or not row.get("uniprot"):
                    raise SystemExit(f"{path}: gene/uniprot 열이 필요합니다")
                candidates.append(dict(row))
        print(f"[b2] 후보 {len(candidates)}", flush=True)
        with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gene", "uniprot", "smiles", "qed", "sa", "mw",
                             "feature_recall", "run_id", "skin_sens", "cosmetic",
                             "autodock_kcal_mol", "gnina_cnn_affinity",
                             "review_required", "note",
                             "safety_availability", "safety_applicability",
                             "claim_eligible", "missing_models"])
            for index, item in enumerate(candidates, 1):
                smiles = item["smiles"].strip()
                run_id = rs._auto_run_id_from_smiles(smiles)
                tag = f"{index:02d}_{item['gene'].replace(' ', '')}"
                cmd = [
                    MM, "run", "-n", "cosmax-base", "snakemake",
                    "-s", "workflow/Snakefile", "--cores", "16",
                    "--use-conda", "--conda-frontend", "conda", "--rerun-triggers=mtime",
                    "skin_sens_consensus", "cosmetic_drug_decision", "--resources", "gpu=1",
                    "--config", f"run_id={run_id}", "mode=comprehensive",
                    f"compound_smiles={smiles}", "run_dti_sanity=true",
                    "evidence_mode=evidence", "target_metadata=data/hpa/proteinatlas.tsv",
                    *DEGRADED_CONFIG,
                ]
                rc = run(cmd, LOGDIR / f"{tag}_safety.log")
                run_dir = ROOT / "results/runs" / run_id
                skin, state = safety_state_for_run(run_dir)
                cosmetic = first_line(run_dir / "02b_cosmetic_drug/cosmetic_drug_decision.txt")
                autodock = gnina = ""
                notes: list[str] = [] if rc == 0 else ["safety_failed"]
                if rc == 0 and not state["docking_eligible"]:
                    # 빈 값/UNKNOWN/HALT는 도킹 미완료로 중단한다(fail-closed).
                    notes.append(f"safety_ineligible:{state['reason']}")
                elif rc == 0 and not state["claimable"]:
                    # 결손 증거/FLAG_HIGH는 진단 도킹은 하되 주장 가능한
                    # 안전성 PASS로 승격하지 않는다. 사유를 그대로 남긴다.
                    notes.append(state["reason"])
                if rc == 0 and state["docking_eligible"]:
                    cmd = [
                        MM, "run", "-n", "cosmax-base", "snakemake",
                        "-s", "workflow/Snakefile", "--cores", "16",
                        "--use-conda", "--conda-frontend", "conda", "--rerun-triggers=mtime",
                        "kg_efficacy_label", "--resources", "gpu=1",
                        "--config", f"run_id={run_id}", "mode=fast",
                        f"compound_smiles={smiles}", "run_dti_sanity=true",
                        "evidence_mode=evidence", "target_metadata=data/hpa/proteinatlas.tsv",
                    ]
                    if run(cmd, LOGDIR / f"{tag}_fast.log") == 0:
                        ligand = run_dir / "03_targets/mode_comprehensive/ligand.pdbqt"
                        autodock, gnina, dock_note = pair_dock(
                            run_id, ligand, item["uniprot"], tag
                        )
                        if dock_note:
                            notes.append(dock_note)
                    else:
                        notes.append("fast_failed")
                note = ";".join(notes)
                writer.writerow([item["gene"], item["uniprot"], smiles, item["qed"],
                                 item["sa"], item["mw"], item["feature_recall"], run_id,
                                 state["decision"] or skin, cosmetic, autodock, gnina,
                                 "true" if state["review_required"] else "false",
                                 note, state["availability"], state["applicability"],
                                 "true" if state["claimable"] else "false",
                                 ",".join(state["missing_models"])])
                handle.flush()
                print(f"[b2] {tag} skin={skin} state={state['decision']}/{state['availability']} "
                      f"cosmetic={cosmetic} autodock={autodock} gnina={gnina} {note}",
                      flush=True)
                time.sleep(3)
        print("[b2] done", flush=True)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
