#!/usr/bin/env python3
"""HuSSPred AD 게이트로 실패한 safety 40건을 degraded 모드로 재실행한다.

입력 모델 JSON은 이미 있으므로 consensus·cosmetic 결정만 다시 계산된다(빠름).
결과는 safety_status_degraded.csv에 기록하고, missing_models를 함께 남긴다.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FUNNEL = Path(os.environ.get("FUNNEL_DIR", "/tmp/opencode/analog_funnel"))
SHORTLIST = FUNNEL / "shortlist.csv"
SAFETY = FUNNEL / "safety_status.csv"
OUT = FUNNEL / "safety_status_degraded.csv"
LOGDIR = FUNNEL / "safety_logs"
LOCK = FUNNEL / ".safety_degraded.lock"
MM = os.environ.get("MICROMAMBA_BIN", "micromamba")

sys.path.insert(0, str(ROOT / "scripts"))
import run_skinscout as rs  # noqa: E402
import safety_states  # noqa: E402


def first_line(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except Exception:
        return ""


def report_state(skin: str, report_path: Path) -> dict:
    """Decision plus the nested evidence gaps from an admet_report.json.

    ``missing_models`` lives under ``skin_sens``. Reading the top-level key
    silently produced an empty list, so every degraded row lost the reason it
    was degraded.
    """
    payload: dict = {}
    if report_path.is_file():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    skin_sens = safety_states.nested_skin_sens(payload)
    return safety_states.safety_state(
        skin or skin_sens.get("decision", ""),
        missing_models=skin_sens.get("missing_models"),
        applicability_limited_models=skin_sens.get("applicability_limited_models"),
        degraded=skin_sens.get("degraded"),
    )


def main() -> None:
    if LOCK.exists():
        raise SystemExit(f"lock exists: {LOCK}")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        shortlist = {
            rs._auto_run_id_from_smiles(row["smiles"].strip()): row
            for row in csv.DictReader(SHORTLIST.open(encoding="utf-8-sig"))
        }
        failed = [
            row for row in csv.DictReader(SAFETY.open(encoding="utf-8-sig"))
            if row.get("returncode") != "0"
        ]
        print(f"[degraded] 대상 {len(failed)}건", flush=True)
        degraded_config = rs._snakemake_config_entries(rs.SAFETY_DEGRADED_CONFIG)
        with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["seed_dir", "candidate_id", "run_id", "returncode",
                             "skin_sens", "cosmetic", "missing_models",
                             "degraded", "applicability_limited_models",
                             "claim_eligible", "safety_reason"])
            for row in failed:
                run_id = row["run_id"]
                item = shortlist.get(run_id)
                if item is None:
                    continue
                cmd = [
                    MM, "run", "-n", "cosmax-base", "snakemake",
                    "-s", "workflow/Snakefile", "--cores", "16",
                    "--use-conda", "--conda-frontend", "conda", "--rerun-triggers=mtime",
                    "skin_sens_consensus", "cosmetic_drug_decision", "--resources", "gpu=1",
                    "--config", f"run_id={run_id}", "mode=comprehensive",
                    f"compound_smiles={item['smiles'].strip()}", "run_dti_sanity=true",
                    "evidence_mode=evidence", "target_metadata=data/hpa/proteinatlas.tsv",
                    *degraded_config,
                ]
                log = (LOGDIR / f"degraded_{row['candidate_id']}_{run_id}.log").open(
                    "w", encoding="utf-8"
                )
                print(f"[degraded] {row['candidate_id']} {run_id}", flush=True)
                rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                log.close()
                run_dir = ROOT / "results/runs" / run_id
                skin = first_line(run_dir / "02_admet/skin_sens_decision.txt")
                cosmetic = first_line(run_dir / "02b_cosmetic_drug/cosmetic_drug_decision.txt")
                state = report_state(skin, run_dir / "02_admet/admet_report.json")
                writer.writerow([
                    row["seed_dir"], row["candidate_id"], run_id, rc,
                    state["decision"] or skin, cosmetic,
                    ",".join(state["missing_models"]),
                    "true" if state["degraded"] else "false",
                    ",".join(state["applicability_limited_models"]),
                    "true" if state["claimable"] else "false",
                    state["reason"],
                ])
                handle.flush()
                print(f"[degraded] {row['candidate_id']} rc={rc} skin={skin} "
                      f"state={state['decision']}/{state['availability']} cosmetic={cosmetic}",
                      flush=True)
                time.sleep(3)
        print("[degraded] done", flush=True)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
