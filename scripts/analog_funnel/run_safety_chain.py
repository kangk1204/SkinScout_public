#!/usr/bin/env python3
"""유사체 펀넬 체인(1단계): 생성 완료 대기 → 선별(게이트+MMR) → safety 프리셋 일괄.

결과: shortlist.csv, safety_status.csv (HALT/FLAG 등 결정 기록)
도킹 재랭킹은 safety 결과를 보고 별도 단계로 실행한다.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FUNNEL_HOME = Path(os.environ.get("FUNNEL_HOME", ""))
if not FUNNEL_HOME.is_dir():
    raise SystemExit("FUNNEL_HOME 환경변수에 비공개 워크스페이스 경로를 지정하세요.")
BASE = FUNNEL_HOME / "target_pipeline_20260913"
FUNNEL = Path("/tmp/opencode/analog_funnel")
GEN = FUNNEL / "gen"
SHORTLIST = FUNNEL / "shortlist.csv"
SAFETY_STATUS = FUNNEL / "safety_status.csv"
LOGDIR = FUNNEL / "safety_logs"
LOCK = FUNNEL / ".chain.lock"
MM = os.environ.get("MICROMAMBA_BIN", "micromamba")
EXPECTED_SEEDS = 13

sys.path.insert(0, str(ROOT / "scripts"))
from run_skinscout import _auto_run_id_from_smiles  # noqa: E402


def wait_for_generation() -> None:
    while True:
        done = len(list(GEN.glob("*/substitute_report.json")))
        running = subprocess.run(
            ["pgrep", "-fc", "discover_substitutes"], capture_output=True, text=True
        ).stdout.strip()
        print(f"[chain] 생성 {done}/{EXPECTED_SEEDS} (실행 중 {running})", flush=True)
        if done >= EXPECTED_SEEDS or (done > 0 and running == "0"):
            return
        time.sleep(60)


def run_selection() -> None:
    print("[chain] 선별(게이트+MMR) 실행", flush=True)
    subprocess.run(
        [MM, "run", "-n", "cosmax-base", "python", "/tmp/opencode/analog_funnel_select.py"],
        cwd=ROOT, check=True,
    )


def read_first_line(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except Exception:
        return ""


def run_safety(shortlist: list[dict]) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if SAFETY_STATUS.is_file():
        done = {
            row["run_id"]
            for row in csv.DictReader(SAFETY_STATUS.open(encoding="utf-8-sig"))
            if row.get("returncode") == "0"
        }
    with SAFETY_STATUS.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if not done:
            writer.writerow(["seed_dir", "candidate_id", "run_id", "returncode",
                             "skin_sens", "cosmetic"])
        seen: set[str] = set()
        for item in shortlist:
            smiles = item["smiles"].strip()
            run_id = _auto_run_id_from_smiles(smiles)
            if run_id in done or run_id in seen:
                continue
            seen.add(run_id)
            cmd = [
                MM, "run", "-n", "cosmax-base", "snakemake",
                "-s", "workflow/Snakefile", "--cores", "16",
                "--use-conda", "--conda-frontend", "conda", "--rerun-triggers=mtime",
                "skin_sens_consensus", "cosmetic_drug_decision", "--resources", "gpu=1",
                "--config", f"run_id={run_id}", "mode=comprehensive",
                f"compound_smiles={smiles}", "run_dti_sanity=true",
                "evidence_mode=evidence", "target_metadata=data/hpa/proteinatlas.tsv",
            ]
            log = (LOGDIR / f"{item['seed_dir']}_{item['candidate_id']}_{run_id}.log").open(
                "w", encoding="utf-8"
            )
            print(f"[chain] safety {item['candidate_id']} {run_id}", flush=True)
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            log.close()
            run_dir = ROOT / "results/runs" / run_id
            skin = read_first_line(run_dir / "02_admet/skin_sens_decision.txt")
            cosmetic = read_first_line(run_dir / "02b_cosmetic_drug/cosmetic_drug_decision.txt")
            writer.writerow([item["seed_dir"], item["candidate_id"], run_id, rc, skin, cosmetic])
            handle.flush()
            print(f"[chain] safety {item['candidate_id']} rc={rc} skin={skin} cosmetic={cosmetic}",
                  flush=True)
            time.sleep(5)
    print("[chain] safety done", flush=True)


def main() -> None:
    if LOCK.exists():
        raise SystemExit(f"chain lock exists: {LOCK}")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        wait_for_generation()
        run_selection()
        with SHORTLIST.open(encoding="utf-8-sig") as handle:
            shortlist = list(csv.DictReader(handle))
        print(f"[chain] 단축 후보 {len(shortlist)}", flush=True)
        run_safety(shortlist)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
