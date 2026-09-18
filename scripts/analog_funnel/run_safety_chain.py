#!/usr/bin/env python3
"""유사체 펀넬 체인(1단계): 생성 완료 대기 → 선별(게이트+MMR) → safety 프리셋 일괄.

결과: shortlist.csv, safety_status.csv (HALT/FLAG 등 결정 기록)
도킹 재랭킹은 safety 결과를 보고 별도 단계로 실행한다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
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
CHAIN_MANIFEST = FUNNEL / "chain_manifest.json"
SELECT_SCRIPT = ROOT / "scripts/analog_funnel/select_candidates.py"
MM = os.environ.get("MICROMAMBA_BIN", "micromamba")
SELECT_ENV = os.environ.get("FUNNEL_PYTHON_ENV", "cosmax-base")
EXPECTED_SEEDS = 13
WAIT_DEADLINE_SECONDS = float(
    os.environ.get("FUNNEL_GENERATION_DEADLINE_SECONDS", "86400")
)
POLL_SECONDS = float(os.environ.get("FUNNEL_POLL_SECONDS", "60"))
CODE_FILES = (
    Path(__file__),
    SELECT_SCRIPT,
    ROOT / "workflow/Snakefile",
    ROOT / "workflow/config.yaml",
)

sys.path.insert(0, str(ROOT / "scripts"))
from run_skinscout import _auto_run_id_from_smiles  # noqa: E402


class GenerationFailed(RuntimeError):
    """Generation cannot reach a terminal state and the chain must not wait."""


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def running_generation_processes() -> int:
    completed = subprocess.run(
        ["pgrep", "-fc", "discover_substitutes"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise GenerationFailed(f"pgrep failed with rc={completed.returncode}")
    return int(completed.stdout.strip() or "0")


def wait_for_generation(
    *,
    gen: Path | None = None,
    expected_seeds: int = EXPECTED_SEEDS,
    deadline_seconds: float | None = None,
    poll_seconds: float | None = None,
    process_counter: Callable[[], int] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for terminal generation with a deadline and an explicit 0/0 failure."""
    gen = gen if gen is not None else GEN
    deadline_seconds = (
        WAIT_DEADLINE_SECONDS if deadline_seconds is None else deadline_seconds
    )
    poll_seconds = POLL_SECONDS if poll_seconds is None else poll_seconds
    counter = process_counter or running_generation_processes
    deadline = clock() + deadline_seconds
    while True:
        done = len(list(gen.glob("*/substitute_report.json")))
        running = counter()
        print(f"[chain] 생성 {done}/{expected_seeds} (실행 중 {running})", flush=True)
        if done >= expected_seeds:
            return
        if done == 0 and running == 0:
            raise GenerationFailed(
                "generation finished with 0 reports and no running process; "
                "refusing to wait forever"
            )
        if done > 0 and running == 0:
            return
        if clock() >= deadline:
            raise GenerationFailed(
                f"generation deadline of {deadline_seconds}s exceeded at "
                f"{done}/{expected_seeds}"
            )
        sleep(poll_seconds)


def run_selection() -> None:
    if not SELECT_SCRIPT.is_file():
        raise GenerationFailed(f"selection entrypoint missing: {SELECT_SCRIPT}")
    print("[chain] 선별(게이트+MMR) 실행", flush=True)
    subprocess.run(
        [MM, "run", "-n", SELECT_ENV, "python", str(SELECT_SCRIPT)],
        cwd=ROOT, check=True,
    )


def code_hashes() -> list[dict]:
    return [
        {"path": str(path), "sha256": file_digest(path)}
        for path in CODE_FILES
        if path.is_file()
    ]


def write_chain_manifest(stage: str, **sections: object) -> None:
    FUNNEL.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "skinscout.analog-chain-manifest.v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "code": code_hashes(),
        **sections,
    }
    temporary = CHAIN_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(CHAIN_MANIFEST)


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
        write_chain_manifest(
            "waiting_for_generation",
            generation={
                "expected_seeds": EXPECTED_SEEDS,
                "deadline_seconds": WAIT_DEADLINE_SECONDS,
                "poll_seconds": POLL_SECONDS,
            },
        )
        wait_for_generation()
        run_selection()
        write_chain_manifest(
            "selection_complete",
            selection={
                "script": str(SELECT_SCRIPT),
                "sha256": file_digest(SELECT_SCRIPT),
                "interpreter": {"micromamba": MM, "env": SELECT_ENV},
            },
        )
        with SHORTLIST.open(encoding="utf-8-sig") as handle:
            shortlist = list(csv.DictReader(handle))
        print(f"[chain] 단축 후보 {len(shortlist)}", flush=True)
        run_safety(shortlist)
        write_chain_manifest(
            "complete",
            selection={
                "script": str(SELECT_SCRIPT),
                "sha256": file_digest(SELECT_SCRIPT),
                "interpreter": {"micromamba": MM, "env": SELECT_ENV},
            },
            safety_run={
                "interpreter": {"micromamba": MM, "env": "cosmax-base"},
                "mode": "comprehensive",
                "run_dti_sanity": True,
                "evidence_mode": "evidence",
                "target_metadata": "data/hpa/proteinatlas.tsv",
                "selections": ["skin_sens_consensus", "cosmetic_drug_decision"],
                "resources": {"cores": 16, "gpu": 1},
            },
        )
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
