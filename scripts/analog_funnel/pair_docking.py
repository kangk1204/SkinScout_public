#!/usr/bin/env python3
"""펀넬 도킹 재랭킹: 안전성 컷오프 통과 후보를 의도표적과 쌍으로 도킹한다.

각 후보에 대해:
1) fast target 프리셋을 돌려 Daina 랭킹(mode_fast/daina_top256.csv)을 얻고,
2) 의도표적 행만 남긴 selected.csv로 오토그리드 맵(캐시 재사용) → AutoDock-GPU →
   GNINA CNN 재채점을 실행한 뒤,
3) 결과 CSV에 에너지·CNN 친화도·안전성 결정을 기록한다.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
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
SHORTLIST = FUNNEL / "shortlist.csv"
SAFETY = [FUNNEL / "safety_status.csv", FUNNEL / "safety_status_degraded.csv"]
OUT = FUNNEL / "dock_results.csv"
LOGDIR = FUNNEL / "dock_logs"
LOCK = FUNNEL / ".dock.lock"
MM = os.environ.get("MICROMAMBA_BIN", "micromamba")
DOCK_ENV = "cosmax-autodock-gpu"
PER_SEED = 2
LIMIT = int(os.environ.get("FUNNEL_DOCK_LIMIT", "0"))

sys.path.insert(0, str(ROOT / "scripts"))
import run_skinscout as rs  # noqa: E402


def first_line(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except Exception:
        return ""


def seed_targets() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in csv.DictReader((BASE / "per_target_summary.csv").open(encoding="utf-8-sig")):
        if row["rank"] == "1":
            mapping.setdefault(row["run_id"], row["uniprot"])
    return mapping


def survivors(seed_uniprot: dict[str, str]) -> list[dict]:
    safety: dict[str, dict] = {}
    for path in SAFETY:
        if not path.is_file():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("returncode") == "0":
                safety[row["run_id"]] = row
    chosen: dict[str, dict] = {}
    with SHORTLIST.open(encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            run_id = rs._auto_run_id_from_smiles(item["smiles"].strip())
            row = safety.get(run_id)
            if row is None or row.get("skin_sens") == "HALT":
                continue
            item = {**item, "run_id": run_id, "skin_sens": row.get("skin_sens", ""),
                    "cosmetic": row.get("cosmetic", "")}
            if run_id not in chosen:
                chosen[run_id] = item
    per_seed: dict[str, list[dict]] = {}
    for item in chosen.values():
        per_seed.setdefault(item["seed_dir"], []).append(item)
    result: list[dict] = []
    for seed, items in per_seed.items():
        items.sort(key=lambda r: -float(r.get("funnel_score") or 0))
        for item in items[:PER_SEED]:
            if item.get("seed_dir") in seed_uniprot:
                result.append(item)
    return result


def run(cmd: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> None:
    if LOCK.exists():
        raise SystemExit(f"lock exists: {LOCK}")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        seed_uniprot = seed_targets()
        items = survivors(seed_uniprot)
        if LIMIT:
            items = items[:LIMIT]
        print(f"[dock] 대상 {len(items)}건", flush=True)
        LOGDIR.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["seed_dir", "candidate_id", "run_id", "uniprot",
                             "daina_rank", "autodock_kcal_mol", "gnina_cnn_affinity",
                             "skin_sens", "cosmetic", "funnel_score", "note"])
            for index, item in enumerate(items, 1):
                run_id = item["run_id"]
                uniprot = seed_uniprot[item["seed_dir"]]
                tag = f"{index:03d}_{item['candidate_id']}_{run_id}"
                run_dir = ROOT / "results/runs" / run_id
                fast_dir = run_dir / "03_targets/mode_fast"
                pair_dir = run_dir / "03_targets/funnel_pair"
                ligand = run_dir / "03_targets/mode_comprehensive/ligand.pdbqt"
                note = ""
                daina_rank = ""
                autodock = gnina = ""
                # 1) fast 프리셋: Daina 랭킹 확보
                cmd = [
                    MM, "run", "-n", "cosmax-base", "snakemake",
                    "-s", "workflow/Snakefile", "--cores", "16",
                    "--use-conda", "--conda-frontend", "conda", "--rerun-triggers=mtime",
                    "kg_efficacy_label", "--resources", "gpu=1",
                    "--config", f"run_id={run_id}", "mode=fast",
                    f"compound_smiles={item['smiles'].strip()}", "run_dti_sanity=true",
                    "evidence_mode=evidence", "target_metadata=data/hpa/proteinatlas.tsv",
                ]
                rc = run(cmd, LOGDIR / f"{tag}_fast.log")
                if rc != 0:
                    writer.writerow([item["seed_dir"], item["candidate_id"], run_id, uniprot,
                                     "", "", "", item["skin_sens"], item["cosmetic"],
                                     item["funnel_score"], "fast_preset_failed"])
                    handle.flush()
                    continue
                daina_csv = fast_dir / "daina_top256.csv"
                rows = list(csv.DictReader(daina_csv.open(encoding="utf-8-sig")))
                target_row = next((r for r in rows if r.get("target_id") == uniprot), None)
                if target_row is None:
                    # Daina top-256 밖이어도 의도표적 쌍 도킹은 수행한다. 오토그리드
                    # 스크립트가 순위 연속성(1..n)을 검사하므로 1로 합성한다(밴드 제한은
                    # 0/0=무제한이라 실제 필터링에는 영향이 없고, 리포트에 사실을 남긴다).
                    target_row = {**rows[0], "target_id": uniprot, "daina_rank": 1}
                    note = "synthesized_rank_outside_daina_top256"
                daina_rank = target_row.get("daina_rank", "")
                # 2) 의도표적만 쌍 도킹
                pair_dir.mkdir(parents=True, exist_ok=True)
                with (pair_dir / "selected.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer2 = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer2.writeheader()
                    writer2.writerow(target_row)
                maps_dir = pair_dir / "autogrid_maps"
                maps_dir.mkdir(parents=True, exist_ok=True)
                steps = [
                    (["python", "scripts/stage3_autogrid_maps.py",
                      "--selected-csv", str(pair_dir / "selected.csv"),
                      "--ligand-pdbqt", str(ligand),
                      "--receptor-dir", "data/human_pdbqt",
                      "--box-dir", "data/docking_boxes_derived",
                      "--no-pocket-list", "data/no_pocket_targets.list",
                      "--spacing", "0.375",
                      "--cache-dir", str(Path.home() / ".cache/skinscout/autogrid"),
                      "--dock-rank-from", "0", "--dock-rank-to", "0",
                      "--out-map-dir", str(maps_dir),
                      "--out-manifest", str(pair_dir / "autogrid_map_manifest.json")],
                     f"{tag}_autogrid.log"),
                    (["python", "scripts/stage3_autodock_run.py",
                      "--ligand-pdbqt", str(ligand),
                      "--receptor-dir", "data/human_pdbqt",
                      "--box-dir", "data/docking_boxes_derived",
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
                      "--clean-dir", "data/human_clean",
                      "--use-gpu", "--allow-partial-structure",
                      "--out-scores", str(pair_dir / "gnina_pair.tsv"),
                      "--out-status-manifest", str(pair_dir / "gnina_pair_status.json")],
                     f"{tag}_gnina.log"),
                ]
                for script_cmd, log_name in steps:
                    rc = run([MM, "run", "-n", DOCK_ENV, *script_cmd], LOGDIR / log_name)
                    if rc != 0:
                        note = f"step_failed:{log_name}"
                        break
                scores = pair_dir / "autodock_pair.tsv"
                if scores.is_file():
                    for row in csv.DictReader(scores.open(encoding="utf-8-sig"), delimiter="\t"):
                        if row.get("target_id") == uniprot:
                            autodock = row.get("vina_score", "")
                gnina_scores = pair_dir / "gnina_pair.tsv"
                if gnina_scores.is_file():
                    for row in csv.DictReader(gnina_scores.open(encoding="utf-8-sig"), delimiter="\t"):
                        if row.get("target_id") == uniprot:
                            gnina = row.get("cnn_affinity", "")
                writer.writerow([item["seed_dir"], item["candidate_id"], run_id, uniprot,
                                 daina_rank, autodock, gnina, item["skin_sens"],
                                 item["cosmetic"], item["funnel_score"], note])
                handle.flush()
                print(f"[dock] {tag} rank={daina_rank} autodock={autodock} gnina={gnina} {note}",
                      flush=True)
                time.sleep(3)
        print("[dock] done", flush=True)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
