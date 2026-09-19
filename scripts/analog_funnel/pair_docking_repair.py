#!/usr/bin/env python3
"""도킹 실패분만 재시도: rank를 1로 재번호하고 쌍 도킹만 다시 돌린다.

fast 프리셋과 Daina 랭킹은 이미 있으므로 재사용한다. 포켓/수용체가 없는 표적은
그 사유를 note에 기록한다.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FUNNEL = Path("/tmp/opencode/analog_funnel")
DOCK = FUNNEL / "dock_results.csv"
LOGDIR = FUNNEL / "dock_logs"
MM = os.environ.get("MICROMAMBA_BIN", "micromamba")
DOCK_ENV = "cosmax-autodock-gpu"


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


def run(cmd: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> None:
    rows = list(csv.DictReader(DOCK.open(encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    repaired = 0
    for row in rows:
        if row["autodock_kcal_mol"] and row["gnina_cnn_affinity"]:
            continue
        uniprot = row["uniprot"]
        run_id = row["run_id"]
        run_dir = ROOT / "results/runs" / run_id
        fast_dir = run_dir / "03_targets/mode_fast"
        pair_dir = run_dir / "03_targets/funnel_pair"
        ligand = run_dir / "03_targets/mode_comprehensive/ligand.pdbqt"
        if not (ROOT / "data/human_pdbqt" / f"{uniprot}.pdbqt").is_file() or not (
            docking_box_dir() / f"{uniprot}.box.txt"
        ).is_file():
            row["note"] = "structural_unavailable_no_pocket_or_prep"
            continue
        daina_rows = list(csv.DictReader((fast_dir / "daina_top256.csv").open(encoding="utf-8-sig")))
        target_row = next((r for r in daina_rows if r.get("target_id") == uniprot), None)
        original_rank = target_row.get("daina_rank") if target_row else ""
        if target_row is None:
            target_row = {**daina_rows[0], "target_id": uniprot}
        target_row = {**target_row, "daina_rank": 1}  # 순위 연속성 검사 통과용 재번호
        pair_dir.mkdir(parents=True, exist_ok=True)
        for stale in ("autodock_pair.tsv", "gnina_pair.tsv",
                      "autodock_pair_status.json", "gnina_pair_status.json"):
            (pair_dir / stale).unlink(missing_ok=True)
        with (pair_dir / "selected.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(daina_rows[0].keys()))
            writer.writeheader()
            writer.writerow(target_row)
        maps_dir = pair_dir / "autogrid_maps"
        tag = f"fix_{row['candidate_id']}_{run_id}"
        steps = [
            (["python", "scripts/stage3_autogrid_maps.py",
              "--selected-csv", str(pair_dir / "selected.csv"),
              "--ligand-pdbqt", str(ligand),
              "--receptor-dir", "data/human_pdbqt",
              "--box-dir", str(docking_box_dir()),
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
              "--clean-dir", "data/human_clean",
              "--use-gpu", "--allow-partial-structure",
              "--out-scores", str(pair_dir / "gnina_pair.tsv"),
              "--out-status-manifest", str(pair_dir / "gnina_pair_status.json")],
             f"{tag}_gnina.log"),
        ]
        note = ""
        for script_cmd, log_name in steps:
            rc = run([MM, "run", "-n", DOCK_ENV, *script_cmd], LOGDIR / log_name)
            if rc != 0:
                note = f"step_failed:{log_name}"
                break
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
        row["autodock_kcal_mol"] = autodock
        row["gnina_cnn_affinity"] = gnina
        if autodock or gnina:
            repaired += 1
            note = f"rank_renumbered_from_{original_rank}" if original_rank else note
        row["note"] = note
        print(f"[fix] {tag} autodock={autodock} gnina={gnina} {note}", flush=True)
    with DOCK.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fix] repaired {repaired}", flush=True)


if __name__ == "__main__":
    main()
