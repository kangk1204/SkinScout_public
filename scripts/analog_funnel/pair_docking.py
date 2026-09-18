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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
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
DOCK_MANIFEST = FUNNEL / "dock_manifest.json"
MM = os.environ.get("MICROMAMBA_BIN", "micromamba")
DOCK_ENV = "cosmax-autodock-gpu"
PER_SEED = 2
LIMIT = int(os.environ.get("FUNNEL_DOCK_LIMIT", "0"))
# 출력 CSV는 고정 schema로만 쓴다. 실행 실패(note)와 정보 note, 안전성 사유를
# 각자의 열에 두어 헤더/값 개수가 어긋날 수 없게 한다.
OUT_FIELDS = (
    "seed_dir", "candidate_id", "run_id", "uniprot",
    "daina_rank", "autodock_kcal_mol", "gnina_cnn_affinity",
    "skin_sens", "cosmetic", "funnel_score", "review_required",
    "claim_eligible", "safety_reason", "note",
)
DOCK_CONFIG = {
    "per_seed": PER_SEED,
    "limit": LIMIT,
    "interpreter": {"micromamba": MM, "env": DOCK_ENV},
    "work_key": "ligand_identity+target",
}
CODE_FILES = (
    Path(__file__),
    ROOT / "scripts/stage3_autogrid_maps.py",
    ROOT / "scripts/stage3_autodock_run.py",
    ROOT / "scripts/stage3_gnina_rescore.py",
    ROOT / "workflow/Snakefile",
    ROOT / "workflow/config.yaml",
)

sys.path.insert(0, str(ROOT / "scripts"))
import run_skinscout as rs  # noqa: E402
import safety_states  # noqa: E402


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_dock_manifest(stage: str, items: list[dict], config: dict) -> None:
    FUNNEL.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "skinscout.analog-dock-manifest.v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "config": config,
        "work_items": [
            {
                key: item.get(key)
                for key in ("seed_dir", "candidate_id", "run_id", "uniprot")
            }
            for item in items
        ],
        "code": [
            {"path": str(path), "sha256": file_digest(path)}
            for path in CODE_FILES
            if path.is_file()
        ],
    }
    temporary = DOCK_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(DOCK_MANIFEST)


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


# 도킹 생존 조건: 진단 도킹이 가능한 결정만. 빈 값/미정/HALT/파일 부재는 검증
# 미완료로 취급한다. FLAG_HIGH 와 degraded(모델 결손)는 진단 도킹을 진행하되
# 주장 가능한 PASS 로 승격하지 않고 사유를 끝까지 보존한다.
SAFE_DECISIONS = safety_states.SAFE_DECISIONS


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
    chosen: dict[tuple[str, str], dict] = {}
    with SHORTLIST.open(encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            run_id = rs._auto_run_id_from_smiles(item["smiles"].strip())
            target = seed_uniprot.get(item["seed_dir"])
            if target is None:
                continue
            row = safety.get(run_id)
            if row is None:
                continue
            state = safety_states.safety_state(
                row.get("skin_sens") or "",
                missing_models=row.get("missing_models"),
                applicability_limited_models=row.get("applicability_limited_models"),
                degraded=row.get("degraded"),
            )
            if not state["docking_eligible"]:
                # HALT/빈값/미정/UNKNOWN은 도킹 생존 조건이 아니다(fail-closed).
                continue
            item = {**item, "run_id": run_id, "uniprot": target,
                    "skin_sens": state["decision"],
                    "cosmetic": row.get("cosmetic", ""),
                    "review_required": state["review_required"],
                    "claim_eligible": state["claimable"],
                    "safety_reason": state["reason"]}
            # 도킹 작업 identity는 ligand+target이다. 같은 리간드라도 표적이
            # 다르면 별개 작업이므로 seed(=표적)별로 모두 남긴다.
            chosen.setdefault((run_id, target), item)
    per_seed: dict[str, list[dict]] = {}
    for item in chosen.values():
        per_seed.setdefault(item["seed_dir"], []).append(item)
    result: list[dict] = []
    for items in per_seed.values():
        items.sort(key=lambda r: -float(r.get("funnel_score") or 0))
        result.extend(items[:PER_SEED])
    return result


def run(cmd: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def result_row(
    item: dict,
    *,
    daina_rank: str = "",
    autodock: str = "",
    gnina: str = "",
    note: str = "",
) -> dict:
    """고정 schema에 맞춘 출력 행. 키 순서와 집합이 OUT_FIELDS와 일치한다."""
    return {
        "seed_dir": item["seed_dir"],
        "candidate_id": item["candidate_id"],
        "run_id": item["run_id"],
        "uniprot": item["uniprot"],
        "daina_rank": daina_rank,
        "autodock_kcal_mol": autodock,
        "gnina_cnn_affinity": gnina,
        "skin_sens": item.get("skin_sens", ""),
        "cosmetic": item.get("cosmetic", ""),
        "funnel_score": item.get("funnel_score", ""),
        "review_required": item.get("review_required", ""),
        "claim_eligible": item.get("claim_eligible", ""),
        "safety_reason": item.get("safety_reason", ""),
        "note": note,
    }


def read_fresh_scores(pair_dir: Path, uniprot: str) -> tuple[str, str]:
    """이번 실행이 남긴 autodock/gnina 점수만 읽는다(과거 파일은 이미 지운 뒤다)."""
    autodock = gnina = ""
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
    return autodock, gnina


def dock_item(item: dict, index: int) -> dict:
    """한 후보의 fast 프리셋 + 의도표적 쌍 도킹을 실행하고 출력 행을 돌려준다.

    실행 실패(failure)와 정보 note(informational)를 분리해 추적한다. 정보 note가
    붙은 성공 행도 단계가 성공했으면 이번 실행의 신선한 점수를 읽는다.
    """
    run_id = item["run_id"]
    uniprot = item["uniprot"]
    tag = f"{index:03d}_{item['candidate_id']}_{run_id}"
    run_dir = ROOT / "results/runs" / run_id
    fast_dir = run_dir / "03_targets/mode_fast"
    pair_dir = run_dir / "03_targets/funnel_pair"
    ligand = run_dir / "03_targets/mode_comprehensive/ligand.pdbqt"
    informational: list[str] = []
    failure = ""
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
    if run(cmd, LOGDIR / f"{tag}_fast.log") != 0:
        failure = "fast_preset_failed"
        note = ";".join(p for p in (failure, *informational) if p)
        print(f"[dock] {tag} note={note}", flush=True)
        return result_row(item, note=note)
    daina_csv = fast_dir / "daina_top256.csv"
    rows = list(csv.DictReader(daina_csv.open(encoding="utf-8-sig")))
    target_row = next((r for r in rows if r.get("target_id") == uniprot), None)
    if target_row is None:
        # Daina top-256 밖이어도 의도표적 쌍 도킹은 수행한다. 오토그리드
        # 스크립트가 순위 연속성(1..n)을 검사하므로 1로 합성한다(밴드 제한은
        # 0/0=무제한이라 실제 필터링에는 영향이 없고, 리포트에 사실을 남긴다).
        target_row = {**rows[0], "target_id": uniprot, "daina_rank": 1}
        informational.append("synthesized_rank_outside_daina_top256")
    daina_rank = str(target_row.get("daina_rank", ""))
    # 2) 의도표적만 쌍 도킹
    pair_dir.mkdir(parents=True, exist_ok=True)
    # 이번 실행의 산출물만 수용한다(과거 점수 재사용 금지).
    for stale in ("autodock_pair.tsv", "gnina_pair.tsv",
                  "autodock_pair_status.json", "gnina_pair_status.json"):
        (pair_dir / stale).unlink(missing_ok=True)
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
    for script_cmd, log_name in steps:
        if run([MM, "run", "-n", DOCK_ENV, *script_cmd], LOGDIR / log_name) != 0:
            failure = f"step_failed:{log_name}"
            break
    if not failure:
        # 정보 note가 있어도 실행이 성공했으면 이번 점수를 읽는다.
        autodock, gnina = read_fresh_scores(pair_dir, uniprot)
    note = ";".join(p for p in (failure, *informational) if p)
    detail = ";".join(p for p in (note, str(item.get("safety_reason", "") or "")) if p)
    print(f"[dock] {tag} rank={daina_rank} autodock={autodock} gnina={gnina} {detail}",
          flush=True)
    return result_row(item, daina_rank=daina_rank, autodock=autodock, gnina=gnina, note=note)


def main() -> None:
    if LOCK.exists():
        raise SystemExit(f"lock exists: {LOCK}")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        seed_uniprot = seed_targets()
        items = survivors(seed_uniprot)
        if LIMIT:
            items = items[:LIMIT]
        write_dock_manifest("selected", items, DOCK_CONFIG)
        print(f"[dock] 대상 {len(items)}건", flush=True)
        LOGDIR.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(OUT_FIELDS))
            writer.writeheader()
            for index, item in enumerate(items, 1):
                writer.writerow(dock_item(item, index))
                handle.flush()
                time.sleep(3)
        write_dock_manifest("complete", items, DOCK_CONFIG)
        print("[dock] done", flush=True)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
