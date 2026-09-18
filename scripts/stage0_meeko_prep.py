#!/usr/bin/env python3
"""stage0_meeko_prep.py

Generate AutoDock-GPU receptor PDBQT + box file for every cleaned protein that
has at least one usable pocket. Receptors listed in `no_pocket_targets.list`
are skipped here and re-routed to DiffDock-L at Stage 3.

Box construction (INSTRUCTIONS.md §3.2 (d)): center on the highest-druggability
P2Rank pocket; box edge = 2 × (max(pocket_radius, 8.0) + 4.0) Å, capped at 30 Å.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pocket_box import PocketBoxError, box_for_target, format_box_text
from stage0_manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    sha256_file,
    write_manifest,
)

LOG = logging.getLogger("stage0.meeko")
CENTER_FIELDS = ("center_x", "center_y", "center_z")


@dataclass(frozen=True)
class Job:
    uniprot: str
    clean_pdb: Path
    pocket_json: Path
    pocket_prediction: Path
    pocket: dict
    out_pdbqt: Path
    out_box: Path


def best_pocket(pocket_json: Path) -> dict | None:
    if not pocket_json.exists():
        return None
    payload = json.loads(pocket_json.read_text())
    pockets = payload.get("pockets") or []
    if not pockets:
        return None
    def pocket_score(pocket: dict, idx: int) -> float:
        key = "druggability" if "druggability" in pocket else "score"
        if key not in pocket:
            raise SystemExit(
                f"{pocket_json} pocket index {idx} missing required score field "
                "('druggability' or 'score')"
            )
        return _finite_pocket_float(pocket[key], pocket_json, idx, key)

    scored = [
        (pocket_score(pocket, idx), validated_pocket(pocket, idx, pocket_json))
        for idx, pocket in enumerate(pockets)
    ]
    return max(scored, key=lambda item: item[0])[1]


def _is_bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    )


def _finite_pocket_float(
    value: object,
    pocket_json: Path,
    idx: int,
    key: str,
) -> float:
    if _is_bool_like(value):
        raise SystemExit(
            f"{pocket_json} pocket index {idx} field '{key}' must be numeric, not boolean"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"{pocket_json} pocket index {idx} field '{key}' must be numeric: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise SystemExit(
            f"{pocket_json} pocket index {idx} field '{key}' must be finite: {value!r}"
        )
    return parsed


def validated_pocket(pocket: dict, idx: int, pocket_json: Path) -> dict:
    center = pocket.get("center")
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        raise SystemExit(
            f"{pocket_json} pocket index {idx} field 'center' must contain "
            "three numeric coordinates"
        )
    parsed_center = tuple(
        _finite_pocket_float(value, pocket_json, idx, key)
        for key, value in zip(CENTER_FIELDS, center, strict=True)
    )
    radius = _finite_pocket_float(pocket.get("radius", 12), pocket_json, idx, "radius")
    if radius <= 0.0:
        raise SystemExit(
            f"{pocket_json} pocket index {idx} field 'radius' must be positive: {radius!r}"
        )
    return {**pocket, "center": parsed_center, "radius": radius}


def write_box(
    box_path: Path,
    pocket: dict,
    *,
    prediction: Path | None = None,
    clean_pdb: Path | None = None,
    legacy_cube_edge: float | None = None,
) -> None:
    """Emit the docking box for one pocket.

    Sizes the box from the rank-1 pocket's surface atoms. The former
    ``min(2 * (max(radius, 8) + 4), 30)`` rule carried no geometry: every
    ``pockets.json`` stores the constant fallback ``radius = 12.0``, so it
    collapsed to the 30 A cap for every receptor. Falling back to that cube is
    now an explicit opt-in rather than what happens when the real extent is
    unavailable.
    """
    if legacy_cube_edge is None:
        if prediction is None or clean_pdb is None:
            raise SystemExit(
                "write_box requires the P2Rank prediction and cleaned structure "
                "to derive a box; pass legacy_cube_edge only for explicit diagnostics"
            )
        try:
            box = box_for_target(prediction, clean_pdb)
        except PocketBoxError as exc:
            raise SystemExit(f"pocket box could not be derived: {exc}") from exc
        _write_box_text(box_path, format_box_text(box))
        return
    cx, cy, cz = pocket["center"]
    edge = float(legacy_cube_edge)
    if not math.isfinite(edge) or edge <= 0.0:
        raise SystemExit(f"legacy_cube_edge must be positive and finite: {edge!r}")
    _write_box_text(
        box_path,
        "\n".join([
            f"center_x = {cx:.3f}",
            f"center_y = {cy:.3f}",
            f"center_z = {cz:.3f}",
            f"size_x = {edge:.3f}",
            f"size_y = {edge:.3f}",
            f"size_z = {edge:.3f}",
            "",
        ]),
    )


def _write_box_text(box_path: Path, text: str) -> None:
    box_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = box_path.with_name(f".{box_path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(box_path)


def remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def nonzero_charge_fraction(pdbqt: Path) -> float:
    """PDBQT 원자 중 부분전하가 0 이 아닌 것의 비율.

    PDBQT 의 전하는 71-76 열이다. 파일이 있고 크기가 0 이 아니어도 이 값이
    0 이면 autogrid4 가 받지 않는다:
    "ERROR: No partial atomic charges were found in the receptor PDBQT file".
    """
    total = 0
    nonzero = 0
    try:
        for line in pdbqt.read_text(errors="replace").splitlines():
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 76:
                continue
            total += 1
            try:
                if abs(float(line[70:76])) > 1e-6:
                    nonzero += 1
            except ValueError:
                continue
    except OSError:
        return 0.0
    return nonzero / total if total else 0.0


# 전하가 붙었다고 보려면 이 비율은 넘어야 한다. 실측: 제대로 만든 파일은
# 2,342 원자 중 2,234 개(95.4%)가 0 이 아니다. 수소 없는 중원자 표현에서
# 일부 원자의 게이스타이거 전하가 0 에 가깝게 나오는 것은 정상이다.
MIN_NONZERO_CHARGE_FRACTION = 0.5

# OpenBabel 폴백은 기본으로 쓰지 않는다. 만들어 내는 수용체가 meeko 것과 화학적으로
# 달라(극성 수소 없음, 질소 타이핑 반전) 같은 순위표에 섞이면 안 되기 때문이다.
# 진단 목적으로만 --allow-obabel-fallback 으로 켠다.
ALLOW_OBABEL_FALLBACK = False


def _obabel_fallback(job: Job) -> tuple[str, bool, str]:
    """OpenBabel rigid-receptor PDBQT (`-xr`). Robust on AlphaFold models where
    meeko's strict residue-template matching rejects His protonation states /
    pLDDT-trimmed residues. AutoDock-GPU accepts OpenBabel PDBQT.

    `--partialcharge gasteiger` 가 없으면 OpenBabel 은 전하를 전부 0 으로 적는다.
    파일은 정상으로 보이고 obabel 은 0 으로 끝나지만, autogrid4 가 그 파일을
    거부한다 - 실측으로 15,038 개 중 1,699 개(11.3%)가 이렇게 만들어져 맵을
    못 얻었고, 그 표적들은 도킹 자체가 되지 않아 순위표에 등장하지 못했다.
    """
    if not shutil.which("obabel"):
        return (job.uniprot, False, "meeko failed and obabel not on PATH")
    res = subprocess.run(
        ["obabel", str(job.clean_pdb), "-O", str(job.out_pdbqt), "-xr",
         "--partialcharge", "gasteiger"],
        capture_output=True, text=True,
    )
    if res.returncode == 0 and job.out_pdbqt.exists() and job.out_pdbqt.stat().st_size > 0:
        # 종료코드만 보면 안 된다. 전하가 0 인 파일도 0 으로 끝난다.
        fraction = nonzero_charge_fraction(job.out_pdbqt)
        if fraction < MIN_NONZERO_CHARGE_FRACTION:
            return (job.uniprot, False,
                    f"obabel: 부분전하가 붙지 않았습니다 (0 아닌 비율 {fraction:.1%}); "
                    "autogrid4 가 이 파일을 거부합니다")
        write_box(
            job.out_box,
            job.pocket,
            prediction=job.pocket_prediction,
            clean_pdb=job.clean_pdb,
        )
        return (job.uniprot, True, "obabel")
    return (job.uniprot, False, ("obabel: " + res.stderr.strip()[:160]))


def _meeko_attempt(job: Job, allow_bad_res: bool) -> tuple[bool, str]:
    """meeko 한 번 실행. 성공하면 박스까지 쓴다."""
    out_base = str(job.out_pdbqt.with_suffix(""))
    cmd = ["mk_prepare_receptor.py", "-i", str(job.clean_pdb), "-o", out_base, "-p"]
    if allow_bad_res:
        cmd.append("-a")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not job.out_pdbqt.exists() or job.out_pdbqt.stat().st_size == 0:
        remove_outputs(job.out_pdbqt)
        return (False, (res.stderr or res.stdout or "").strip()[:160])
    # 1차 경로도 결과를 확인한다. 종료코드와 파일 크기만 보는 술어가 바로
    # 전하 없는 파일 1,699 개를 성공으로 통과시킨 그 술어다.
    fraction = nonzero_charge_fraction(job.out_pdbqt)
    if fraction < MIN_NONZERO_CHARGE_FRACTION:
        remove_outputs(job.out_pdbqt)
        return (False, f"meeko: 부분전하가 붙지 않았습니다 ({fraction:.1%})")
    write_box(
        job.out_box,
        job.pocket,
        prediction=job.pocket_prediction,
        clean_pdb=job.clean_pdb,
    )
    return (True, "meeko" + ("_allow_bad_res" if allow_bad_res else ""))


def run_meeko(job: Job) -> tuple[str, bool, str]:
    """수용체 PDBQT 하나. meeko 로 만들고, 안 되면 잔기를 버려서라도 meeko 로 만든다.

    예전에는 meeko 가 거절하면 곧장 OpenBabel 로 넘어갔다. 그 폴백은 화학적으로
    다른 수용체를 만든다 - 실측(A0A087WXS9):

        meeko   HD 553  N 414  NA   8     (수소결합 주개가 있다)
        obabel  HD   0  N  11  NA 414     (주개가 없고 질소가 전부 받개)

    HD 가 없으면 AutoDock 의 수소결합 항이 통째로 죽는다. 그렇게 만든 수용체를
    meeko 수용체 13,339 개와 같은 순위표에 넣으면, 점수가 체계적으로 얕게 나와
    실제 결합체가 상위에서 밀려난다. 빠진 것보다 나쁘다 - 빠진 것은 보이지만
    이것은 평범한 숫자로 보인다.

    그래서 폴백을 meeko 재시도로 바꾼다. `-a`(--allow_bad_res)는 원자가 빠진
    잔기를 지우고 진행하므로 원자 타이핑과 전하 모델이 나머지와 같다. 실측으로
    실패했던 표적이 살아난다(A0A087WXS9 HD 553, Q16827 HD 1746).
    """
    if not shutil.which("mk_prepare_receptor.py"):
        return (job.uniprot, False, "mk_prepare_receptor.py not on PATH")
    ok, detail = _meeko_attempt(job, allow_bad_res=False)
    if ok:
        return (job.uniprot, True, detail)
    strict_detail = detail
    # 엄격 모드가 거절한 구조는 원자가 빠진 잔기 때문인 경우가 대부분이다.
    ok, detail = _meeko_attempt(job, allow_bad_res=True)
    if ok:
        return (job.uniprot, True, detail)
    if ALLOW_OBABEL_FALLBACK:
        return _obabel_fallback(job)
    return (job.uniprot, False,
            f"meeko 엄격 실패({strict_detail}); -a 재시도도 실패({detail})")


def build_jobs(
    clean_dir: Path,
    pocket_dir: Path,
    pdbqt_dir: Path,
    box_dir: Path,
    skip_set: set[str],
) -> list[Job]:
    jobs: list[Job] = []
    for clean_pdb in sorted(clean_dir.glob("*_clean.pdb")):
        uid = clean_pdb.stem.replace("_clean", "")
        if uid in skip_set:
            continue
        pocket_json = pocket_dir / f"{uid}.pockets.json"
        if not pocket_json.exists():
            continue
        pocket = best_pocket(pocket_json)
        if pocket is None:
            continue
        pocket_prediction = pocket_dir / f"{uid}_clean.pdb_predictions.csv"
        if not pocket_prediction.exists():
            # No P2Rank prediction means no measurable extent; a default cube
            # here is what made every box a constant in the first place.
            continue
        out_pdbqt = pdbqt_dir / f"{uid}.pdbqt"
        out_box = box_dir / f"{uid}.box.txt"
        jobs.append(
            Job(
                uniprot=uid,
                clean_pdb=clean_pdb,
                pocket_json=pocket_json,
                pocket_prediction=pocket_prediction,
                pocket=pocket,
                out_pdbqt=out_pdbqt,
                out_box=out_box,
            )
        )
    return jobs


def build_manifest_records(
    jobs: list[Job],
    outcomes: dict[str, tuple[bool, str]],
) -> list[dict]:
    """One record per queued receptor: status, reason, and input hashes."""
    records: list[dict] = []
    for job in jobs:
        ok, detail = outcomes.get(
            job.uniprot,
            (False, "worker did not report a result"),
        )
        record = {
            "uniprot": job.uniprot,
            "status": "ok" if ok else "failed",
            "method": detail if ok else None,
            "reason": None if ok else detail,
            "clean_pdb": str(job.clean_pdb),
            "pocket_json": str(job.pocket_json),
            "clean_pdb_sha256": sha256_file(job.clean_pdb) or "",
            "pocket_json_sha256": sha256_file(job.pocket_json) or "",
        }
        if ok:
            record["output"] = job.out_pdbqt.name
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--pocket-dir", required=True, type=Path)
    parser.add_argument("--no-pocket-list", required=True, type=Path)
    parser.add_argument("--out-pdbqt-dir", required=True, type=Path)
    parser.add_argument("--out-box-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-success-fraction", type=float, default=0.8)
    parser.add_argument("--min-success-count", type=int, default=1)
    parser.add_argument(
        "--manifest-path", type=Path, default=None,
        help="수용체 준비 manifest 경로. 기본값은 out-pdbqt-dir 아래의 "
             f"{MANIFEST_FILENAME} 이며 verifier 와 readiness 가 같은 파일을 본다.",
    )
    parser.add_argument(
        "--restrict", type=Path, default=None,
        help="여기 적힌 UniProt 만 다시 만든다(한 줄에 하나). 나머지 수용체와 "
             "박스는 건드리지 않는다. 이 옵션이 없으면 큐에 든 모든 수용체를 "
             "먼저 지우므로, 일부만 고치려는 재실행이 나머지까지 날린다.",
    )
    parser.add_argument(
        "--allow-obabel-fallback", action="store_true",
        help="meeko 가 두 번 다 실패한 수용체를 OpenBabel 로 만든다. 그렇게 만든 "
             "수용체는 극성 수소가 없고 질소 타이핑이 달라 meeko 수용체와 같은 "
             "순위표에 넣으면 안 된다. 진단용으로만 쓴다.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out_pdbqt_dir.mkdir(parents=True, exist_ok=True)
    args.out_box_dir.mkdir(parents=True, exist_ok=True)

    global ALLOW_OBABEL_FALLBACK
    ALLOW_OBABEL_FALLBACK = bool(args.allow_obabel_fallback)
    if ALLOW_OBABEL_FALLBACK:
        LOG.warning(
            "OpenBabel 폴백을 켰습니다. 그렇게 만든 수용체는 극성 수소가 없어 "
            "meeko 수용체와 같은 순위표에서 비교할 수 없습니다"
        )

    skip_set: set[str] = set()
    if args.no_pocket_list.exists():
        skip_set = {ln.strip() for ln in args.no_pocket_list.read_text().splitlines() if ln.strip()}
    LOG.info("Skipping %d no-pocket receptors", len(skip_set))

    jobs = build_jobs(args.clean_dir, args.pocket_dir,
                      args.out_pdbqt_dir, args.out_box_dir, skip_set)
    if args.restrict is not None:
        if not args.restrict.exists():
            raise SystemExit(f"--restrict list is missing: {args.restrict}")
        wanted = {ln.strip() for ln in args.restrict.read_text().splitlines() if ln.strip()}
        if not wanted:
            raise SystemExit(f"--restrict list is empty: {args.restrict}")
        queued = {job.uniprot for job in jobs}
        missing = sorted(wanted - queued)
        if missing:
            raise SystemExit(
                f"--restrict names {len(missing)} receptor(s) that are not "
                f"queued (no clean PDB, no pocket, or skipped): {missing[:5]}"
            )
        jobs = [job for job in jobs if job.uniprot in wanted]
        LOG.info("Restricted to %d of %d receptors", len(jobs), len(queued))
    # 큐에 든 것만 지운다. --restrict 를 주면 나머지 수용체와 박스는 그대로 남는다.
    for job in jobs:
        remove_outputs(job.out_pdbqt, job.out_box)
    LOG.info("Meeko jobs queued: %d", len(jobs))
    if not jobs:
        raise SystemExit("No receptor PDBQT jobs were queued")

    n_ok = n_fail = 0
    outcomes: dict[str, tuple[bool, str]] = {}
    with mp.Pool(args.workers) as pool:
        for uid, ok, err in pool.imap_unordered(run_meeko, jobs, chunksize=8):
            outcomes[uid] = (ok, err)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                LOG.warning("FAIL %s: %s", uid, err)
            if (n_ok + n_fail) % 500 == 0:
                LOG.info("Progress: %d ok, %d fail", n_ok, n_fail)
    LOG.info("Done: %d ok, %d fail", n_ok, n_fail)
    success_fraction = n_ok / len(jobs)
    manifest_document = build_manifest(
        scope="restricted" if args.restrict is not None else "full",
        policy={
            "min_success_fraction": args.min_success_fraction,
            "min_success_count": args.min_success_count,
        },
        records=build_manifest_records(jobs, outcomes),
    )
    write_manifest(
        args.manifest_path or (args.out_pdbqt_dir / MANIFEST_FILENAME),
        manifest_document,
    )
    if n_ok < args.min_success_count or success_fraction < args.min_success_fraction:
        raise SystemExit(
            "Receptor PDBQT preparation did not meet quality gate: "
            f"{n_ok}/{len(jobs)} succeeded "
            f"({success_fraction:.3f}); required at least "
            f"{args.min_success_count} and fraction >= {args.min_success_fraction:.3f}"
        )


if __name__ == "__main__":
    main()
