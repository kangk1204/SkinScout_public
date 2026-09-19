#!/usr/bin/env python3
"""stage7_mmgbsa.py — gmx_MMPBSA wrapper per target across all MD replicas.

The emitted number is the entropy-free ``DELTA TOTAL`` of an implicit-solvent
GB calculation (``gmx_MMPBSA`` entropy is off by default), evaluated at
0.15 M salt. It is not an entropy-corrected binding free energy; the report
carries ``mmgbsa_quantity``/``entropy_included``/``salt_concentration_molar``
so the value and the conditions it was computed under travel together.

Finite 0 and positive values are valid results. The producer never drops them;
downstream selection requires a negative value and records its reason.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from mmgbsa_policy import ENTROPY_INCLUDED, QUANTITY, SALT_CONCENTRATION_MOLAR

LOG = logging.getLogger("stage7.mmgbsa")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trajectory_index(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(
            f"Trajectory index is required and must be non-empty: {path}"
        )
    try:
        traj = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"Trajectory index failed to parse: {path}: {exc}") from exc

    required_cols = {"target_id", "trajectory_xtc", "status"}
    missing_cols = sorted(required_cols - set(traj.columns))
    if missing_cols:
        raise SystemExit(f"Trajectory index missing required columns: {missing_cols}")
    if traj.empty:
        raise SystemExit(f"Trajectory index contains no rows: {path}")

    for col in sorted(required_cols):
        normalized = traj[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ",".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"Trajectory index column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        traj[col] = normalized

    invalid_statuses = sorted(set(traj.loc[traj["status"] != "ok", "status"]))
    if invalid_statuses:
        shown = ",".join(invalid_statuses[:10])
        suffix = "..." if len(invalid_statuses) > 10 else ""
        raise SystemExit(
            "Trajectory index column 'status' contains invalid values: "
            f"{shown}{suffix}"
        )
    resolved_trajectories = traj["trajectory_xtc"].map(
        lambda value: str(Path(value).expanduser().resolve(strict=False))
    )
    duplicate_trajectories = resolved_trajectories[
        resolved_trajectories.duplicated()
    ].tolist()
    if duplicate_trajectories:
        shown = ",".join(duplicate_trajectories[:10])
        suffix = "..." if len(duplicate_trajectories) > 10 else ""
        raise SystemExit(
            "Trajectory index contains duplicate trajectory_xtc values: "
            f"{shown}{suffix}"
        )
    provenance_cols = {
        "replica", "trajectory_xtc_sha256", "tpr", "tpr_sha256",
        "topology_top", "topology_top_sha256",
    }
    missing_provenance = sorted(provenance_cols - set(traj.columns))
    if missing_provenance:
        raise SystemExit(
            f"Trajectory index missing required columns: {missing_provenance}"
        )
    for col in sorted(provenance_cols):
        normalized = traj[col].fillna("").astype(str).str.strip()
        blanks = normalized[normalized == ""].index.tolist()
        if blanks:
            raise SystemExit(
                f"Trajectory index column '{col}' contains blank values at "
                f"row index(es) {','.join(map(str, blanks[:10]))}: {path}"
            )
        traj[col] = normalized
    numeric_replicas = pd.to_numeric(traj["replica"], errors="coerce")
    invalid_replicas = traj.index[
        numeric_replicas.isna()
        | (numeric_replicas < 1)
        | (numeric_replicas % 1 != 0)
    ].tolist()
    if invalid_replicas:
        raise SystemExit(
            "Trajectory index column 'replica' must contain positive integers at "
            f"row index(es) {','.join(map(str, invalid_replicas[:10]))}: {path}"
        )
    traj["replica"] = numeric_replicas.astype(int)
    duplicate_replicas = traj.duplicated(subset=["target_id", "replica"])
    if duplicate_replicas.any():
        rows = traj.index[duplicate_replicas].tolist()
        raise SystemExit(
            "Trajectory index contains duplicate target_id/replica rows at "
            f"row index(es) {','.join(map(str, rows[:10]))}: {path}"
        )
    return traj


def validate_trajectory_files(traj: pd.DataFrame) -> pd.DataFrame:
    validated = traj.copy()
    for idx, row in validated.iterrows():
        target_id = str(row["target_id"])
        replica = int(row["replica"])
        paths = {
            "trajectory_xtc": Path(str(row["trajectory_xtc"])).expanduser().resolve(strict=False),
            "tpr": Path(str(row["tpr"])).expanduser().resolve(strict=False),
            "topology_top": Path(str(row["topology_top"])).expanduser().resolve(strict=False),
        }
        target_dir = paths["trajectory_xtc"].parent
        expected_names = {
            "trajectory_xtc": f"prod_r{replica}.xtc",
            "tpr": f"prod_r{replica}.tpr",
            "topology_top": "topol.top",
        }
        if target_dir.name != target_id:
            raise SystemExit(
                f"Trajectory index target_id/path mismatch at row {idx}: "
                f"{target_id} != {target_dir.name}"
            )
        for path_col, path in paths.items():
            if path.parent != target_dir or path.name != expected_names[path_col]:
                raise SystemExit(
                    f"Trajectory index {path_col} path mismatch for {target_id} "
                    f"replica {replica}: expected {target_dir / expected_names[path_col]}, "
                    f"got {path}"
                )
            if not nonempty(path):
                raise SystemExit(
                    f"Trajectory index {path_col} is missing or empty for "
                    f"{target_id} replica {replica}: {path}"
                )
            hash_col = f"{path_col}_sha256"
            expected_hash = str(row[hash_col]).strip().lower()
            if not SHA256_RE.fullmatch(expected_hash):
                raise SystemExit(
                    f"Trajectory index {hash_col} must be a SHA-256 digest for "
                    f"{target_id} replica {replica}"
                )
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise SystemExit(
                    f"Trajectory index {path_col} sha256 mismatch for {target_id} "
                    f"replica {replica}: expected {expected_hash}, got {actual_hash}"
                )
            validated.at[idx, path_col] = str(path)
            validated.at[idx, hash_col] = expected_hash
    for target_id, rows in validated.groupby("target_id"):
        if rows["topology_top"].nunique() != 1 or rows["topology_top_sha256"].nunique() != 1:
            raise SystemExit(
                f"Trajectory index has conflicting topology provenance for {target_id}"
            )
    return validated


def parse_delta_total(text: str) -> tuple[float, float | None] | None:
    """ΔTOTAL 줄에서 평균과 SEM 을 함께 읽는다.

    gmx_MMPBSA 의 열 순서는 `Average SD(Prop.) SD SEM(Prop.) SEM` 이다. 평균만
    꺼내 쓰면 그 값이 얼마나 흔들리는 값인지가 사라진다 - 실측으로 P11086 의
    프레임 간 SD 는 2.6 kcal/mol 이고, 세 자리까지 적힌 순위 차이(1.56)가 그
    흔들림보다 작다.
    """
    for line in text.splitlines():
        tokens = line.strip().replace("_", " ").split()
        if not tokens:
            continue
        upper = [token.upper().replace("Δ", "DELTA") for token in tokens]
        if len(upper) >= 3 and upper[0] == "DELTA" and upper[1] == "TOTAL":
            numeric_tokens = tokens[2:]
        elif len(upper) >= 2 and upper[0] in {"DELTATOTAL", "DELTA-TOTAL"}:
            numeric_tokens = tokens[1:]
        else:
            continue
        numbers: list[float] = []
        for token in numeric_tokens:
            try:
                numbers.append(float(token))
            except ValueError:
                continue
        if not numbers or not math.isfinite(numbers[0]):
            return None
        # 마지막 열이 SEM. 열이 다 나오지 않는 판본도 있으므로 없으면 None.
        sem = numbers[-1] if len(numbers) >= 5 and math.isfinite(numbers[-1]) else None
        return numbers[0], sem
    return None


def parse_delta_total_average(text: str) -> float | None:
    parsed = parse_delta_total(text)
    return None if parsed is None else parsed[0]


def write_mmpbsa_input(target_dir: Path, n_frames: int) -> Path:
    """gmx_MMPBSA 입력 파일. 아무도 만들지 않아서 이 단계가 돈 적이 없었다.

    형식을 직접 적지 않고 `gmx_MMPBSA --create_input gb` 가 내주는 표준 파일을
    받아 필요한 값만 바꾼다. 손으로 적은 네임리스트는 파서가 받아 주지 않았고
    (`Invalid input file! Error reading namelist`), 형식은 버전마다 달라진다.
    """
    path = target_dir / "mmpbsa.in"
    path.unlink(missing_ok=True)
    res = subprocess.run(
        ["gmx_MMPBSA", "--create_input", "gb"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=target_dir,
    )
    if res.returncode != 0 or not nonempty(path):
        raise SystemExit(
            "gmx_MMPBSA --create_input failed; cannot build an MM-GBSA input "
            f"file in {target_dir}: {(res.stderr or res.stdout)[-500:]}"
        )
    # 앞 20% 는 아직 평형에 있다. 최소 한 프레임은 남긴다.
    startframe = max(1, int(n_frames * 0.2))
    replacements = {
        "sys_name": '"SkinScout"',
        "startframe": str(startframe),
        # 생리 식염 농도. 기본값 0.0 은 이온이 없는 물이다. Stage 7 prep 의
        # genion 도 같은 농도로 배경 염을 넣으므로 샘플링과 평가 조건이 맞는다.
        "saltcon": f"{SALT_CONCENTRATION_MOLAR:.3f}",
    }
    lines = []
    for line in path.read_text().splitlines():
        key = line.split("=", 1)[0].strip()
        if key in replacements:
            comment = line.split("#", 1)
            tail = f"  # {comment[1].strip()}" if len(comment) > 1 else ""
            lines.append(f"  {key:<20} = {replacements[key]}{tail}")
        else:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_index_file(target_dir: Path, structure: Path, gmx_cmd: str = "gmx") -> Path:
    """수용체와 리간드 그룹을 담은 index.ndx.

    gmx_MMPBSA 는 `-cg <수용체그룹> <리간드그룹>` 으로 둘을 가리키는데, 그 번호가
    가리킬 index 파일 자체가 없었다. `gmx make_ndx` 의 기본 그룹에는 단백질과
    "기타"가 들어 있으므로 그것을 그대로 쓴다.
    """
    path = target_dir / "index.ndx"
    res = subprocess.run(
        [gmx_cmd, "make_ndx", "-f", str(structure), "-o", str(path)],
        input="q\n", capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=target_dir,
    )
    if res.returncode != 0 or not nonempty(path):
        raise SystemExit(
            f"gmx make_ndx could not build an index file for MM-GBSA: {path}: "
            f"{(res.stderr or res.stdout)[-500:]}"
        )
    return path


def index_group_numbers(index_ndx: Path, ligand_resname: str = "LIG") -> tuple[int, int]:
    """(수용체 그룹 번호, 리간드 그룹 번호).

    번호를 1 과 13 으로 박아 두면 계 구성이 조금만 달라져도 다른 원자들의
    상호작용 에너지를 계산하고, 그 값은 정상적인 숫자로 보고된다.
    """
    names = [
        line.strip().strip("[").strip("]").strip()
        for line in index_ndx.read_text(errors="replace").splitlines()
        if line.strip().startswith("[")
    ]
    lowered = [n.lower() for n in names]
    try:
        receptor = lowered.index("protein")
    except ValueError as exc:
        raise SystemExit(
            f"MM-GBSA index file has no Protein group: {index_ndx} ({names})"
        ) from exc
    for candidate in (ligand_resname.lower(), "other", "lig"):
        if candidate in lowered:
            return receptor, lowered.index(candidate)
    raise SystemExit(
        f"MM-GBSA index file has no ligand group: {index_ndx} ({names})"
    )


def run_mmgbsa(
    target_dir: Path,
    trajectories: list[Path],
    gmx_cmd: str = "gmx",
    *,
    structures: list[Path] | None = None,
    topology: Path | None = None,
    replicas: list[int] | None = None,
) -> dict | None:
    """한 표적의 MM-GBSA ΔG.

    예전에는 이 함수가 만들어지지도 않는 세 파일(`mmpbsa.in`, `index.ndx`,
    `prod_r1.tpr`)을 가리키고, 궤적 인자(`-ct`)와 토폴로지 인자(`-cp`)를 아예
    넘기지 않았다. 그리고 실패하면 사유 없이 None 을 돌려주어, 스테이지는
    "MM-GBSA produced no successful target energies" 한 줄만 남겼다.
    """
    for tool in ("gmx_MMPBSA", gmx_cmd):
        if not shutil.which(tool):
            LOG.error("%s is not on PATH; run this stage in the env built "
                      "from envs/md.yml", tool)
            return None
    structures = structures or [target_dir / "prod.tpr"] * len(trajectories)
    topology = topology or target_dir / "topol.top"
    replicas = replicas or list(range(1, len(trajectories) + 1))
    if not (len(trajectories) == len(structures) == len(replicas)):
        raise ValueError("trajectory, structure, and replica inputs must have equal lengths")
    for required in (*structures, topology):
        if not nonempty(required):
            LOG.error("MM-GBSA input is missing or empty: %s", required)
            return None
    usable = [
        (trajectory, structure, replica)
        for trajectory, structure, replica in zip(trajectories, structures, replicas)
        if nonempty(trajectory) and nonempty(structure)
    ]
    if not usable:
        LOG.error("MM-GBSA has no non-empty trajectory in %s", target_dir)
        return None

    n_frames = trajectory_frame_count(usable[0][0], usable[0][1], gmx_cmd)
    input_file = write_mmpbsa_input(target_dir, n_frames)
    index_ndx = write_index_file(target_dir, usable[0][1], gmx_cmd)
    receptor_group, ligand_group = index_group_numbers(index_ndx)

    # 복제는 **하나씩** 넘긴다. 여러 개를 `-ct` 에 함께 주면 gmx_MMPBSA 가 세
    # 성분을 모두 계산하고 21프레임을 다 처리한 뒤 결과를 쓰는 자리에서
    # "Some energy terms are undefined" 로 멈춘다. 같은 계·같은 설정으로 복제
    # 하나만 주면 통과한다(실측: P11086 단일 -19.53 kcal/mol, 두 개 실패).
    #
    # 복제를 따로 도는 편이 옳기도 하다. 복제는 독립 표본이므로 평균과 편차를
    # 낼 수 있는데, 궤적을 이어 붙이면 그 정보가 사라진다.
    values: list[float] = []
    sems: list[float] = []
    for trajectory, structure, replica in usable:
        outcome = _run_one_trajectory(
            target_dir, trajectory, replica, input_file, structure, topology,
            index_ndx, receptor_group, ligand_group,
        )
        if outcome is not None:
            values.append(outcome[0])
            if outcome[1] is not None:
                sems.append(outcome[1])
    if not values:
        return None
    spread = max(values) - min(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        LOG.info("%s: 복제 %d개 ΔG %s (폭 %.2f kcal/mol)", target_dir.name,
                 len(values), ", ".join(f"{v:.2f}" for v in values), spread)
    # 복제가 요청한 만큼 돌지 않았으면 조용히 넘어가지 않는다. 하나가 PBC 처리
    # 실패로 빠져도 평균은 평범한 숫자로 나오기 때문에, 읽는 쪽에서는 그것이
    # 복제 두 개의 평균인지 하나짜리 값인지 구분할 방법이 없다.
    if len(values) < len(trajectories):
        LOG.warning(
            "%s: 요청한 복제 %d개 중 %d개만 성공했다. 보고되는 ΔG 는 그 %d개의 "
            "평균이다", target_dir.name, len(trajectories), len(values), len(values),
        )
    # 불확실성은 복제 간 폭으로 잡는다. 프레임 SEM 은 쓰지 않는다.
    #
    # gmx_MMPBSA 가 내주는 SEM 은 한 궤적 안의 프레임을 독립 표본으로 세는데,
    # 10 ps 간격 프레임은 서로 강하게 상관돼 있어 그 N 이 허수다. 실측으로
    # 얼마나 어긋나는지 분명하다(5 ns, 402 프레임, 복제 2개씩):
    #
    #   P52788  프레임 SEM 0.13 / 0.15   복제 간 폭 2.30
    #   P11086  프레임 SEM 0.13 / 0.13   복제 간 폭 3.22
    #
    # 20배 가까이 작다. 복제가 하나뿐이면 불확실성을 아예 비워 둔다 - 0.13 을
    # 적으면 "±0.1 로 안다"고 읽히지만 실제로는 ±1.6 이다. 모르는 것을 작은
    # 숫자로 적는 것보다 비워 두는 편이 정직하다.
    frame_sem = max(sems) if sems else None
    uncertainty = spread / 2 if len(values) > 1 else None
    if uncertainty is None and frame_sem is not None:
        LOG.warning(
            "%s: 복제가 하나뿐이라 불확실성을 낼 수 없습니다. 프레임 SEM %.2f 는 "
            "상관된 프레임에서 나온 값이라 실제보다 20배 가까이 작습니다",
            target_dir.name, frame_sem,
        )
    return {
        "dg_kcal_mol": sum(values) / len(values),
        "uncertainty_kcal_mol": uncertainty,
        "spread_kcal_mol": spread if len(values) > 1 else None,
        "n_replicas_ok": len(values),
        "n_replicas_requested": len(trajectories),
        "n_frames": n_frames,
        "per_replica": values,
    }


def strip_pbc(trajectory: Path, structure: Path, out: Path,
              gmx_cmd: str = "gmx") -> Path | None:
    """분자를 주기 경계 너머로 이어 붙인 궤적을 만든다.

    gmx_MMPBSA 의 `-ct` 도움말이 명시한다: "Make sure the trajectory is fitted
    and pbc have been removed." 그렇지 않으면 상자를 가로지른 분자가 끊어진 채로
    들어가고, 결합 하나가 상자 폭만큼 길어진다.

    정육면체 상자에서는 우연히 드러나지 않았다. 십이면체로 바꾸자(부피를 29%
    줄이려고) 삼사정계 상자가 되면서 실측으로 74.6 A 짜리 백본 결합이 11개
    생겼고, sander 의 BOND 항이 9,480만 kcal/mol 로 폭발했다. 그 값은 출력
    서식(13자리)을 넘겨 `*************` 로 찍히고, gmx_MMPBSA 는 그것을 읽지
    못해 "Some energy terms are undefined" 로 멈춘다 - 원인을 짐작할 수 없는
    메시지다.

    `-pbc whole` 이 끊어진 분자를 잇고, `-center` 가 용질을 상자 가운데로 옮긴다.
    """
    # `-pbc mol` 은 분자를 통째로 유지하면서 각 분자의 중심을 상자 안에 넣고,
    # `-center` 로 단백질을 가운데 둔다. `-pbc whole` + `-center` 만 쓰면 단백질만
    # 옮겨져 리간드가 상자 반대편에 남는다 - 실측으로 리간드가 수용체에서 72.6 A
    # 떨어졌고, 결합 에너지가 전부 0.00 으로 나왔다(접촉이 없으니 당연하다).
    # 그 0.00 은 "결합하지 않는다"로 읽히지만 실제로는 계를 잘못 자른 것이다.
    tmp = out.with_suffix(".tmp.xtc")
    tmp.unlink(missing_ok=True)
    res = subprocess.run(
        [gmx_cmd, "trjconv", "-f", str(trajectory), "-s", str(structure),
         "-o", str(tmp), "-pbc", "mol", "-center", "-ur", "compact"],
        # 중심으로 삼을 그룹과 출력 그룹. Protein(1) 을 가운데 두고 System(0) 을 쓴다.
        input="1\n0\n", capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=trajectory.parent,
    )
    if res.returncode != 0 or not nonempty(tmp):
        LOG.error("gmx trjconv could not remove PBC from %s: %s",
                  trajectory, (res.stderr or res.stdout or "")[-400:])
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(out)
    return out


def _run_one_trajectory(target_dir: Path, trajectory: Path, replica: int,
                        input_file: Path, structure: Path, topology: Path,
                        index_ndx: Path, receptor_group: int,
                        ligand_group: int) -> tuple[float, float | None] | None:
    """복제 하나의 (ΔG, 프레임 SEM)."""
    whole = strip_pbc(
        trajectory, structure,
        trajectory.with_name(f"{trajectory.stem}_whole.xtc"),
    )
    if whole is None:
        return None
    cmd = ["gmx_MMPBSA", "-O",
           "-i", str(input_file),
           "-cs", str(structure),
           "-cp", str(topology),
           "-ci", str(index_ndx),
           "-cg", str(receptor_group), str(ligand_group),
           "-ct", str(whole),
           "-nogui"]
    final = target_dir / "FINAL_RESULTS_MMPBSA.dat"
    final.unlink(missing_ok=True)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=target_dir)
    # 종료코드만 보면 안 된다. gmx_MMPBSA 는 에너지를 다 계산해 결과 파일까지
    # 쓴 뒤, 끝에서 GUI 분석기(`gmx_MMPBSA_ana`)를 띄우려다 실패하면 0 이 아닌
    # 코드로 끝난다. 헤드리스 서버에서는 그것이 정상이고 계산과 무관하다.
    # 결과 파일이 있으면 계산은 된 것이다.
    combined_output = f"{res.stdout}\n{res.stderr}".lower()
    analyser_only_failure = (
        res.returncode != 0
        and nonempty(final)
        and ("gmx_mmpbsa_ana" in combined_output or "analyzer" in combined_output
             or "analyser" in combined_output)
    )
    if res.returncode != 0 and not analyser_only_failure:
        LOG.error("gmx_MMPBSA failed in %s (exit %d): %s",
                  target_dir, res.returncode,
                  (res.stderr or res.stdout or "")[-800:])
        return None
    if analyser_only_failure:
        LOG.info(
            "gmx_MMPBSA exited %d in %s but wrote %s; the failure is the GUI "
            "analyser, not the energies", res.returncode, target_dir, final.name,
        )
    if not nonempty(final):
        LOG.error("gmx_MMPBSA exited 0 but wrote no results file: %s", final)
        return None
    # 복제마다 결과를 남긴다. 하나만 두면 마지막 복제가 앞의 것을 덮어써서
    # 무엇을 평균했는지 뒤에 확인할 수 없다.
    keep = target_dir / f"FINAL_RESULTS_MMPBSA_r{replica}.dat"
    keep.write_text(final.read_text())
    parsed = parse_delta_total(final.read_text())
    if parsed is None:
        LOG.error("Could not read ΔTOTAL from %s", final)
    return parsed


def trajectory_frame_count(trajectory: Path, structure: Path,
                           gmx_cmd: str = "gmx") -> int:
    """궤적의 프레임 수. 버릴 앞부분을 정하는 데 쓴다."""
    res = subprocess.run(
        [gmx_cmd, "check", "-f", str(trajectory)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in (res.stderr or "").splitlines() + (res.stdout or "").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "Step":
            try:
                return int(fields[1])
            except ValueError:
                continue
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-index", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help="Write successful target rows even when some targets fail MM-GBSA.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_report.exists():
        args.out_report.unlink()

    traj = validate_trajectory_files(read_trajectory_index(args.trajectory_index))
    if not shutil.which("gmx_MMPBSA"):
        raise SystemExit("gmx_MMPBSA is not available on PATH")
    per_target = traj.groupby("target_id")
    n_ok = 0
    failed_targets: list[str] = []
    rows: list[list[str]] = []
    for uid, _ in per_target:
        traj_paths = traj[traj["target_id"] == uid].sort_values("replica")
        traj_paths = validate_trajectory_files(traj_paths)
        trajectories = [Path(p) for p in traj_paths["trajectory_xtc"].astype(str)]
        structures = [Path(p) for p in traj_paths["tpr"].astype(str)]
        topologies = [Path(p) for p in traj_paths["topology_top"].astype(str)]
        replicas = traj_paths["replica"].astype(int).tolist()
        if not trajectories:
            failed_targets.append(str(uid))
            continue
        target_dir = trajectories[0].parent
        result = run_mmgbsa(
            target_dir,
            trajectories,
            structures=structures,
            topology=topologies[0],
            replicas=replicas,
        )
        if result is None:
            failed_targets.append(str(uid))
            continue
        partial = result["n_replicas_ok"] < result["n_replicas_requested"]
        if partial and not args.allow_partial_output:
            # 복제 하나가 빠진 값을 온전한 값과 같은 자리에 적으면, 읽는 쪽은
            # 두 줄을 같은 근거로 비교하게 된다. 명시적으로 열어 주지 않는 한
            # 이 표적은 실패로 둔다.
            LOG.error("%s: 복제 %d/%d 만 성공했다. 부분 결과를 쓰려면 "
                      "--allow-partial-output 을 명시하라", uid,
                      result["n_replicas_ok"], result["n_replicas_requested"])
            failed_targets.append(str(uid))
            continue
        n_ok += 1
        uncertainty = result["uncertainty_kcal_mol"]
        spread = result["spread_kcal_mol"]
        # 세 자리는 이 표집이 뒷받침하지 못한다. 실측 프레임 간 SD 가
        # 1.9~2.8 kcal/mol 이라 소수 첫째 자리까지가 한계다.
        rows.append([
            uid,
            f"{result['dg_kcal_mol']:.1f}",
            "" if uncertainty is None else f"{uncertainty:.1f}",
            "" if spread is None else f"{spread:.2f}",
            str(result["n_replicas_ok"]),
            str(result["n_replicas_requested"]),
            str(result["n_frames"]),
            "ok_partial" if partial else "ok",
            QUANTITY,
            "yes" if ENTROPY_INCLUDED else "no",
            f"{SALT_CONCENTRATION_MOLAR:.3f}",
        ])
    if n_ok == 0:
        raise SystemExit("MM-GBSA produced no successful target energies")
    if failed_targets and not args.allow_partial_output:
        raise SystemExit(
            "MM-GBSA failed for final targets "
            f"{','.join(failed_targets)}; use --allow-partial-output only for "
            "explicit degraded diagnostics."
        )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = args.out_report.with_suffix(args.out_report.suffix + ".tmp")
    with tmp_report.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "target_id", "mmgbsa_dg_kcal_mol", "mmgbsa_uncertainty_kcal_mol",
            "mmgbsa_replica_spread_kcal_mol", "n_replicas_ok",
            "n_replicas_requested", "n_frames", "status",
            "mmgbsa_quantity", "entropy_included", "salt_concentration_molar",
        ])
        w.writerows(rows)
    tmp_report.replace(args.out_report)
    LOG.info("Wrote %s", args.out_report)


if __name__ == "__main__":
    main()
