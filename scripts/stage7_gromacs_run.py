#!/usr/bin/env python3
"""stage7_gromacs_run.py — Production MD (gmx mdrun) for top candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage7.run")


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replica_tpr(target_dir: Path, base_tpr: Path, system_gro: Path,
                topology: Path, replica: int, seed: int,
                gmx_cmd: str) -> Path | None:
    """복제마다 **독립된** 시작 상태를 가진 tpr 을 만든다.

    예전에는 복제 전부가 같은 prod.tpr 을 쓰고 `mdrun -reseed <n>` 로 구분한다고
    여겼다. 그 플래그는 replica exchange 전용이라(`-replex` 없이는 무효) 아무 일도
    하지 않는다. prod.mdp 는 `gen_vel = no` 이고 `ld-seed` 를 적지 않았는데,
    ld-seed 의 기본값 -1 은 **grompp 시점에** 한 번 생성돼 tpr 에 박힌다. 그래서
    실측으로 두 복제가 같은 씨앗(-549562402)과 같은 시작 속도로 돌았다.

    궤적이 서로 다르게 나오기는 했다 - GPU 의 부동소수 축약 순서가 실행마다
    달라서다. 그것은 독립 표본이 아니라 수치 잡음이고, 그렇게 얻은 두 값의 폭을
    "복제 간 편차"로 보고하면 없는 불확실성을 보고하는 셈이다.

    복제마다 Maxwell-Boltzmann 분포에서 속도를 새로 뽑는다. NPT 가 정한 상자와
    밀도는 그대로 쓰고 속도만 다시 뽑는 것이 복제 MD 의 표준 관행이다.
    """
    # **평형화된** 좌표에서 출발해야 한다. 매니페스트의 `system_gro` 는 이온을
    # 넣은 직후의 구조라 아직 최소화·NVT·NPT 를 거치지 않았다. 거기에 300 K 속도를
    # 새로 얹으면 계가 터진다(실측: mdrun 이 SIGSEGV 로 죽었다).
    equilibrated = target_dir / "npt.gro"
    start = equilibrated if _nonempty(equilibrated) else system_gro
    if start is system_gro:
        LOG.warning("%s 가 없어 평형 전 좌표로 복제를 시작합니다: %s",
                    equilibrated.name, system_gro)
    mdp = target_dir / f"prod_r{replica}.mdp"
    base_mdp = target_dir / "prod.mdp"
    if not _nonempty(base_mdp):
        LOG.error("Cannot build a per-replica tpr without %s", base_mdp)
        return None
    body = []
    for line in base_mdp.read_text(errors="replace").splitlines():
        key = line.split("=", 1)[0].strip().replace("_", "-").lower()
        if key in {"gen-vel", "gen-seed", "gen-temp", "continuation", "ld-seed"}:
            continue
        body.append(line)
    body += [
        "; 복제별 독립 표본. 속도를 새로 뽑고 확률적 열욕의 씨앗도 따로 준다.",
        "gen_vel = yes",
        f"gen_seed = {seed}",
        "gen_temp = 300",
        # 속도를 새로 뽑으면 제약을 다시 걸어야 한다.
        "continuation = no",
        f"ld_seed = {seed}",
    ]
    mdp.write_text("\n".join(body) + "\n")

    out_tpr = target_dir / f"prod_r{replica}.tpr"
    res = subprocess.run(
        [gmx_cmd, "grompp", "-f", str(mdp), "-c", str(start),
         "-p", str(topology), "-o", str(out_tpr), "-maxwarn", "1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=target_dir,
    )
    if res.returncode != 0 or not _nonempty(out_tpr):
        LOG.error("grompp could not build a tpr for replica %d: %s",
                  replica, (res.stderr or res.stdout or "")[-500:])
        return None
    return out_tpr


def mdrun(
    target_dir: Path,
    tpr: Path,
    duration_ns: float,
    replica: int,
    seed: int,
    gmx_cmd: str,
) -> bool:
    xtc = target_dir / f"prod_r{replica}.xtc"
    if xtc.exists():
        xtc.unlink()
    nsteps = int(duration_ns * 500_000)  # 2 fs step
    cmd = [
        gmx_cmd, "mdrun",
        "-s", str(tpr),
        "-deffnm", f"{target_dir}/prod_r{replica}",
        "-nsteps", str(nsteps),
        # `-reseed` 는 replica exchange 전용이라 여기서는 아무 일도 하지 않는다.
        # 복제 간 독립성은 replica_tpr() 이 만드는 per-replica 시작 속도가 준다.
        "-nb", "gpu",
        "-pme", "gpu",
        "-pin", "on",
        "-pinoffset", "0",
        "-pinstride", "1",
    ]
    # 좌표 갱신과 결합항까지 GPU 에 두면 매 스텝의 GPU-CPU 왕복이 사라진다.
    # 실측(P52788, 89,184 원자): 7.1 → 18.0 → 68.6 ns/day. 이것을 켜지 않으면
    # 50 ns 를 여섯 번 도는 데 41일이 걸린다.
    #
    # 다만 계에 따라 GROMACS 가 GPU 갱신을 거부한다(가상 자리, 일부 압력 결합,
    # 자유에너지 섭동 등). 거부당하면 조용히 느려지는 대신 한 번 물러선다.
    accelerated = cmd + ["-update", "gpu", "-bonded", "gpu"]
    res = subprocess.run(accelerated, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=target_dir)
    if res.returncode != 0:
        message = (res.stderr or "") + (res.stdout or "")
        if "update" in message.lower() or "bonded" in message.lower():
            LOG.warning(
                "GPU 갱신을 쓸 수 없는 계입니다. CPU 갱신으로 물러섭니다 - "
                "훨씬 느립니다: %s", message.strip()[-300:],
            )
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", cwd=target_dir)
    if res.returncode != 0:
        # 사유를 삼키면 "0/2 복제 완료"만 남는다. 그 문장은 MD 가 실패했다는
        # 사실만 말하고 왜인지는 말하지 않아서, 진단이 처음부터 다시 시작된다.
        LOG.error(
            "gmx mdrun failed for replica %d (exit %d): %s",
            replica, res.returncode, (res.stderr or res.stdout or "")[-800:],
        )
        return False
    if not _nonempty(xtc):
        LOG.error(
            "gmx mdrun exited 0 for replica %d but wrote no trajectory at %s. "
            "stdout tail: %s", replica, xtc, (res.stdout or "")[-400:],
        )
        return False
    return True


def read_md_manifest(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"MD manifest is required and must be non-empty: {path}")
    try:
        manifest = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"MD manifest failed to parse: {path}: {exc}") from exc
    required = {"target_id", "system_gro", "topology_top", "tpr", "status"}
    missing_cols = sorted(required - set(manifest.columns))
    if missing_cols:
        raise SystemExit(f"MD manifest missing required columns {missing_cols}")
    if manifest.empty:
        raise SystemExit("MD manifest contains no rows")
    for col in sorted(required):
        invalid = [
            int(idx) for idx, value in manifest[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"MD manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        manifest[col] = manifest[col].astype(str).str.strip()
    invalid_status = sorted(set(manifest.loc[~manifest["status"].isin({"ready"}), "status"]))
    if invalid_status:
        shown = ", ".join(invalid_status[:10])
        suffix = "..." if len(invalid_status) > 10 else ""
        raise SystemExit(f"MD manifest column 'status' contains invalid values: {shown}{suffix}")
    duplicate_ids = manifest["target_id"][manifest["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(f"MD manifest contains duplicate target_id values: {shown}{suffix}")
    for col in ("system_gro", "topology_top", "tpr"):
        resolved_paths = manifest[col].map(
            lambda value: str(Path(value).expanduser().resolve(strict=False))
        )
        duplicate_paths = resolved_paths[resolved_paths.duplicated()].tolist()
        if duplicate_paths:
            shown = ", ".join(duplicate_paths[:10])
            suffix = "..." if len(duplicate_paths) > 10 else ""
            raise SystemExit(
                f"MD manifest contains duplicate {col} values: {shown}{suffix}"
            )
    return manifest


def validate_md_output_contract(manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_root = out_dir.expanduser().resolve(strict=False)
    expected_names = {
        "system_gro": "system.gro",
        "topology_top": "topol.top",
        "tpr": "prod.tpr",
    }
    validated = manifest.copy()
    for idx, row in validated.iterrows():
        target_id = str(row["target_id"])
        target_dir = (out_root / target_id).resolve(strict=False)
        if target_dir.parent != out_root:
            raise SystemExit(
                f"MD manifest target_id must name one direct child of --out-dir: "
                f"{target_id}"
            )
        for column, expected_name in expected_names.items():
            resolved = Path(str(row[column])).expanduser().resolve(strict=False)
            if resolved.parent != target_dir or resolved.name != expected_name:
                raise SystemExit(
                    f"MD manifest {column} for {target_id} must be "
                    f"{target_dir / expected_name}: {resolved}"
                )
            validated.at[idx, column] = str(resolved)
    return validated


def trajectory_paths(manifest: pd.DataFrame, replicas: int) -> list[Path]:
    return [
        Path(str(row["system_gro"])).parent / f"prod_r{replica}.xtc"
        for _, row in manifest.iterrows()
        for replica in range(1, replicas + 1)
    ]


def remove_trajectories(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def validate_ready_files(manifest: pd.DataFrame) -> None:
    for _, row in manifest.iterrows():
        uid = str(row["target_id"])
        system_gro = Path(row["system_gro"])
        topology_top = Path(row["topology_top"])
        tpr = Path(row["tpr"])
        if not _nonempty(system_gro) or not _nonempty(topology_top) or not _nonempty(tpr):
            raise SystemExit(
                f"Ready MD system files missing or empty for {uid}: "
                f"system_gro={system_gro} topology_top={topology_top} tpr={tpr}"
            )


def require_tool(command: str) -> str:
    if not shutil.which(command):
        raise SystemExit(f"gmx is not available on PATH: {command}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--duration-ns", type=float, default=50.0)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17391)
    parser.add_argument("--gmx-command", default="gmx")
    parser.add_argument("--out-index", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # An existing manifest belongs to the current preparation attempt, so its
    # downstream trajectory index cannot remain as a stale success artifact
    # when validation fails. If the manifest itself is absent, leave unrelated
    # prior output untouched until the upstream preparation gate is restored.
    if args.manifest.exists() and args.manifest.resolve() != args.out_index.resolve():
        args.out_index.unlink(missing_ok=True)

    if args.duration_ns <= 0:
        raise SystemExit(f"--duration-ns must be greater than 0: {args.duration_ns}")
    if args.replicas < 1:
        raise SystemExit(f"--replicas must be >= 1: {args.replicas}")
    if args.seed < 0:
        raise SystemExit(f"--seed must be >= 0: {args.seed}")

    manifest = validate_md_output_contract(
        read_md_manifest(args.manifest),
        args.out_dir,
    )
    validate_ready_files(manifest)
    gmx_cmd = require_tool(args.gmx_command)
    expected_trajectories = trajectory_paths(manifest, args.replicas)
    tmp_index = args.out_index.with_suffix(args.out_index.suffix + ".tmp")
    if args.out_index.exists():
        args.out_index.unlink()
    tmp_index.unlink(missing_ok=True)
    remove_trajectories(expected_trajectories)
    n_ok = 0
    rows: list[list[str | int]] = []
    try:
        for _, row in manifest.iterrows():
            uid = str(row["target_id"])
            system_gro = Path(row["system_gro"])
            tpr = Path(row["tpr"])
            target_dir = system_gro.parent
            for rep in range(1, args.replicas + 1):
                seed = args.seed + rep - 1
                # 복제마다 자기 tpr 을 만든다. 같은 tpr 을 나눠 쓰면 시작 속도와
                # 열욕 씨앗이 같아, 두 실행의 차이가 GPU 수치 잡음뿐이 된다.
                replica_input = replica_tpr(
                    target_dir, tpr, system_gro, Path(row["topology_top"]),
                    rep, seed, gmx_cmd,
                )
                if replica_input is None:
                    trajectory = target_dir / f"prod_r{rep}.xtc"
                    rows.append([
                        uid, rep, str(trajectory), "failed", "",
                        str(tpr), sha256_file(tpr), seed, args.duration_ns,
                        int(args.duration_ns * 500_000),
                    ])
                    continue
                ok = mdrun(target_dir, replica_input, args.duration_ns, rep, seed, gmx_cmd)
                trajectory = target_dir / f"prod_r{rep}.xtc"
                if ok:
                    n_ok += 1
                rows.append([
                    uid,
                    rep,
                    str(trajectory),
                    "ok" if ok else "failed",
                    sha256_file(trajectory) if ok else "",
                    str(replica_input),
                    sha256_file(replica_input),
                    seed,
                    args.duration_ns,
                    int(args.duration_ns * 500_000),
                ])
    except BaseException:
        remove_trajectories(expected_trajectories)
        raise
    expected = len(manifest) * args.replicas
    if n_ok != expected:
        remove_trajectories(expected_trajectories)
        raise SystemExit(f"GROMACS completed {n_ok}/{expected} requested replicas")
    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp_index.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow([
                "target_id",
                "replica",
                "trajectory_xtc",
                "status",
                "trajectory_xtc_sha256",
                "tpr",
                "tpr_sha256",
                "seed",
                "duration_ns",
                "nsteps",
            ])
            w.writerows(rows)
        tmp_index.replace(args.out_index)
    except BaseException:
        tmp_index.unlink(missing_ok=True)
        remove_trajectories(expected_trajectories)
        raise
    LOG.info("Wrote %s", args.out_index)


if __name__ == "__main__":
    main()
