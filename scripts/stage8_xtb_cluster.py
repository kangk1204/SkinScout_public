#!/usr/bin/env python3
"""stage8_xtb_cluster.py — xTB single-point over CREST free-ligand conformers."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage8.xtb_cluster")


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_crest_manifest(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(f"CREST manifest is required and must be non-empty: {path}")
    try:
        df = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"CREST manifest failed to parse: {path}: {exc}") from exc

    required = {"target_id", "conformer_xyz", "status"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise SystemExit(f"CREST manifest missing required columns {missing_cols}")
    if df.empty:
        raise SystemExit(f"CREST manifest contains no rows: {path}")

    for col in sorted(required):
        normalized = df[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ",".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"CREST manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        df[col] = normalized

    duplicate_ids = df["target_id"][df["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ",".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"CREST manifest contains duplicate target_id values: {shown}{suffix}"
        )
    resolved_conformers = df["conformer_xyz"].map(
        lambda value: str(Path(value).expanduser().resolve(strict=False))
    )
    # 이 함수는 바로 아래에서 `conformer_scope` 가 전부 "free_ligand" 이기를
    # **요구한다.** 자유 리간드 앙상블은 표적과 무관한 같은 분자의 것이므로,
    # 모든 행이 같은 파일을 가리키는 것이 옳은 상태다. 예전에는 표적마다 CREST
    # 를 한 번씩 돌려 경로가 달랐고, 이 검사는 그 시절의 것이다 - 중복을 금지
    # 하면서 동시에 같은 것이기를 요구하는 모순이었다.
    #
    # 그래서 검사를 뒤집는다: 자유 리간드 앙상블은 **하나여야** 한다. 표적마다
    # 다른 파일이 오면 그것이 이상한 상태다(같은 분자에 다른 답이 나온 것).
    distinct = sorted(set(resolved_conformers))
    if len(distinct) > 1:
        shown = ", ".join(distinct[:5])
        suffix = "..." if len(distinct) > 5 else ""
        raise SystemExit(
            "CREST manifest declares conformer_scope=free_ligand but points at "
            f"{len(distinct)} different ensembles: {shown}{suffix}. The free "
            "ligand is the same molecule for every target, so one ensemble is "
            "expected."
        )

    invalid_statuses = sorted(set(df.loc[df["status"] != "ok", "status"]))
    if invalid_statuses:
        shown = ",".join(invalid_statuses[:10])
        suffix = "..." if len(invalid_statuses) > 10 else ""
        raise SystemExit(
            "CREST manifest column 'status' contains invalid values: "
            f"{shown}{suffix}"
        )
    if "conformer_scope" in df.columns:
        scope = df["conformer_scope"].fillna("").astype(str).str.strip()
        invalid_scope = sorted(set(scope[scope != "free_ligand"]))
        if invalid_scope:
            shown = ",".join(invalid_scope[:10])
            suffix = "..." if len(invalid_scope) > 10 else ""
            raise SystemExit(
                "CREST manifest column 'conformer_scope' must be free_ligand; "
                f"got {shown}{suffix}"
            )
        df["conformer_scope"] = scope
    return df


def _int_value(value: object, label: str) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise SystemExit(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be an integer") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise SystemExit(f"{label} must be an integer")
    return parsed


def read_xyz_frames(path: Path) -> list[str]:
    """Return strict, normalized single-frame XYZ documents.

    CREST writes concatenated XYZ frames.  A partial or malformed frame must
    invalidate the ensemble; scoring a truncated molecule would attach the
    energy of a different chemical system to the ligand.
    """
    if not nonempty(path):
        raise SystemExit(f"CREST conformer XYZ is missing or empty: {path}")
    lines = path.read_text(errors="strict").splitlines()
    frames: list[str] = []
    offset = 0
    while offset < len(lines):
        if not lines[offset].strip():
            raise SystemExit(f"CREST conformer XYZ has a blank atom-count row: {path}")
        try:
            n_atoms = int(lines[offset].strip())
        except ValueError as exc:
            raise SystemExit(
                f"CREST conformer XYZ has an invalid atom count at line {offset + 1}: {path}"
            ) from exc
        if n_atoms < 1:
            raise SystemExit(f"CREST conformer XYZ atom count must be >= 1: {path}")
        end = offset + n_atoms + 2
        if end > len(lines):
            raise SystemExit(f"CREST conformer XYZ contains a truncated frame: {path}")
        atom_lines = lines[offset + 2:end]
        for line_no, line in enumerate(atom_lines, start=offset + 3):
            parts = line.split()
            if len(parts) != 4:
                raise SystemExit(
                    f"CREST conformer XYZ has a malformed atom row at line {line_no}: {path}"
                )
            try:
                coordinates = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise SystemExit(
                    f"CREST conformer XYZ has nonnumeric coordinates at line {line_no}: {path}"
                ) from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise SystemExit(
                    f"CREST conformer XYZ has nonfinite coordinates at line {line_no}: {path}"
                )
        frames.append("\n".join(lines[offset:end]) + "\n")
        offset = end
    return frames


def select_lowest_energy_conformer(
    ensemble_xyz: Path,
    out_dir: Path,
    gfn: str,
    *,
    charge: int,
    spin: int,
) -> tuple[Path, float]:
    frames = read_xyz_frames(ensemble_xyz)
    frame_dir = out_dir / "xtb_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    scored: list[tuple[float, int, Path]] = []
    for index, content in enumerate(frames):
        frame_path = frame_dir / f"frame_{index:06d}.xyz"
        frame_path.write_text(content)
        energy = xtb_singlepoint(frame_path, gfn, charge=charge, spin=spin)
        if energy is None:
            raise SystemExit(f"xTB single-point failed for conformer {index}: {frame_path}")
        scored.append((energy, index, frame_path))
    energy, _, selected = min(scored, key=lambda item: (item[0], item[1]))
    return selected, energy


def xtb_singlepoint(xyz: Path, gfn: str, *, charge: int, spin: int) -> float | None:
    """xTB 단일점 에너지.

    xtb 는 `charges`, `wbo`, `xtbrestart`, `xtbtopo.mol` 을 **현재 작업
    디렉터리에** 쓴다. cwd 를 주지 않으면 스크립트를 실행한 자리에 그것들이
    쌓이는데, 이 저장소에서는 그 자리가 저장소 루트였다.
    """
    if not nonempty(xyz):
        return None
    if not shutil.which("xtb"):
        LOG.error("xtb is not on PATH; run this stage in the env built from "
                  "envs/qm.yml")
        return None
    cmd = [
        "xtb",
        str(xyz),
        f"--{gfn}",
        "--sp",
        "--alpb",
        "water",
        "--chrg",
        str(charge),
    ]
    if spin > 0:
        cmd.extend(["--uhf", str(spin)])
    # 부산물은 입력 XYZ 옆에 남긴다. 진단할 때 그 자리에서 함께 보인다.
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=xyz.parent)
    if res.returncode != 0:
        LOG.error("xtb failed for %s (exit %d): %s", xyz, res.returncode,
                  (res.stderr or res.stdout or "")[-500:])
        return None
    for line in res.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            try:
                energy = float(line.split()[3])
            except (IndexError, ValueError):
                continue
            if not math.isfinite(energy):
                return None
            return energy
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crest-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gfn", default="gfn2")
    parser.add_argument("--out-cluster-manifest", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_cluster_manifest.exists():
        args.out_cluster_manifest.unlink()

    df = read_crest_manifest(args.crest_manifest)
    if not shutil.which("xtb"):
        raise SystemExit("xtb is not available on PATH")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    # 자유 리간드는 표적이 달라도 같은 분자다. 같은 파일에 같은 전하·스핀이면
    # xTB 결과도 같으므로 한 번만 계산하고 나눠 쓴다. CREST 와 같은 이유다.
    cache: dict[tuple[str, int, int], tuple[Path, float]] = {}
    for _, row in df.iterrows():
        uid = str(row["target_id"])
        conf_xyz = Path(row["conformer_xyz"]) if row["conformer_xyz"] else None
        if conf_xyz is None or not nonempty(conf_xyz):
            raise SystemExit(f"Missing or empty CREST conformer file for {uid}: {conf_xyz}")
        charge = (
            _int_value(row["charge"], f"xTB charge for {uid}")
            if "charge" in df.columns and not pd.isna(row["charge"])
            else 0
        )
        spin = (
            _int_value(row["spin"], f"xTB spin for {uid}")
            if "spin" in df.columns and not pd.isna(row["spin"])
            else 0
        )
        if spin < 0:
            raise SystemExit(f"xTB spin for {uid} must be >= 0")
        key = (str(conf_xyz.resolve()), charge, spin)
        if key in cache:
            selected_xyz, e = cache[key]
        else:
            try:
                selected_xyz, e = select_lowest_energy_conformer(
                    conf_xyz, args.out_dir, args.gfn, charge=charge, spin=spin
                )
            except SystemExit as exc:
                raise SystemExit(
                    f"xTB single-point failed for {uid}: {conf_xyz}: {exc}"
                ) from exc
            cache[key] = (selected_xyz, e)
        rows.append([uid, str(selected_xyz), f"{e:.8f}", charge, spin])
    if not rows:
        raise SystemExit("xTB clustering produced no scored clusters")
    args.out_cluster_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.out_cluster_manifest.with_suffix(
        args.out_cluster_manifest.suffix + ".tmp"
    )
    with tmp_manifest.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["target_id", "cluster_xyz", "xtb_energy_hartree", "charge", "spin"])
        w.writerows(rows)
    tmp_manifest.replace(args.out_cluster_manifest)
    LOG.info("Wrote %s", args.out_cluster_manifest)


if __name__ == "__main__":
    main()
