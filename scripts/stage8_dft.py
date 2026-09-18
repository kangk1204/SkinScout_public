#!/usr/bin/env python3
"""stage8_dft.py — DFT single-point via PySCF (+ GPU4PySCF) on selected
free-ligand cluster XYZ files for the final 1-3 candidates.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage8.dft")


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_cluster_manifest(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(
            f"xTB cluster manifest is required and must be non-empty: {path}"
        )
    try:
        df = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(
            f"xTB cluster manifest failed to parse: {path}: {exc}"
        ) from exc

    required = {"target_id", "cluster_xyz", "xtb_energy_hartree"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise SystemExit(f"xTB cluster manifest missing required columns {missing_cols}")
    if df.empty:
        raise SystemExit(f"xTB cluster manifest contains no rows: {path}")

    for col in ("target_id", "cluster_xyz"):
        normalized = df[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ",".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"xTB cluster manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        df[col] = normalized

    duplicate_ids = df["target_id"][df["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ",".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"xTB cluster manifest contains duplicate target_id values: {shown}{suffix}"
        )
    resolved_clusters = df["cluster_xyz"].map(
        lambda value: str(Path(value).expanduser().resolve(strict=False))
    )
    bool_like_energy_indexes = [
        int(idx)
        for idx, value in df["xtb_energy_hartree"].items()
        if (
            isinstance(value, bool)
            or type(value).__name__ == "bool_"
            or (
                isinstance(value, str)
                and value.strip().lower() in {"true", "false"}
            )
        )
    ]
    if bool_like_energy_indexes:
        shown = ",".join(str(idx) for idx in bool_like_energy_indexes[:10])
        suffix = "..." if len(bool_like_energy_indexes) > 10 else ""
        raise SystemExit(
            "xTB cluster manifest column 'xtb_energy_hartree' must be numeric at "
            f"row index(es) {shown}{suffix}: {path}"
        )

    energy = pd.to_numeric(df["xtb_energy_hartree"], errors="coerce")
    invalid_energy_indexes = energy[energy.isna()].index.tolist()
    if invalid_energy_indexes:
        shown = ",".join(str(idx) for idx in invalid_energy_indexes[:10])
        suffix = "..." if len(invalid_energy_indexes) > 10 else ""
        raise SystemExit(
            "xTB cluster manifest column 'xtb_energy_hartree' must be numeric at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    nonfinite_energy_indexes = [
        int(idx)
        for idx, value in energy.items()
        if not math.isfinite(float(value))
    ]
    if nonfinite_energy_indexes:
        shown = ",".join(str(idx) for idx in nonfinite_energy_indexes[:10])
        suffix = "..." if len(nonfinite_energy_indexes) > 10 else ""
        raise SystemExit(
            "xTB cluster manifest column 'xtb_energy_hartree' must be finite at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    df["xtb_energy_hartree"] = energy
    resolved = resolved_clusters.rename("_resolved_cluster")
    for cluster_path, indexes in resolved.groupby(resolved).groups.items():
        if len(indexes) < 2:
            continue
        energies = {float(df.loc[index, "xtb_energy_hartree"]) for index in indexes}
        if len(energies) != 1:
            raise SystemExit(
                "xTB cluster manifest contains duplicate cluster_xyz values "
                f"with inconsistent energies: {cluster_path}"
            )
    return df


def parse_xyz(xyz: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    if not nonempty(xyz):
        return [], []
    lines = xyz.read_text().splitlines()
    if len(lines) < 2:
        return [], []
    try:
        n = int(lines[0].strip())
    except ValueError:
        return [], []
    if n < 1 or len(lines) != n + 2:
        return [], []
    elems: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for line in lines[2:2 + n]:
        parts = line.split()
        if len(parts) != 4 or not parts[0].isalpha():
            return [], []
        try:
            point = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            return [], []
        if not all(math.isfinite(value) for value in point):
            return [], []
        elems.append(parts[0])
        coords.append(point)
    return elems, coords


def _int_value(value: object, label: str) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise SystemExit(f"{label} must be an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be an integer") from exc
    if not isinstance(value, int) and str(value).strip() != str(parsed):
        raise SystemExit(f"{label} must be an integer")
    return parsed


def run_dft(
    xyz: Path,
    functional: str,
    basis: str,
    *,
    charge: int,
    spin: int,
) -> tuple[float | None, str]:
    """Return (energy, backend). backend is "gpu4pyscf" or "pyscf_cpu".

    Which backend actually ran matters: the same functional/basis on CPU is
    orders of magnitude slower, so a run that silently fell back looks like a
    GPU run that was merely slow. The caller records the backend per target.
    """
    try:
        from pyscf import gto, dft
    except ImportError:
        LOG.warning("PySCF not installed; skipping DFT")
        return None, "unavailable"
    elems, coords = parse_xyz(xyz)
    if not elems:
        return None, "unavailable"
    atoms = [(e, c) for e, c in zip(elems, coords, strict=True)]
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin, unit="Angstrom")
    backend = "gpu4pyscf"
    try:
        if spin > 0:
            from gpu4pyscf.dft import uks as gpu_uks
            mf = gpu_uks.UKS(mol)
        else:
            from gpu4pyscf.dft import rks as gpu_rks
            mf = gpu_rks.RKS(mol)
    except (ImportError, OSError) as exc:
        # gpu4pyscf 는 CUDA 런타임 라이브러리가 없으면 실패하는데, 예외 종류가
        # **얼마나 멀리 갔느냐에 따라 달라진다.** 모듈을 못 찾으면 ImportError,
        # 모듈은 찾고 그 안에서 dlopen 이 실패하면 OSError 다. 실측으로
        # nvidia-*-cu12 를 설치했더니 libnvJitLink.so(ImportError)에서
        # libcusolver.so(OSError)로 옮겨 갔고, ImportError 만 잡던 코드가
        # 그 자리에서 죽었다.
        #
        # 조용히 내려앉으면 CPU 결과를 GPU 결과로 읽게 되므로 사유를 남긴다.
        LOG.warning("gpu4pyscf unavailable (%s: %s); falling back to CPU PySCF",
                    exc.__class__.__name__, exc)
        backend = "pyscf_cpu"
        mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
    mf.xc = functional
    mf.max_cycle = 200
    e = mf.kernel()
    if not mf.converged:
        return None, backend
    energy = float(e)
    if not math.isfinite(energy) or energy >= 0:
        return None, backend
    return energy, backend


def validate_functional(functional: str) -> None:
    """긴 계산을 시작하기 전에 범함수 이름이 받아들여지는지 본다.

    58원자 def2-TZVP 계산은 CPU 에서 시간 단위다. 이름이 틀렸다는 사실을
    그 뒤에 알게 되면 그 시간이 통째로 버려진다.
    """
    try:
        from pyscf.scf.dispersion import parse_dft
    except ImportError:
        return  # PySCF 가 없으면 아래에서 따로 걸린다
    try:
        parse_dft(functional.lower())
    except NotImplementedError as exc:
        raise SystemExit(
            f"PySCF does not support the functional '{functional}': {exc}. "
            "Set qm.dft_functional to a supported name (e.g. wB97X-D3BJ)."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    # PySCF 는 `wb97x-d3` 를 **블랙리스트**에 두고 있다(파라미터화가 없다).
    # 그 이름으로는 계산이 시작조차 못 하고 NotImplementedError 로 끝나는데,
    # 이 스테이지가 한 번도 돌지 않아 드러나지 않았다. 지원되는 대응은
    # `wb97x-d3bj` 로, PySCF 안에서 ωB97X-V + D3BJ 감쇠로 풀린다.
    parser.add_argument("--functional", default="wB97X-D3BJ")
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--spin", type=int, default=None)
    parser.add_argument(
        "--multiplicity",
        type=int,
        help="Optional spin multiplicity; when supplied, spin is multiplicity - 1.",
    )
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help="Write converged rows even when some final candidates fail DFT.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    validate_functional(args.functional)

    if args.out_report.exists():
        args.out_report.unlink()
    if args.multiplicity is not None:
        if args.multiplicity < 1:
            raise SystemExit("--multiplicity must be >= 1")
        args.spin = args.multiplicity - 1
    if args.spin is not None and args.spin < 0:
        raise SystemExit("--spin must be >= 0")

    df = read_cluster_manifest(args.cluster_manifest)
    try:
        import pyscf  # noqa: F401
    except ImportError as exc:
        raise SystemExit("pyscf is not installed") from exc

    n_ok = 0
    failed_targets: list[str] = []
    rows: list[list[str]] = []
    cache: dict[tuple[str, int, int], tuple[float | None, str]] = {}
    for _, row in df.iterrows():
        uid = str(row["target_id"])
        xyz = Path(row["cluster_xyz"]) if row["cluster_xyz"] else None
        if xyz is None or not nonempty(xyz):
            raise SystemExit(f"Missing or empty cluster XYZ for {uid}: {xyz}")
        charge = (
            _int_value(row["charge"], f"DFT charge for {uid}")
            if "charge" in df.columns and not pd.isna(row["charge"])
            else args.charge
        )
        spin = (
            _int_value(row["spin"], f"DFT spin for {uid}")
            if "spin" in df.columns and not pd.isna(row["spin"])
            else args.spin
        )
        if charge is None:
            raise SystemExit(f"DFT charge for {uid} must be provided by manifest column or --charge")
        if spin is None:
            raise SystemExit(f"DFT spin for {uid} must be provided by manifest column, --spin, or --multiplicity")
        if spin < 0:
            raise SystemExit(f"DFT spin for {uid} must be >= 0")
        key = (str(xyz.resolve()), charge, spin)
        if key not in cache:
            cache[key] = run_dft(
                xyz, args.functional, args.basis, charge=charge, spin=spin
            )
        e, backend = cache[key]
        if e is None:
            failed_targets.append(uid)
            continue
        n_ok += 1
        rows.append([
            uid,
            f"{e:.8f}",
            args.functional,
            args.basis,
            str(charge),
            str(spin),
            backend,
            "ok",
        ])
    if n_ok == 0:
        raise SystemExit("DFT produced no converged energies")
    if failed_targets and not args.allow_partial_output:
        raise SystemExit(
            "DFT failed for final candidates "
            f"{','.join(failed_targets)}; use --allow-partial-output only for "
            "explicit degraded diagnostics."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_report = args.out_report.with_suffix(args.out_report.suffix + ".tmp")
    with tmp_report.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "target_id",
            "dft_energy_hartree",
            "functional",
            "basis",
            "charge",
            "spin",
            "compute_backend",
            "status",
        ])
        w.writerows(rows)
    tmp_report.replace(args.out_report)
    LOG.info("Wrote %s", args.out_report)


if __name__ == "__main__":
    main()
