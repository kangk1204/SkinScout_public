#!/usr/bin/env python3
"""stage8_crest.py — CREST free-ligand conformer ensemble for final targets."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage8.crest")


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_mmgbsa_report(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(f"MM-GBSA report is required and must be non-empty: {path}")
    try:
        mmgbsa = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"MM-GBSA report failed to parse: {path}: {exc}") from exc

    required_cols = {"target_id", "mmgbsa_dg_kcal_mol", "status"}
    missing_cols = sorted(required_cols - set(mmgbsa.columns))
    if missing_cols:
        raise SystemExit(f"MM-GBSA report missing required columns: {missing_cols}")
    if mmgbsa.empty:
        raise SystemExit(f"MM-GBSA report contains no rows: {path}")

    for col in ("target_id", "status"):
        normalized = mmgbsa[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ",".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"MM-GBSA report column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        mmgbsa[col] = normalized

    invalid_statuses = sorted(set(mmgbsa.loc[mmgbsa["status"] != "ok", "status"]))
    if invalid_statuses:
        shown = ",".join(invalid_statuses[:10])
        suffix = "..." if len(invalid_statuses) > 10 else ""
        raise SystemExit(
            "MM-GBSA report column 'status' contains invalid values: "
            f"{shown}{suffix}"
        )
    duplicate_ids = mmgbsa["target_id"][mmgbsa["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ",".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"MM-GBSA report contains duplicate target_id values: {shown}{suffix}"
        )

    bool_like_energy_indexes = [
        int(idx)
        for idx, value in mmgbsa["mmgbsa_dg_kcal_mol"].items()
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
            "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be numeric at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    energy = pd.to_numeric(mmgbsa["mmgbsa_dg_kcal_mol"], errors="coerce")
    invalid_energy_indexes = energy[energy.isna()].index.tolist()
    if invalid_energy_indexes:
        shown = ",".join(str(idx) for idx in invalid_energy_indexes[:10])
        suffix = "..." if len(invalid_energy_indexes) > 10 else ""
        raise SystemExit(
            "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be numeric at "
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
            "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be finite at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    nonnegative_energy_indexes = energy[energy >= 0].index.tolist()
    if nonnegative_energy_indexes:
        shown = ",".join(str(idx) for idx in nonnegative_energy_indexes[:10])
        suffix = "..." if len(nonnegative_energy_indexes) > 10 else ""
        raise SystemExit(
            "MM-GBSA report column 'mmgbsa_dg_kcal_mol' must be < 0 at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    mmgbsa["mmgbsa_dg_kcal_mol"] = energy
    return mmgbsa


def run_crest(
    ligand_xyz: Path, out_dir: Path, *, charge: int = 0, spin: int = 0
) -> Path | None:
    if not nonempty(ligand_xyz):
        return None
    if not shutil.which("crest"):
        return None
    cmd = [
        "crest", str(ligand_xyz), "-gfn2", "-T", "8", "-niceprint",
        "--chrg", str(charge), "--uhf", str(spin),
    ]
    res = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        return None
    conformers = out_dir / "crest_conformers.xyz"
    return conformers if nonempty(conformers) else None


def ligand_charge_spin(ligand_sdf: Path) -> tuple[int, int]:
    from rdkit import Chem

    try:
        supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    except OSError as exc:
        raise SystemExit(
            f"Could not read ligand SDF for charge/spin metadata: {ligand_sdf}"
        ) from exc
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        raise SystemExit(f"Could not read ligand SDF for charge/spin metadata: {ligand_sdf}")
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    spin = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if spin < 0:
        raise SystemExit(f"Ligand spin metadata must be >= 0: {ligand_sdf}")
    return int(charge), int(spin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmgbsa-report", required=True, type=Path)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help=(
            "Accepted for interface compatibility with the other stages, but "
            "this stage has no partial state: CREST runs once for the free "
            "ligand and every target shares that ensemble, so it either "
            "succeeds for all targets or for none."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_manifest.exists():
        args.out_manifest.unlink()

    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1")
    mmgbsa = read_mmgbsa_report(args.mmgbsa_report)
    if not shutil.which("obabel"):
        raise SystemExit("obabel is not available on PATH")
    if not shutil.which("crest"):
        raise SystemExit("crest is not available on PATH")
    mmgbsa = mmgbsa.sort_values("mmgbsa_dg_kcal_mol").head(args.top_n)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Convert SDF → XYZ once.
    ligand_xyz = args.out_dir / "ligand.xyz"
    obabel = subprocess.run(["obabel", str(args.ligand_sdf), "-O", str(ligand_xyz)],
                            capture_output=True)
    if obabel.returncode != 0 or not nonempty(ligand_xyz):
        raise SystemExit("Failed to convert ligand SDF to XYZ for CREST")
    charge, spin = ligand_charge_spin(args.ligand_sdf)

    # CREST 는 **자유 리간드**의 구조 앙상블을 찾는다. 표적이 무엇이든 같은
    # 분자이므로 답도 같다 - 매니페스트의 `conformer_scope` 가 이미 그렇게
    # 적고 있었다. 그런데 표적마다 한 번씩 돌리고 있었다. 아다팔렌 실측으로
    # 한 번에 2시간 20분이라, `top_n_for_qm: 3` 이면 같은 답을 얻는 데 7시간을
    # 쓴다. 한 번 돌리고 모든 표적이 그 앙상블을 가리킨다.
    targets = list(mmgbsa["target_id"].astype(str))
    shared_dir = args.out_dir / "free_ligand"
    shared_dir.mkdir(parents=True, exist_ok=True)
    conf = run_crest(ligand_xyz, shared_dir, charge=charge, spin=spin)
    if not conf:
        raise SystemExit("CREST produced no conformer ensembles")
    rows: list[list[str]] = [
        [uid, str(conf), "free_ligand", charge, spin, "ok"] for uid in targets
    ]
    LOG.info("CREST ran once for the free ligand; %d target(s) share %s",
             len(targets), conf)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.out_manifest.with_suffix(args.out_manifest.suffix + ".tmp")
    with tmp_manifest.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["target_id", "conformer_xyz", "conformer_scope", "charge", "spin", "status"])
        w.writerows(rows)
    tmp_manifest.replace(args.out_manifest)
    LOG.info("Wrote %s", args.out_manifest)


if __name__ == "__main__":
    main()
