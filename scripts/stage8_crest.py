#!/usr/bin/env python3
"""stage8_crest.py — CREST free-ligand conformer ensemble for final targets."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from mmgbsa_policy import (
    REASON_COLUMN,
    SELECTED_COLUMN,
    annotate_selection,
    select_favorable,
)

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
    # 0/positive values are valid computed results. They are simply not
    # selected for QM refinement, and that selection decision is recorded in a
    # separate column by `mmgbsa_policy`. Rejecting them here conflated
    # "the calculation converged" with "the candidate is favourable".
    mmgbsa["mmgbsa_dg_kcal_mol"] = energy
    return mmgbsa


def run_crest(
    ligand_xyz: Path, out_dir: Path, *, charge: int = 0, spin: int = 0
) -> Path | None:
    if not nonempty(ligand_xyz):
        return None
    if not shutil.which("crest"):
        return None
    conformers = out_dir / "crest_conformers.xyz"
    conformers.unlink(missing_ok=True)
    cmd = [
        "crest", str(ligand_xyz), "-gfn2", "-T", "8", "-niceprint",
        "--chrg", str(charge), "--uhf", str(spin),
    ]
    res = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        return None
    if not nonempty(conformers):
        return None
    try:
        input_signature = xyz_atom_signature(ligand_xyz, require_single=True)
        output_signatures = xyz_atom_signature(conformers, require_single=False)
    except ValueError as exc:
        LOG.error("CREST wrote malformed XYZ output: %s", exc)
        conformers.unlink(missing_ok=True)
        return None
    if any(signature != input_signature for signature in output_signatures):
        LOG.error("CREST output atom identity differs from its ligand input")
        conformers.unlink(missing_ok=True)
        return None
    return conformers


def xyz_atom_signature(path: Path, *, require_single: bool) -> tuple[str, ...] | list[tuple[str, ...]]:
    lines = path.read_text(errors="replace").splitlines()
    frames: list[tuple[str, ...]] = []
    cursor = 0
    while cursor < len(lines):
        try:
            atom_count = int(lines[cursor].strip())
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid atom count in {path}") from exc
        if atom_count < 1 or cursor + atom_count + 2 > len(lines):
            raise ValueError(f"truncated frame in {path}")
        symbols: list[str] = []
        for line in lines[cursor + 2:cursor + atom_count + 2]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"invalid atom row in {path}: {line!r}")
            try:
                coordinates = [float(value) for value in fields[1:4]]
            except ValueError as exc:
                raise ValueError(f"invalid coordinates in {path}: {line!r}") from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(f"non-finite coordinates in {path}: {line!r}")
            symbols.append(fields[0].upper())
        frames.append(tuple(symbols))
        cursor += atom_count + 2
    if not frames:
        raise ValueError(f"no XYZ frames in {path}")
    if require_single:
        if len(frames) != 1:
            raise ValueError(f"expected one XYZ frame in {path}, got {len(frames)}")
        return frames[0]
    return frames


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
    annotated = annotate_selection(mmgbsa)
    selected = select_favorable(mmgbsa, top_n=args.top_n)
    if selected.empty:
        # 계산 오류가 아니다. 모든 ΔTOTAL 이 유한하게 계산됐지만 그중 어느
        # 것도 유리(음수)하지 않아 QM 으로 진행할 후보가 없다는 선택 결과다.
        raise SystemExit(
            "No MM-GBSA target has a favorable (negative) delta TOTAL; "
            f"{len(mmgbsa)} value(s) were computed but 0 and positive results "
            "are not selected for QM refinement."
        )
    if not shutil.which("obabel"):
        raise SystemExit("obabel is not available on PATH")
    if not shutil.which("crest"):
        raise SystemExit("crest is not available on PATH")
    selected_ids = set(selected["target_id"].astype(str))
    skipped = annotated[~annotated["target_id"].astype(str).isin(selected_ids)]
    for _, row in skipped.iterrows():
        LOG.info(
            "MM-GBSA target %s: delta TOTAL %s kcal/mol is not selected (%s)",
            row["target_id"],
            row["mmgbsa_dg_kcal_mol"],
            row[REASON_COLUMN],
        )
    targets = list(selected["target_id"].astype(str))

    # CREST 는 **자유 리간드**의 구조 앙상블을 찾는다. 표적이 무엇이든 같은
    # 분자이므로 답도 같다 - 매니페스트의 `conformer_scope` 가 이미 그렇게
    # 적고 있었다. 그런데 표적마다 한 번씩 돌리고 있었다. 아다팔렌 실측으로
    # 한 번에 2시간 20분이라, `top_n_for_qm: 3` 이면 같은 답을 얻는 데 7시간을
    # 쓴다. 한 번 돌리고 모든 표적이 그 앙상블을 가리킨다.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = args.out_dir / "free_ligand"
    shared_dir.mkdir(parents=True, exist_ok=True)
    final_conf = shared_dir / "crest_conformers.xyz"
    final_conf.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix=".crest_attempt_", dir=args.out_dir) as attempt:
        attempt_dir = Path(attempt)
        ligand_xyz = attempt_dir / "ligand.xyz"
        obabel = subprocess.run(
            ["obabel", str(args.ligand_sdf), "-O", str(ligand_xyz)],
            capture_output=True,
        )
        if obabel.returncode != 0 or not nonempty(ligand_xyz):
            raise SystemExit("Failed to convert ligand SDF to XYZ for CREST")
        try:
            xyz_atom_signature(ligand_xyz, require_single=True)
        except ValueError as exc:
            raise SystemExit(f"Converted ligand XYZ is invalid for CREST: {exc}") from exc
        charge, spin = ligand_charge_spin(args.ligand_sdf)
        conf = run_crest(ligand_xyz, attempt_dir, charge=charge, spin=spin)
        if not conf:
            raise SystemExit("CREST produced no conformer ensembles")
        staged_conf = shared_dir / ".crest_conformers.xyz.tmp"
        staged_conf.unlink(missing_ok=True)
        try:
            shutil.copyfile(conf, staged_conf)
            staged_conf.replace(final_conf)
        finally:
            staged_conf.unlink(missing_ok=True)
    conf = final_conf
    ordered = annotated.sort_values(
        [SELECTED_COLUMN, "mmgbsa_dg_kcal_mol"],
        ascending=[False, True],
        kind="mergesort",
    )
    rows: list[list[str]] = [
        [
            str(row["target_id"]),
            str(conf),
            "free_ligand",
            str(charge),
            str(spin),
            "ok",
            f"{float(row['mmgbsa_dg_kcal_mol']):.1f}",
            "true" if bool(row[SELECTED_COLUMN]) else "false",
            str(row[REASON_COLUMN]),
        ]
        for _, row in ordered.iterrows()
    ]
    LOG.info("CREST ran once for the free ligand; %d of %d target(s) selected",
             len(targets), len(annotated))
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.out_manifest.with_suffix(args.out_manifest.suffix + ".tmp")
    with tmp_manifest.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "target_id", "conformer_xyz", "conformer_scope", "charge", "spin",
            "status", "mmgbsa_dg_kcal_mol", SELECTED_COLUMN, REASON_COLUMN,
        ])
        w.writerows(rows)
    tmp_manifest.replace(args.out_manifest)
    LOG.info("Wrote %s", args.out_manifest)


if __name__ == "__main__":
    main()
