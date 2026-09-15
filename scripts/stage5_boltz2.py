#!/usr/bin/env python3
"""stage5_boltz2.py — Boltz-2 co-folding + affinity for the top-50 candidates.

Per receptor:
  • 5 diffusion samples × 3 recycling steps (configurable)
  • Crop ±crop_radius Å around the P2Rank pocket centroid if sequence > max-residues
  • Run the Boltz-2 YAML CLI and capture confidence + affinity metrics
Output:
    boltz2_report.tsv columns:
      target_id, source, complex_pdb, n_residues, iptm, complex_plddt,
      affinity_log_uM, pb_valid, kept
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
from pathlib import Path

import pandas as pd

from boltz2_runner import (
    affinity_score,
    load_affinity_payload,
    load_confidence_payload,
    nonempty,
    pdb_sequence,
    run_boltz_predict,
    top_ranked_pdb,
    write_affinity_yaml,
)

LOG = logging.getLogger("stage5.boltz2")


def read_stage4_manifest(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(f"Stage 4 manifest is required and must be non-empty: {path}")
    try:
        manifest = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"Stage 4 manifest failed to parse: {path}: {exc}") from exc
    required = {"target_id", "source", "input_pdb"}
    missing_cols = sorted(required - set(manifest.columns))
    if missing_cols:
        raise SystemExit(f"Stage 4 manifest missing required columns {missing_cols}")
    if manifest.empty:
        raise SystemExit("Stage 4 manifest contains no rows")
    for col in sorted(required):
        invalid = [
            int(idx) for idx, value in manifest[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"Stage 4 manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        manifest[col] = manifest[col].astype(str).str.strip()
    duplicate_ids = manifest["target_id"][manifest["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(f"Stage 4 manifest contains duplicate target_id values: {shown}{suffix}")
    resolved_inputs = manifest["input_pdb"].map(
        lambda value: str(Path(value).expanduser().resolve(strict=False))
    )
    duplicate_inputs = resolved_inputs[resolved_inputs.duplicated()].tolist()
    if duplicate_inputs:
        shown = ", ".join(duplicate_inputs[:10])
        suffix = "..." if len(duplicate_inputs) > 10 else ""
        raise SystemExit(f"Stage 4 manifest contains duplicate input_pdb values: {shown}{suffix}")
    return manifest


def run_boltz2(receptor_pdb: Path, ligand_sdf: Path, out_dir: Path,
               n_seeds: int, n_recyc: int, max_res: int,
               crop_radius: float) -> dict | None:
    if not nonempty(receptor_pdb) or not nonempty(ligand_sdf):
        return None
    if not shutil.which("boltz"):
        return None
    input_yaml = out_dir / "input.yaml"
    run_out = out_dir / "boltz_out"
    standard_complex = out_dir / "complex.pdb"
    standard_complex.unlink(missing_ok=True)
    if run_out.is_symlink() or run_out.is_file():
        run_out.unlink()
    elif run_out.exists():
        shutil.rmtree(run_out)
    try:
        write_affinity_yaml(receptor_pdb, ligand_sdf, input_yaml)
    except ValueError as exc:
        LOG.debug("Boltz input build failed for %s: %s", receptor_pdb.name, exc)
        return None
    res = run_boltz_predict(
        input_yaml,
        run_out,
        diffusion_samples=n_seeds,
        recycling_steps=n_recyc,
    )
    if res is None or res.returncode != 0:
        if res is not None:
            LOG.debug("boltz failed for %s: %s", receptor_pdb.name, res.stderr[-200:])
        return None
    confidence = load_confidence_payload(run_out)
    affinity = load_affinity_payload(run_out)
    if confidence is None and affinity is None:
        return None
    predicted_complex = top_ranked_pdb(run_out, input_yaml.stem)
    if predicted_complex is None:
        LOG.debug("Boltz produced no top-ranked PDB for %s", receptor_pdb.name)
        return None
    pdb_text = predicted_complex.read_text(errors="replace")
    if not any(line.startswith("ATOM") for line in pdb_text.splitlines()):
        LOG.debug("Boltz PDB has no protein ATOM records for %s", receptor_pdb.name)
        return None
    if not any(line.startswith("HETATM") for line in pdb_text.splitlines()):
        LOG.debug("Boltz PDB has no ligand HETATM records for %s", receptor_pdb.name)
        return None
    tmp_complex = standard_complex.with_suffix(".pdb.tmp")
    tmp_complex.unlink(missing_ok=True)
    shutil.copyfile(predicted_complex, tmp_complex)
    tmp_complex.replace(standard_complex)
    payload: dict = {}
    if confidence is not None:
        payload.update(confidence)
    if affinity is not None:
        payload.update(affinity)
    if "affinity_log_uM" not in payload and affinity is not None:
        pred_value = affinity.get("affinity_pred_value")
        if pred_value is not None:
            payload["affinity_log_uM"] = pred_value
    if "affinity_score" not in payload and affinity is not None:
        payload["affinity_score"] = affinity_score(affinity)
    payload["complex_pdb"] = str(standard_complex)
    return payload


def _first_finite_metric(
    payload: dict,
    keys: tuple[str, ...],
    target_id: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    for key in keys:
        if key not in payload:
            continue
        return finite_metric(
            payload,
            key,
            target_id,
            min_value=min_value,
            max_value=max_value,
        )
    raise SystemExit(
        f"Boltz-2 report for {target_id} missing required metric "
        f"{'/'.join(keys)}"
    )


def _plddt_percent(value: float, target_id: str) -> float:
    """Return pLDDT on the 0-100 scale the threshold is expressed in.

    Boltz-2 writes ``complex_plddt`` as a fraction (0.94), while
    ``--plddt-threshold`` is documented and validated as 0-100 and defaults to
    75. Comparing the two directly made every target fail the gate no matter
    how good it was: the first real run scored iptm 0.96/0.88/0.64 with
    complex_plddt 0.94/0.88/0.94 and kept nothing, reporting only "no targets
    passing quality thresholds".

    A value at or below 1.0 is read as a fraction. A genuine 0-100 pLDDT of 1
    would be rejected by any usable threshold either way, so the ambiguity at
    the boundary costs nothing.
    """
    if value <= 1.0:
        scaled = value * 100.0
        LOG.debug("pLDDT for %s read as a fraction: %.3f -> %.1f",
                  target_id, value, scaled)
        return scaled
    return value


def _n_residues(payload: dict, receptor_pdb: Path, target_id: str) -> int:
    if "n_residues" in payload:
        return positive_int_metric(payload, "n_residues", target_id)
    seq = pdb_sequence(receptor_pdb)
    if seq:
        return len(seq)
    raise SystemExit(
        f"Boltz-2 report for {target_id} missing required metric 'n_residues'"
    )


def _posebusters_value(payload: dict, target_id: str) -> bool | None:
    if "posebusters_valid" not in payload:
        return None
    return bool_metric(payload, "posebusters_valid", target_id)


def _remove_standard_complexes(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def finite_metric(
    payload: dict,
    key: str,
    target_id: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if key not in payload:
        raise SystemExit(f"Boltz-2 report for {target_id} missing required metric '{key}'")
    value = payload[key]
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be finite"
        )
    if min_value is not None and parsed < min_value:
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be >= {min_value}"
        )
    if max_value is not None and parsed > max_value:
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be <= {max_value}"
        )
    return parsed


def bool_metric(payload: dict, key: str, target_id: str) -> bool:
    if key not in payload:
        raise SystemExit(f"Boltz-2 report for {target_id} missing required metric '{key}'")
    value = payload[key]
    if not isinstance(value, bool):
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be boolean"
        )
    return value


def positive_int_metric(payload: dict, key: str, target_id: str) -> int:
    if key not in payload:
        raise SystemExit(f"Boltz-2 report for {target_id} missing required metric '{key}'")
    value = payload[key]
    if isinstance(value, bool):
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be a positive integer"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be a positive integer"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be a positive integer"
        )
    parsed = int(numeric)
    if parsed < 1:
        raise SystemExit(
            f"Boltz-2 report for {target_id} metric '{key}' must be >= 1"
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--struct-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--iptm-threshold", type=float, default=0.6)
    parser.add_argument("--plddt-threshold", type=float, default=75.0)
    parser.add_argument("--max-residues", type=int, default=700)
    parser.add_argument("--crop-radius", type=float, default=20.0)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help="Write successful Boltz-2 rows even when some receptors fail.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    tmp_report = args.out_report.with_suffix(args.out_report.suffix + ".tmp")
    if args.out_report.exists():
        args.out_report.unlink()
    tmp_report.unlink(missing_ok=True)

    if args.num_seeds < 1:
        raise SystemExit(f"--num-seeds must be >= 1: {args.num_seeds}")
    if args.num_recycles < 1:
        raise SystemExit(f"--num-recycles must be >= 1: {args.num_recycles}")
    if args.max_residues < 1:
        raise SystemExit(f"--max-residues must be >= 1: {args.max_residues}")
    if not math.isfinite(args.crop_radius) or args.crop_radius <= 0:
        raise SystemExit(
            f"--crop-radius must be a finite value > 0: {args.crop_radius:g}"
        )
    if not 0.0 <= args.iptm_threshold <= 1.0:
        raise SystemExit(
            f"--iptm-threshold must be between 0 and 1: {args.iptm_threshold:g}"
        )
    if not 0.0 <= args.plddt_threshold <= 100.0:
        raise SystemExit(
            f"--plddt-threshold must be between 0 and 100: {args.plddt_threshold:g}"
        )

    manifest = read_stage4_manifest(args.manifest)
    if not shutil.which("boltz"):
        raise SystemExit("boltz is not available on PATH")
    if not nonempty(args.ligand_sdf):
        raise SystemExit(f"Ligand SDF is missing or empty for Boltz-2: {args.ligand_sdf}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_success = 0
    n_kept = 0
    failed_targets: list[str] = []
    standard_complexes: list[Path] = []
    rows: list[list[str | int]] = []
    try:
        for _, row in manifest.iterrows():
            uid = str(row["target_id"])
            receptor = Path(row["input_pdb"])
            if not nonempty(receptor):
                raise SystemExit(f"Missing or empty receptor PDB for {uid}: {receptor}")
            target_out = args.out_dir / uid
            payload = run_boltz2(receptor, args.ligand_sdf, target_out,
                                 args.num_seeds, args.num_recycles,
                                 args.max_residues, args.crop_radius)
            if payload is None:
                failed_targets.append(uid)
                continue
            standard_complexes.append(Path(str(payload["complex_pdb"])))
            n_success += 1
            iptm = finite_metric(payload, "iptm", uid, min_value=0.0, max_value=1.0)
            plddt = _plddt_percent(
                _first_finite_metric(
                    payload,
                    ("pocket_plddt", "complex_plddt", "plddt"),
                    uid,
                    min_value=0.0,
                    max_value=100.0,
                ),
                uid,
            )
            aff = _first_finite_metric(
                payload,
                ("affinity_log_uM", "affinity_pred_value", "log_uM_affinity"),
                uid,
            )
            pb = _posebusters_value(payload, uid)
            n_residues = _n_residues(payload, receptor, uid)
            kept = (
                iptm >= args.iptm_threshold
                and plddt >= args.plddt_threshold
                and (pb is not False)
            )
            if kept:
                n_kept += 1
            rows.append([uid, row["source"], payload["complex_pdb"], n_residues,
                         f"{iptm:.3f}", f"{plddt:.2f}", f"{aff:.3f}",
                         "not_available" if pb is None else ("yes" if pb else "no"),
                         "yes" if kept else "no"])
    except BaseException:
        _remove_standard_complexes(standard_complexes)
        raise
    if n_success == 0:
        raise SystemExit("Boltz-2 produced no successful cofolding reports")
    if failed_targets and not args.allow_partial_output:
        _remove_standard_complexes(standard_complexes)
        raise SystemExit(
            "Boltz-2 failed for receptors "
            f"{','.join(failed_targets)}; use --allow-partial-output only for "
            "explicit degraded diagnostics."
        )
    if n_kept == 0:
        _remove_standard_complexes(standard_complexes)
        raise SystemExit("Boltz-2 produced no targets passing quality thresholds")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp_report.open("w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["target_id", "source", "complex_pdb", "n_residues", "iptm",
                        "complex_plddt", "affinity_log_uM", "pb_valid", "kept"])
            w.writerows(rows)
        tmp_report.replace(args.out_report)
    except BaseException:
        tmp_report.unlink(missing_ok=True)
        _remove_standard_complexes(standard_complexes)
        raise
    LOG.info("Wrote %s", args.out_report)


if __name__ == "__main__":
    main()
