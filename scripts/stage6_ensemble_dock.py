#!/usr/bin/env python3
"""stage6_ensemble_dock.py — Dock the ligand against each BioEmu cluster medoid
via GNINA + RTMScore, then consensus per receptor.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
from pathlib import Path

import pandas as pd

from stage3_gnina_rescore import gnina_score
from stage3_rtmscore import rtmscore

LOG = logging.getLogger("stage6.ensemble_dock")


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_bioemu_manifest(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(f"BioEmu medoid manifest is required and must be non-empty: {path}")
    try:
        manifest = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"BioEmu medoid manifest failed to parse: {path}: {exc}") from exc
    required = {"target_id", "medoid_pdb"}
    missing_cols = sorted(required - set(manifest.columns))
    if missing_cols:
        raise SystemExit(f"BioEmu medoid manifest missing required columns {missing_cols}")
    if manifest.empty:
        raise SystemExit("BioEmu medoid manifest contains no rows")
    for col in sorted(required):
        invalid = [
            int(idx) for idx, value in manifest[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"BioEmu medoid manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        manifest[col] = manifest[col].astype(str).str.strip()
    if "reconstruction_status" in manifest.columns:
        status = manifest["reconstruction_status"].fillna("").astype(str).str.strip()
        invalid = status[status != "pdbfixer_sidechains_openmm_relaxed"]
        if not invalid.empty:
            raise SystemExit(
                "BioEmu medoid manifest contains ineligible side-chain "
                f"reconstruction status values: {sorted(set(invalid))}"
            )
    resolved_medoids = manifest["medoid_pdb"].map(
        lambda value: str(Path(value).expanduser().resolve(strict=False))
    )
    duplicate_medoids = resolved_medoids[resolved_medoids.duplicated()].tolist()
    if duplicate_medoids:
        shown = ",".join(duplicate_medoids[:10])
        suffix = "..." if len(duplicate_medoids) > 10 else ""
        raise SystemExit(
            "BioEmu medoid manifest contains duplicate medoid_pdb values: "
            f"{shown}{suffix}"
        )
    return manifest


def positive_score(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (
            isinstance(value, str)
            and value.strip().lower() in {"true", "false"}
        )
    ):
        raise SystemExit(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"{label} must be finite")
    if parsed <= 0:
        raise SystemExit(f"{label} must be > 0")
    return parsed


def rank_normalize(values: list[float]) -> list[float]:
    """Return higher-is-better rank scores in (0, 1], preserving ties."""
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    sorted_unique = sorted(set(values), reverse=True)
    denom = max(len(sorted_unique) - 1, 1)
    ranks = {value: 1.0 - (rank / denom) for rank, value in enumerate(sorted_unique)}
    return [ranks[value] for value in values]


# 백본만 있는 수용체를 도킹에 쓰면 안 된다. RTMScore 는 포켓 그래프를 만들지 못해
# 멈추지만 GNINA 는 멈추지 않고 점수를 낸다 - 곁사슬이 없는 포켓은 실제보다 훨씬
# 열려 있어 그 점수가 다른 구조와 견줄 수 없는데도 화면에는 정상적인 숫자로 나온다.
# BioEmu 는 백본(N/CA/C/O)과 CB 만 내므로, 스테이지 6 이 곁사슬을 복원했는지를
# 여기서 확인한다.
SIDECHAIN_BEARING_RESIDUES = {
    "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "SER", "THR", "TRP", "TYR", "VAL",
}
BACKBONE_AND_CB = {"N", "CA", "C", "O", "CB", "OXT"}
MIN_SIDECHAIN_FRACTION = 0.5


def sidechain_fraction(pdb: Path) -> float:
    """곁사슬을 가진 잔기 중 실제로 곁사슬 원자가 있는 것의 비율."""
    atoms: dict[tuple[str, str], set[str]] = {}
    names: dict[tuple[str, str], str] = {}
    for line in pdb.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        key = (line[21:22], line[22:27].strip())
        names[key] = resname
        atoms.setdefault(key, set()).add(line[12:16].strip().upper())
    eligible = [k for k, n in names.items() if n in SIDECHAIN_BEARING_RESIDUES]
    if not eligible:
        return 1.0
    with_sidechain = sum(1 for k in eligible if atoms[k] - BACKBONE_AND_CB)
    return with_sidechain / len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-consensus", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_consensus.exists():
        args.out_consensus.unlink()

    manifest = read_bioemu_manifest(args.manifest)
    if not nonempty(args.ligand_sdf):
        raise SystemExit(f"Ligand SDF is missing or empty for ensemble docking: {args.ligand_sdf}")
    if not shutil.which("gnina"):
        raise SystemExit("gnina is not available on PATH")
    try:
        import rtmscore as rtmscore_module  # noqa: F401
    except ImportError as exc:
        raise SystemExit("rtmscore is not installed") from exc
    scored: list[dict[str, float | str]] = []
    for _, row in manifest.iterrows():
        uid = str(row["target_id"])
        medoid = Path(row["medoid_pdb"])
        if not nonempty(medoid):
            raise SystemExit(f"BioEmu medoid PDB is missing or empty for {uid}: {medoid}")
        fraction = sidechain_fraction(medoid)
        if fraction < MIN_SIDECHAIN_FRACTION:
            raise SystemExit(
                f"Receptor for {uid} has side chains on only {fraction:.0%} of the "
                f"residues that should have them: {medoid}. BioEmu emits backbone+CB "
                "only; Stage 6 must reconstruct side chains (pdbfixer) before docking. "
                "GNINA would score this without complaining and the number would not "
                "be comparable to anything."
            )
        g = gnina_score(medoid, args.ligand_sdf)
        r = rtmscore(medoid, args.ligand_sdf)
        if g is None or r is None:
            raise SystemExit(f"Ensemble scoring failed for {uid}: {medoid}")
        g = positive_score(g, f"GNINA score for {uid}")
        r = positive_score(r, f"RTMScore for {uid}")
        scored.append({"target_id": uid, "gnina": g, "rtm": r,
                       "medoid_pdb": str(medoid)})
    if not scored:
        raise SystemExit("Ensemble docking produced no scored targets")
    gnina_norm = rank_normalize([float(item["gnina"]) for item in scored])
    rtm_norm = rank_normalize([float(item["rtm"]) for item in scored])
    per_target: dict[str, list[dict[str, float | str]]] = {}
    for item, g_norm, r_norm in zip(scored, gnina_norm, rtm_norm, strict=True):
        item["gnina_rank_score"] = g_norm
        item["rtm_rank_score"] = r_norm
        item["consensus_score"] = (g_norm + r_norm) / 2.0
        per_target.setdefault(str(item["target_id"]), []).append(item)

    args.out_consensus.parent.mkdir(parents=True, exist_ok=True)
    tmp_consensus = args.out_consensus.with_suffix(args.out_consensus.suffix + ".tmp")
    with tmp_consensus.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "target_id",
            # 어느 medoid 가 이겼는지. 이 열이 없으면 "가장 좋은 점수"는 있는데
            # 그 점수를 낸 구조가 무엇인지 알 수 없고, 스테이지 7 이 MD 를 걸
            # 수용체를 고를 근거가 사라진다. 앙상블에서 표적당 여러 구조가
            # 나오는 것이 정상이므로, 고른 하나를 적어 두는 것이 이 단계의 일이다.
            "best_medoid_pdb",
            "best_gnina",
            "best_rtm",
            "gnina_rank_score",
            "rtm_rank_score",
            "consensus_score",
            "n_medoids_scored",
        ])
        for uid, vals in per_target.items():
            best = max(vals, key=lambda item: float(item["consensus_score"]))
            w.writerow([
                uid,
                str(best["medoid_pdb"]),
                f"{float(best['gnina']):.4f}",
                f"{float(best['rtm']):.4f}",
                f"{float(best['gnina_rank_score']):.4f}",
                f"{float(best['rtm_rank_score']):.4f}",
                f"{float(best['consensus_score']):.4f}",
                str(len(vals)),
            ])
    tmp_consensus.replace(args.out_consensus)
    LOG.info("Wrote %s", args.out_consensus)


if __name__ == "__main__":
    main()
