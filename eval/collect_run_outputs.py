#!/usr/bin/env python3
"""Collect SkinScout run outputs into the evaluation harness layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd

RANK_COLUMNS = ("final_rank", "rank", "rank_skin", "target_rank", "global_rank")
SCORE_COLUMNS = (
    "final_skin_weighted",
    "final_score",
    "skin_weighted_score",
    "rrf_score",
    "score",
    "docking_rrf",
    "psichic_score",
    "pred",
    "predicted_affinity",
)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(target)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copied_artifact(source: Path, target: Path) -> dict[str, object]:
    size = target.stat().st_size
    if size == 0:
        raise SystemExit(f"Copied evaluation artifact is empty: {target}")
    source_size = source.stat().st_size
    if source_size == 0:
        raise SystemExit(f"Source evaluation artifact is empty: {source}")
    return {
        "path": str(target),
        "source_path": str(source),
        "source_bytes": source_size,
        "source_sha256": _sha256(source),
        "bytes": size,
        "sha256": _sha256(target),
    }


def _unlink_if_file(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def _clear_generated_outputs(
    out_manifest: Path,
    out_dir: Path,
    rankings_dir: Path,
) -> None:
    _unlink_if_file(out_manifest)
    _unlink_if_file(out_dir / "iteration_manifest.json")
    _unlink_if_file(out_dir / "eval_targets.csv")
    for path in rankings_dir.glob("cold_start__*.csv"):
        _unlink_if_file(path)
    retro_dir = rankings_dir / "cosmetic_retro"
    for pattern in (
        "*__ranked_targets_v3.csv",
        "*__ranked_targets_v3_with_efficacy.csv",
    ):
        for path in retro_dir.glob(pattern):
            _unlink_if_file(path)


def _read_required_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _read_required_csv(
    path: Path,
    label: str,
    required_cols: set[str],
    numeric_cols: set[str] | None = None,
    fraction_cols: set[str] | None = None,
) -> pd.DataFrame:
    return _read_required_table(
        path,
        label,
        required_cols,
        sep=",",
        numeric_cols=numeric_cols,
        fraction_cols=fraction_cols,
    )


def _read_required_table(
    path: Path,
    label: str,
    required_cols: set[str],
    sep: str,
    numeric_cols: set[str] | None = None,
    fraction_cols: set[str] | None = None,
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as exc:
        raise SystemExit(f"{label} failed to parse: {path}: {exc}") from exc
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns {missing}: {path}")
    if df.empty:
        raise SystemExit(f"{label} contains no rows: {path}")
    for column in sorted(required_cols):
        invalid = [
            int(idx)
            for idx, value in df[column].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"{label} column '{column}' contains blank values at row "
                f"index(es) {shown}{suffix}: {path}"
            )
        df[column] = df[column].astype(str).str.strip()
    if "target_id" in required_cols:
        duplicate_ids = df["target_id"][df["target_id"].duplicated()].tolist()
        if duplicate_ids:
            shown = ", ".join(duplicate_ids[:10])
            suffix = "..." if len(duplicate_ids) > 10 else ""
            raise SystemExit(
                f"{label} contains duplicate target_id values: {shown}{suffix}: {path}"
            )
    for column in sorted((numeric_cols or set()) & set(df.columns)):
        bool_like = [
            int(idx)
            for idx, value in df[column].items()
            if _is_bool_like(value)
        ]
        if bool_like:
            shown = ", ".join(str(idx) for idx in bool_like[:10])
            suffix = "..." if len(bool_like) > 10 else ""
            raise SystemExit(
                f"{label} column '{column}' must be numeric at row "
                f"index(es) {shown}{suffix}: {path}"
            )
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = [int(idx) for idx, value in numeric.items() if pd.isna(value)]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"{label} column '{column}' must be numeric at row "
                f"index(es) {shown}{suffix}: {path}"
            )
        nonfinite = [
            int(idx)
            for idx, value in numeric.items()
            if not math.isfinite(float(value))
        ]
        if nonfinite:
            shown = ", ".join(str(idx) for idx in nonfinite[:10])
            suffix = "..." if len(nonfinite) > 10 else ""
            raise SystemExit(
                f"{label} column '{column}' must be finite at row "
                f"index(es) {shown}{suffix}: {path}"
            )
        if column in (fraction_cols or set()):
            out_of_range = [
                int(idx)
                for idx, value in numeric.items()
                if float(value) < 0 or float(value) > 1
            ]
            if out_of_range:
                shown = ", ".join(str(idx) for idx in out_of_range[:10])
                suffix = "..." if len(out_of_range) > 10 else ""
                raise SystemExit(
                    f"{label} column '{column}' must be in [0, 1] at row "
                    f"index(es) {shown}{suffix}: {path}"
                )
        df[column] = numeric
    return df


def _is_bool_like(value: object) -> bool:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "false"}
    return False


def _case_name(run_dir: Path) -> str:
    return run_dir.name.replace(" ", "_")


def _require_valid_smiles(smiles: str, label: str) -> None:
    from rdkit import Chem

    if Chem.MolFromSmiles(smiles) is None:
        raise SystemExit(f"{label} contains invalid canonical_smiles: {smiles}")


def _require_efficacy_columns(df: pd.DataFrame, path: Path) -> None:
    efficacy_cols = [col for col in df.columns if col.startswith("efficacy_top")]
    if not efficacy_cols:
        raise SystemExit(
            "Skin-weighted target ranking with efficacy missing efficacy_top* "
            f"columns required for KG recovery: {path}"
        )
    top_rows = df.head(10)
    rows_without_evidence = [
        int(idx)
        for idx, row in top_rows.iterrows()
        if not any(
            not pd.isna(row[col]) and str(row[col]).strip()
            for col in efficacy_cols
        )
    ]
    if rows_without_evidence:
        shown = ", ".join(str(idx) for idx in rows_without_evidence[:10])
        raise SystemExit(
            "Skin-weighted target ranking with efficacy top-10 rows must each "
            "contain at least one non-empty efficacy_top* value required for "
            f"KG recovery; missing row index(es) {shown}: {path}"
        )


def _ordered_ranking(df: pd.DataFrame, label: str, path: Path) -> pd.DataFrame:
    for column in RANK_COLUMNS:
        if column not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[column].dtype) or df[column].map(
            lambda value: isinstance(value, bool)
        ).any():
            raise SystemExit(
                f"{label} column '{column}' must not contain boolean values: {path}"
            )
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | ~values.map(math.isfinite)
        if invalid.any() or ((values % 1) != 0).any() or (values < 1).any():
            raise SystemExit(
                f"{label} column '{column}' must contain finite positive integer ranks: {path}"
            )
        if values.duplicated().any():
            raise SystemExit(f"{label} column '{column}' contains duplicate ranks: {path}")
        ordered = df.copy()
        ordered[column] = values.astype(int)
        return ordered.sort_values(
            [column, "target_id"], ascending=[True, True], kind="mergesort"
        ).reset_index(drop=True)
    for column in SCORE_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise SystemExit(f"{label} column '{column}' must be finite: {path}")
        ordered = df.copy()
        ordered[column] = values
        return ordered.sort_values(
            [column, "target_id"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
    raise SystemExit(
        f"{label} missing ranking order column; expected one of {list(RANK_COLUMNS)} "
        f"or {list(SCORE_COLUMNS)}: {path}"
    )


def _source_labels(value: object, label: str, path: Path, row_idx: int) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        raise SystemExit(
            f"{label} column 'sources' contains blank values at row "
            f"index(es) {row_idx}: {path}"
        )
    labels = [part.strip() for part in str(value).split(";")]
    if any(source == "" for source in labels):
        raise SystemExit(
            f"{label} column 'sources' contains empty source labels at row "
            f"index(es) {row_idx}: {path}"
        )
    duplicates = sorted({source for source in labels if labels.count(source) > 1})
    if duplicates:
        shown = ", ".join(duplicates[:10])
        raise SystemExit(
            f"{label} column 'sources' contains duplicate labels at row "
            f"index(es) {row_idx}: {shown}: {path}"
        )
    return labels


def _require_source_support(
    df: pd.DataFrame,
    label: str,
    path: Path,
    min_source_count: int,
) -> None:
    missing = sorted({"source_count", "sources"} - set(df.columns))
    if missing:
        raise SystemExit(
            f"{label} missing required scorer-rationale columns {missing}: {path}"
        )
    for idx, row in df.iterrows():
        if _is_bool_like(row["source_count"]):
            raise SystemExit(
                f"{label} column 'source_count' must contain integer values at "
                f"row index(es) {int(idx)}: {path}"
            )
        try:
            source_count = float(row["source_count"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{label} column 'source_count' must contain integer values at "
                f"row index(es) {int(idx)}: {path}"
            ) from exc
        if not math.isfinite(source_count) or not source_count.is_integer():
            raise SystemExit(
                f"{label} column 'source_count' must contain integer values at "
                f"row index(es) {int(idx)}: {path}"
            )
        source_count_int = int(source_count)
        if source_count_int < min_source_count:
            raise SystemExit(
                f"{label} column 'source_count' must be >= {min_source_count} "
                f"at row index(es) {int(idx)}: {path}"
            )
        labels = _source_labels(row["sources"], label, path, int(idx))
        if source_count_int != len(labels):
            raise SystemExit(
                f"{label} source_count={source_count_int} but sources lists "
                f"{len(labels)} label(s) at row index {int(idx)}: {path}"
            )


def _require_target_subset(
    child_df: pd.DataFrame,
    child_label: str,
    parent_df: pd.DataFrame,
    parent_label: str,
    path: Path,
) -> None:
    child_targets = set(child_df["target_id"].astype(str))
    parent_targets = set(parent_df["target_id"].astype(str))
    extra = sorted(child_targets - parent_targets)
    if extra:
        shown = ", ".join(extra[:10])
        suffix = "..." if len(extra) > 10 else ""
        raise SystemExit(
            f"{child_label} contains target_id values absent from "
            f"{parent_label}: {shown}{suffix}: {path}"
        )


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _load_sequence_map(fasta: Path | None) -> dict[str, str]:
    if fasta is None:
        return {}
    if not fasta.is_file() or fasta.stat().st_size == 0:
        raise SystemExit(
            f"Explicit sequence FASTA is required/diagnostics requested but missing or empty: {fasta}"
        )
    out: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def flush_current() -> None:
        if current is None:
            return
        sequence = "".join(chunks)
        if not sequence:
            raise SystemExit(
                f"Sequence FASTA entry '{current}' has no sequence: {fasta}"
            )
        if current in out:
            raise SystemExit(
                f"Sequence FASTA contains duplicate entry '{current}': {fasta}"
            )
        out[current] = sequence

    for line in fasta.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush_current()
            parts = line[1:].split()
            if not parts or not parts[0].strip():
                raise SystemExit(f"Sequence FASTA contains a blank header: {fasta}")
            current = parts[0].strip()
            chunks = []
        else:
            if current is None:
                raise SystemExit(
                    f"Sequence FASTA contains sequence before first header: {fasta}"
                )
            chunks.append(line)
    flush_current()
    return out


def _reject_ambiguous_cold_start_outputs(run_dirs: list[Path]) -> None:
    for mode, rel_paths in (
        (
            "comprehensive",
            [Path("03_targets") / "mode_comprehensive" / "top50_4way_consensus.csv"],
        ),
        ("fast", [Path("03_targets") / "mode_fast" / "top50.csv"]),
        (
            "dti_only",
            [
                Path("03_targets") / "mode_comprehensive" / "psichic_sanity.tsv",
                Path("03_targets") / "mode_fast" / "psichic_proteome.tsv",
            ],
        ),
    ):
        producers = [
            rd
            for rd in run_dirs
            if any((rd / rel_path).exists() for rel_path in rel_paths)
        ]
        if len(producers) > 1:
            preview = ", ".join(str(p) for p in producers[:5])
            raise SystemExit(
                f"Multiple run dirs contain {mode} cold-start rankings, "
                "which would overwrite the single evaluator input. "
                f"Collect one run at a time or aggregate explicitly: {preview}"
            )


def _reject_duplicate_case_ids(run_dirs: list[Path]) -> None:
    seen: dict[str, Path] = {}
    duplicates: list[str] = []
    for run_dir in run_dirs:
        case = _case_name(run_dir)
        previous = seen.get(case)
        if previous is not None:
            duplicates.append(f"{case}: {previous}, {run_dir}")
            continue
        seen[case] = run_dir
    if duplicates:
        preview = "; ".join(duplicates[:5])
        raise SystemExit(
            "Multiple run dirs normalize to the same evaluation case id, "
            "which would overwrite retrospective rankings and leakage run_id values. "
            f"Use unique run directory names: {preview}"
        )


def _reject_duplicate_resolved_run_dirs(run_dirs: list[Path]) -> None:
    seen: dict[Path, Path] = {}
    duplicates: list[str] = []
    for run_dir in run_dirs:
        resolved = run_dir.expanduser().resolve(strict=False)
        previous = seen.get(resolved)
        if previous is not None:
            duplicates.append(f"{resolved}: {previous}, {run_dir}")
            continue
        seen[resolved] = run_dir
    if duplicates:
        preview = "; ".join(duplicates[:5])
        raise SystemExit(
            "Multiple run dirs resolve to the same run directory, "
            "which would duplicate evaluation provenance. "
            f"Pass each completed run directory once: {preview}"
        )


def collect_run(run_dir: Path,
                out_dir: Path,
                rankings_dir: Path,
                sequence_map: dict[str, str],
                top_n_leakage: int) -> dict:
    case = _case_name(run_dir)
    compound = _read_required_json(
        run_dir / "01_input" / "compound_canonical.json",
        "Canonical compound metadata",
    )
    smiles = compound.get("canonical_smiles", "")
    if not isinstance(smiles, str) or not smiles.strip():
        raise SystemExit(
            "Canonical compound metadata missing non-empty 'canonical_smiles': "
            f"{run_dir / '01_input' / 'compound_canonical.json'}"
        )
    smiles = smiles.strip()
    _require_valid_smiles(smiles, "Canonical compound metadata")

    ranked_v3_with_efficacy = (
        run_dir / "03_targets" / "ranked_targets_v3_with_efficacy.csv"
    )
    ranked_v3_plain = run_dir / "03_targets" / "ranked_targets_v3.csv"
    ranked_v3 = _first_existing([ranked_v3_with_efficacy, ranked_v3_plain])
    comprehensive = run_dir / "03_targets" / "mode_comprehensive" / "top50_4way_consensus.csv"
    fast = run_dir / "03_targets" / "mode_fast" / "top50.csv"
    dti_only = _first_existing([
        run_dir / "03_targets" / "mode_comprehensive" / "psichic_sanity.tsv",
        run_dir / "03_targets" / "mode_fast" / "psichic_proteome.tsv",
    ])

    copied: list[str] = []
    copied_artifacts: list[dict[str, object]] = []
    leakage_rows: list[dict] = []
    skin_weighted: pd.DataFrame | None = None
    skin_weighted_path: Path | None = None
    if ranked_v3:
        df = _read_required_csv(
            ranked_v3,
            "Skin-weighted target ranking",
            {"target_id"},
            numeric_cols={
                "final_score",
                "skin_score",
                "docking_rrf",
                "efficacy_score",
            },
            fraction_cols={
                "final_score",
                "skin_score",
                "docking_rrf",
                "efficacy_score",
            },
        )
        ordered_df = _ordered_ranking(df, "Skin-weighted target ranking", ranked_v3)
        if ranked_v3 == ranked_v3_with_efficacy:
            _require_efficacy_columns(ordered_df, ranked_v3)
        final_min_sources = 2 if fast.exists() else 3
        _require_source_support(
            df,
            "Skin-weighted target ranking",
            ranked_v3,
            min_source_count=final_min_sources,
        )
        skin_weighted = ordered_df
        skin_weighted_path = ranked_v3
        retro_dir = rankings_dir / "cosmetic_retro"
        retro_dir.mkdir(parents=True, exist_ok=True)
        target = retro_dir / f"{case}__ranked_targets_v3.csv"
        _copy_atomic(ranked_v3, target)
        copied.append(str(target))
        copied_artifacts.append(_copied_artifact(ranked_v3, target))
        if ranked_v3 == ranked_v3_with_efficacy:
            efficacy_target = (
                retro_dir / f"{case}__ranked_targets_v3_with_efficacy.csv"
            )
            _copy_atomic(ranked_v3_with_efficacy, efficacy_target)
            copied.append(str(efficacy_target))
            copied_artifacts.append(
                _copied_artifact(ranked_v3_with_efficacy, efficacy_target)
            )

        for uid in ordered_df["target_id"].astype(str).head(top_n_leakage):
            row = {"run_id": case, "uniprot": uid, "smiles": smiles}
            if uid in sequence_map:
                row["sequence"] = sequence_map[uid]
            leakage_rows.append(row)

    for mode, path in (("comprehensive", comprehensive), ("fast", fast)):
        if path.exists():
            ranking = _read_required_csv(
                path,
                f"{mode.title()} target ranking",
                {"target_id"},
                numeric_cols={
                    "score",
                    "rrf_score",
                    "source_count",
                    "final_score",
                    "psichic_score",
                },
                fraction_cols={
                    "score",
                    "rrf_score",
                    "final_score",
                    "psichic_score",
                },
            )
            _require_source_support(
                ranking,
                f"{mode.title()} target ranking",
                path,
                min_source_count=3 if mode == "comprehensive" else 2,
            )
            if skin_weighted is not None and (
                (mode == "comprehensive")
                or (mode == "fast" and not comprehensive.exists())
            ):
                _require_target_subset(
                    skin_weighted,
                    "Skin-weighted target ranking",
                    ranking,
                    f"{mode.title()} target ranking",
                    skin_weighted_path or ranked_v3 or path,
                )
            target = rankings_dir / f"cold_start__{mode}.csv"
            _copy_atomic(path, target)
            copied.append(str(target))
            copied_artifacts.append(_copied_artifact(path, target))

    if dti_only is not None:
        df = _read_required_table(
            dti_only,
            "DTI-only target ranking",
            {"target_id"},
            sep="\t" if dti_only.suffix.lower() == ".tsv" else ",",
            numeric_cols={"score", "psichic_score", "pred", "predicted_affinity"},
            fraction_cols={"score", "psichic_score", "pred", "predicted_affinity"},
        )
        target = rankings_dir / "cold_start__dti_only.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_csv_atomic(df, target)
        copied.append(str(target))
        copied_artifacts.append(_copied_artifact(dti_only, target))

    if not copied:
        raise SystemExit(f"No evaluation ranking outputs found for run: {run_dir}")

    return {
        "run_id": case,
        "run_dir": str(run_dir),
        "canonical_smiles": smiles,
        "ranked_v3": str(ranked_v3) if ranked_v3 else "",
        "leakage_rows": leakage_rows,
        "copied": copied,
        "copied_artifacts": copied_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("results/eval"), type=Path)
    parser.add_argument("--rankings-dir",
                        default=Path("results/eval/rankings"),
                        type=Path)
    parser.add_argument("--sequence-fasta", type=Path, default=None,
                        help="Optional FASTA keyed by UniProt for leakage sequence axis.")
    parser.add_argument("--top-n-leakage", type=int, default=50)
    parser.add_argument("--out-manifest", type=Path, default=None)
    args = parser.parse_args()

    out_manifest = args.out_manifest or (args.out_dir / "collected_runs.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.rankings_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_outputs(out_manifest, args.out_dir, args.rankings_dir)
    try:
        if args.top_n_leakage <= 0:
            raise SystemExit("--top-n-leakage must be a positive integer")
        sequence_map = _load_sequence_map(args.sequence_fasta)
        _reject_duplicate_resolved_run_dirs(args.run_dirs)
        _reject_duplicate_case_ids(args.run_dirs)
        _reject_ambiguous_cold_start_outputs(args.run_dirs)

        runs = [
            collect_run(
                rd,
                args.out_dir,
                args.rankings_dir,
                sequence_map,
                args.top_n_leakage,
            )
            for rd in args.run_dirs
        ]
        leakage_rows = [row for run in runs for row in run["leakage_rows"]]
        if leakage_rows:
            _write_csv_atomic(
                pd.DataFrame(leakage_rows),
                args.out_dir / "eval_targets.csv",
            )

        manifest = {
            "out_dir": str(args.out_dir),
            "rankings_dir": str(args.rankings_dir),
            "n_runs": len(runs),
            "n_leakage_rows": len(leakage_rows),
            "runs": runs,
        }
        _write_text_atomic(out_manifest, json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    except (Exception, SystemExit):
        _clear_generated_outputs(out_manifest, args.out_dir, args.rankings_dir)
        raise


if __name__ == "__main__":
    main()
