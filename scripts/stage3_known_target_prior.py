#!/usr/bin/env python3
"""Create known target priors for exact and similarity-matched known compounds.

The input panel must contain compound SMILES and target annotations. Exact canonical
SMILES matching is preferred; when no exact match is found and similarity fallback is
enabled, Morgan fingerprint Tanimoto matching can be used.
"""

from __future__ import annotations

import argparse
import math
import json
import logging
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd

LOG = logging.getLogger("stage3.known_target_prior")
INCHI_KEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
KNOWN_PRIOR_COMBINATION_CHOICES = {"max", "sum_capped", "bayesian_or"}


def _parse_prior_combination(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in KNOWN_PRIOR_COMBINATION_CHOICES:
        return text
    raise SystemExit(
        "--prior-combination must be one of: max, sum_capped, bayesian_or"
    )


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(
            f"Compound metadata is required and must be non-empty: {path}"
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Compound metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Compound metadata must be an object: {path}")
    return payload


def _canonical_smiles(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise SystemExit(f"{label} contains blank SMILES value")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise SystemExit(f"{label} contains an invalid SMILES: {text!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def _split_semicolon_list(value: object, label: str, row_id: str) -> list[str]:
    text = str(value).strip()
    if not text:
        raise SystemExit(f"{label} contains blank values: {row_id}")
    tokens = [token.strip() for token in text.split(";")]
    if any(not token for token in tokens):
        raise SystemExit(
            f"{label} contains empty ';'-separated token(s): {row_id}"
        )
    return tokens


def _parse_known_targets(value: object, row_id: str) -> list[str]:
    targets = _split_semicolon_list(value, "known_targets", row_id)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for target in targets:
        if target in seen:
            duplicates.add(target)
        seen.add(target)
    if duplicates:
        raise SystemExit(
            f"known_targets contains duplicate target(s) {','.join(sorted(duplicates))}: {row_id}"
        )
    return targets


def _parse_bool(value: object, label: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    raise SystemExit(f"{label} must be a boolean-like value: {value!r}")


def _parse_inchi_key(value: object, label: str, row_id: str) -> str | None:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return None
    text = text.upper()
    if not INCHI_KEY_RE.fullmatch(text):
        raise SystemExit(f"{label} contains invalid InChIKey: {row_id}")
    return text


def _parse_weight(value: object, label: str, row_id: str) -> float:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        raise SystemExit(
            f"{label} contains blank values; omit the optional column or provide "
            f"a finite weight in (0, 1]: {row_id}"
        )
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"{label} must be a finite number in (0, 1]: {row_id}"
        ) from exc
    if not (0.0 < parsed <= 1.0):
        raise SystemExit(
            f"{label} must be a finite number in (0, 1]: {row_id}"
        )
    return parsed


def _parse_weight_list(
    value: object,
    *,
    fallback_weight: float,
    expected_count: int,
    label: str,
    row_id: str,
) -> list[float]:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        raise SystemExit(
            f"{label} contains blank values; omit the optional column or provide "
            f"one finite weight per known target: {row_id}"
        )
    tokens = _split_semicolon_list(value, label, row_id)
    if len(tokens) != expected_count:
        raise SystemExit(f"{label} count must match known_targets count: {row_id}")
    return [
        _parse_weight(token, label, f"{row_id} target_weight[{idx}]")
        for idx, token in enumerate(tokens, start=1)
    ]


def _nonempty_smiles_from_panel(value: object, row_id: str) -> str:
    return _canonical_smiles(value, f"case_id={row_id}")


def _morgan_fingerprint(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"Invalid SMILES encountered during fingerprinting: {smiles!r}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _scale_similarity_score(similarity: float, threshold: float) -> float:
    if similarity >= 1.0:
        return 1.0
    if similarity <= threshold:
        return 0.0
    return (similarity - threshold) / (1.0 - threshold)


def _combine_prior_scores(values: list[float], strategy: str) -> float:
    if not values:
        return 0.0
    if strategy == "max":
        return max(values)
    if strategy == "sum_capped":
        return min(1.0, sum(values))
    if strategy == "bayesian_or":
        miss = 1.0
        for value in values:
            miss *= 1.0 - value
        return 1.0 - miss
    raise SystemExit(f"Unsupported prior-combination strategy: {strategy}")


def _read_known_panel(path: Path, source_panel: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Known-target panel is required and must be non-empty: {path}")
    try:
        panel = pd.read_csv(path)
    except Exception as exc:
        raise SystemExit(f"Known-target panel failed to parse: {path}: {exc}") from exc
    required = {"case_id", "smiles", "known_targets"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise SystemExit(
            "Known-target panel missing required columns "
            + ", ".join(missing)
            + f": {path}"
        )
    if panel.empty:
        raise SystemExit(f"Known-target panel contains no rows: {path}")
    panel = panel.copy()
    panel["case_id"] = panel["case_id"].astype(str).str.strip()
    if panel["case_id"].duplicated().any():
        duplicates = ", ".join(panel["case_id"][panel["case_id"].duplicated()].tolist())
        raise SystemExit(
            f"{source_panel} contains duplicate case_id values: {duplicates}"
        )
    panel["canonical_smiles"] = [
        _nonempty_smiles_from_panel(smiles, f"case_id={case_id}")
        for case_id, smiles in zip(panel["case_id"], panel["smiles"], strict=True)
    ]
    has_inchi_key = "inchi_key" in panel.columns
    panel["inchi_key"] = [
        _parse_inchi_key(
            row.inchi_key if has_inchi_key else None,
            "Panel column 'inchi_key'",
            f"case_id={case_id}",
        )
        for case_id, row in zip(panel["case_id"], panel.itertuples(index=False), strict=True)
    ]
    panel["known_targets"] = [
        _parse_known_targets(targets, f"case_id={case_id}")
        for case_id, targets in zip(panel["case_id"], panel["known_targets"], strict=True)
    ]
    panel["source_weight"] = [
        _parse_weight(
            row.source_weight if "source_weight" in panel.columns else 1.0,
            "Panel column 'source_weight'",
            f"case_id={case_id}",
        )
        for case_id, row in zip(panel["case_id"], panel.itertuples(index=False), strict=True)
    ]
    has_target_weights = "known_target_weights" in panel.columns
    panel["known_target_weights"] = [
        _parse_weight_list(
            row.known_target_weights if has_target_weights else ";".join(
                str(row.source_weight) for _ in row.known_targets
            ),
            fallback_weight=float(row.source_weight),
            expected_count=len(row.known_targets),
            label="Panel column 'known_target_weights'",
            row_id=f"case_id={case_id}",
        )
        for case_id, row in zip(panel["case_id"], panel.itertuples(index=False), strict=True)
    ]
    panel["source_panel"] = source_panel
    return panel[[
        "case_id",
        "canonical_smiles",
        "inchi_key",
        "source_weight",
        "known_target_weights",
        "source_panel",
        "known_targets",
    ]]


def _write_empty_prior(path: Path) -> None:
    _write_csv_atomic(
        pd.DataFrame(
            columns=[
                "target_id",
                "prior_score",
                "source_case_id",
                "source_panel",
                "evidence_count",
                "evidence_case_ids",
                "evidence_panels",
            ]
        ),
        path,
    )


def _build_known_target_prior(
    compound_json: Path,
    panel_csv: Path,
    supplemental_panel_csvs: list[Path],
    out_csv: Path,
    enabled: bool,
    *,
    similarity_enabled: bool = False,
    similarity_threshold: float = 0.85,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    max_similarity_candidates: int = 0,
    prior_combination: str = "max",
    similarity_score_power: float = 1.0,
) -> None:
    if not enabled:
        _write_empty_prior(out_csv)
        return

    compound = _read_json_object(compound_json)
    compound_smiles = compound.get("canonical_smiles")
    if not compound_smiles:
        raise SystemExit("compound_canonical.json is missing canonical_smiles")
    compound_canonical = _canonical_smiles(
        compound_smiles,
        "compound_canonical.json canonical_smiles",
    )
    compound_inchi = Chem.MolToInchiKey(Chem.MolFromSmiles(compound_canonical))
    compound_fp = (
        _morgan_fingerprint(compound_canonical, morgan_radius, morgan_n_bits)
        if similarity_enabled else None
    )

    panel_sources = [("skin_known_target_panel", panel_csv)]
    for idx, path in enumerate(supplemental_panel_csvs, start=1):
        panel_sources.append((f"skin_known_target_panel_{idx}", path))
    panel = pd.concat(
        [_read_known_panel(path, source_panel) for source_panel, path in panel_sources],
        ignore_index=True,
    )
    matches: dict[str, list[tuple[str, str, float]]] = {}
    similarity_candidates: list[tuple[float, str, str, list[tuple[str, float]]]] = []
    exact_seen = False
    for row in panel.itertuples(index=False):
        case_id = str(row.case_id)
        row_score: dict[str, float] = {}
        target_weights = {
            target: float(weight)
            for target, weight in zip(row.known_targets, row.known_target_weights, strict=True)
        }

        if row.canonical_smiles == compound_canonical:
            exact_seen = True
            row_score = target_weights
        elif row.inchi_key and row.inchi_key == compound_inchi:
            exact_seen = True
            row_score = target_weights
        elif similarity_enabled and compound_fp is not None:
            row_fp = _morgan_fingerprint(row.canonical_smiles, morgan_radius, morgan_n_bits)
            similarity = DataStructs.TanimotoSimilarity(compound_fp, row_fp)
            prior_score = _scale_similarity_score(similarity, similarity_threshold)
            if similarity_score_power != 1.0:
                prior_score = prior_score ** similarity_score_power
            if prior_score > 0.0:
                similarity_candidates.append(
                    (
                        prior_score,
                        row.source_panel,
                        case_id,
                        [
                            (target, prior_score * target_weights[target])
                            for target in row.known_targets
                        ],
                    )
                )

        for target, prior_score in row_score.items():
            matches.setdefault(target, []).append(
                (case_id, row.source_panel, prior_score)
            )

    if similarity_candidates and not exact_seen:
        similarity_candidates.sort(reverse=True, key=lambda item: item[0])
        if max_similarity_candidates > 0:
            similarity_candidates = similarity_candidates[:max_similarity_candidates]
        for _row_score, source_panel, case_id, target_scores in similarity_candidates:
            for target, prior_score in target_scores:
                matches.setdefault(target, []).append((case_id, source_panel, prior_score))

    if not matches:
        LOG.info(
            "No known-target panel match for compound %s; wrote empty prior",
            compound_canonical,
        )
        _write_empty_prior(out_csv)
        return

    out_rows: list[dict[str, object]] = []
    for target_id, evidences in sorted(matches.items(), key=lambda item: item[0]):
        evidences.sort(key=lambda item: (-item[2], item[0], item[1]))
        prior_score = _combine_prior_scores(
            [prior_score for _case_id, _source_panel, prior_score in evidences],
            prior_combination,
        )
        source_case_ids = sorted(set(item[0] for item in evidences))
        source_panels = sorted(set(item[1] for item in evidences))
        top_case_id, top_source_panel, _ = evidences[0]
        out_rows.append(
            {
                "target_id": target_id,
                "prior_score": round(prior_score, 6),
                "source_case_id": top_case_id,
                "source_panel": top_source_panel,
                "evidence_count": len(evidences),
                "evidence_case_ids": ";".join(source_case_ids),
                "evidence_panels": ";".join(source_panels),
            }
        )
    out = pd.DataFrame(out_rows)
    out = out.sort_values(["prior_score", "target_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    _write_csv_atomic(out, out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compound-json", required=True, type=Path)
    parser.add_argument("--known-panel", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--enabled", type=str, default="false")
    parser.add_argument("--similarity-enabled", type=str, default="false")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-n-bits", type=int, default=2048)
    parser.add_argument("--supplemental-panels", type=str, default="")
    parser.add_argument("--prior-combination", type=str, default="max")
    parser.add_argument("--similarity-score-power", type=float, default=1.0)
    parser.add_argument("--max-similarity-candidates", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_csv)
    enabled = _parse_bool(args.enabled, "--enabled")
    similarity_enabled = _parse_bool(args.similarity_enabled, "--similarity-enabled")
    if not 0.0 < args.similarity_threshold <= 1.0:
        raise SystemExit(
            "--similarity-threshold must be in (0.0, 1.0]: "
            f"{args.similarity_threshold:g}"
        )
    if args.morgan_radius < 1 or args.morgan_n_bits < 128:
        raise SystemExit(
            "--morgan-radius and --morgan-n-bits must be positive and >=128"
        )
    if args.max_similarity_candidates < 0:
        raise SystemExit(
            "--max-similarity-candidates must be >= 0 (0 means unlimited)"
        )
    if not (args.similarity_score_power > 0.0 and math.isfinite(args.similarity_score_power)):
        raise SystemExit("--similarity-score-power must be > 0 and finite")
    prior_combination = _parse_prior_combination(args.prior_combination)
    supplemental_panels = [
        Path(path.strip())
        for path in args.supplemental_panels.split(",")
        if path.strip()
    ]

    _build_known_target_prior(
        compound_json=args.compound_json,
        panel_csv=args.known_panel,
        supplemental_panel_csvs=supplemental_panels,
        out_csv=args.out_csv,
        enabled=enabled,
        similarity_enabled=similarity_enabled,
        similarity_threshold=args.similarity_threshold,
        morgan_radius=args.morgan_radius,
        morgan_n_bits=args.morgan_n_bits,
        prior_combination=prior_combination,
        similarity_score_power=args.similarity_score_power,
        max_similarity_candidates=args.max_similarity_candidates,
    )


if __name__ == "__main__":
    main()
