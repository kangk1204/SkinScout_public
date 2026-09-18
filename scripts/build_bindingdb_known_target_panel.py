#!/usr/bin/env python3
"""Build a supplemental known-compound panel from BindingDB.

This generates a CSV compatible with ``stage3_known_target_prior.py`` and intended
for use as ``known_target_priors.supplemental_source_csv`` in ``config.yaml``.

Input assumptions:
- ``BindingDB_All.tsv`` has ligand SMILES and per-chain target UniProt IDs.
- Human targets are preferred for skin pipelines.
- RDKit is used for compound canonicalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Iterable

from rdkit import Chem


LOG = logging.getLogger("build_bindingdb_known_target_panel")
INCHI_KEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
UNIPROT_ID_RE = re.compile(r"^[A-Z][0-9A-Z]{5,10}$")
CSV_FIELD_SIZE_LIMIT = 20_000_000
AFFECTED_AFFINITY_COLUMNS = ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]
REQUIRED_BINDINGDB_COLUMNS = {
    "Ligand SMILES",
    "Target Source Organism According to Curator or DataSource",
}


@dataclass
class TargetEvidence:
    target_label: str
    source_weight: float
    evidence_note: str


@dataclass
class CompoundEvidence:
    case_id: str
    smiles: str
    inchi_key: str = ""
    targets: dict[str, TargetEvidence] = field(default_factory=dict)


@contextmanager
def iter_bindingdb_rows(path: Path) -> Iterable[tuple[list[str], dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"BindingDB TSV required and must be non-empty: {path}")

    csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
    handle = path.open("r", encoding="utf-8", errors="replace")
    try:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"BindingDB TSV missing header: {path}")
        yield list(reader.fieldnames), reader
    finally:
        handle.close()


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _split_multi(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    out = []
    for token in re.split(r"[;,]", text):
        token = token.strip()
        if token and token.upper() != "NA":
            out.append(token)
    return out


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip(",")
    if not text or text.lower() in {"na", "nan", "null"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalize_smiles(smiles: str) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _normalize_inchi_key(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text if INCHI_KEY_RE.fullmatch(text) else None


def _is_human(value: object) -> bool:
    text = str(value or "").strip().lower()
    return "homo sapiens" in text or text == "human" or ",human," in f",{text},"


def _build_chain_columns(fieldnames: list[str], max_chains: int) -> dict[int, dict[str, list[str]]]:
    groups = {idx: {"ids": [], "labels": []} for idx in range(1, max_chains + 1)}
    for idx in range(1, max_chains + 1):
        id_cols = [
            f"UniProt (SwissProt) Primary ID of Target Chain {idx}",
            f"UniProt (SwissProt) Secondary ID(s) of Target Chain {idx}",
            f"UniProt (SwissProt) Alternative ID(s) of Target Chain {idx}",
            f"UniProt (TrEMBL) Primary ID of Target Chain {idx}",
            f"UniProt (TrEMBL) Secondary ID(s) of Target Chain {idx}",
            f"UniProt (TrEMBL) Alternative ID(s) of Target Chain {idx}",
        ]
        label_cols = [
            f"UniProt (SwissProt) Recommended Name of Target Chain {idx}",
            f"UniProt (SwissProt) Entry Name of Target Chain {idx}",
            f"UniProt (TrEMBL) Submitted Name of Target Chain {idx}",
            f"UniProt (TrEMBL) Entry Name of Target Chain {idx}",
        ]
        groups[idx]["ids"] = [col for col in id_cols if col in fieldnames]
        groups[idx]["labels"] = [col for col in label_cols if col in fieldnames]
    return groups


def _validate_bindingdb_headers(fieldnames: list[str], chain_columns: dict[int, dict[str, list[str]]]) -> None:
    available = set(fieldnames)
    missing = sorted(REQUIRED_BINDINGDB_COLUMNS - available)
    if missing:
        raise SystemExit(
            "BindingDB TSV missing required column(s): " + ", ".join(missing)
        )
    if not any(col in available for col in AFFECTED_AFFINITY_COLUMNS):
        raise SystemExit(
            "BindingDB TSV missing required affinity column; expected one of "
            + ", ".join(AFFECTED_AFFINITY_COLUMNS)
        )
    if not any(groups["ids"] for groups in chain_columns.values()):
        raise SystemExit(
            "BindingDB TSV missing required target chain UniProt ID column"
        )


def _collect_chain_targets(
    row: dict[str, str], chain_columns: dict[int, dict[str, list[str]]]
) -> dict[str, str]:
    targets: dict[str, str] = {}
    for groups in chain_columns.values():
        labels: dict[str, str] = {}
        for col in groups["labels"]:
            for raw_name in _split_multi(row.get(col)):
                if raw_name:
                    labels[raw_name.lower()] = raw_name

        for col in groups["ids"]:
            for uniprot in _split_multi(row.get(col)):
                if not UNIPROT_ID_RE.fullmatch(uniprot):
                    continue
                if uniprot in targets:
                    continue
                target_label = ", ".join(labels.values()) if labels else uniprot
                targets[uniprot] = target_label.split(",")[0][:80]
    return targets


def _best_affinity_value(row: dict[str, str]) -> float | None:
    values: list[float] = []
    for col in AFFECTED_AFFINITY_COLUMNS:
        parsed = _parse_float(row.get(col))
        if parsed is not None and parsed > 0:
            values.append(parsed)
    return min(values) if values else None


def _weight_from_affinity(affinity_nm: float | None, default_weight: float) -> float:
    if affinity_nm is None:
        return default_weight
    if affinity_nm <= 0:
        return 1.0
    # 1 nM -> 1.0; 10 µM (1e4 nM) -> ~0.4; 1 mM (1e6 nM) -> 0.2
    decade = math.log10(max(affinity_nm, 1.0))
    value = 1.0 - (decade / 6.0) * 0.8
    return min(1.0, max(0.2, value))


def _evidence_descriptor(row: dict[str, str]) -> str:
    ref = (
        row.get("BindingDB Entry DOI")
        or row.get("BindingDB Ligand Title")
        or row.get("BindingDB PMID")
        or "BindingDB"
    )
    return str(ref).strip()[:120] if ref else "BindingDB"


def _case_id(inchi_key: str | None, smiles: str) -> str:
    if inchi_key and INCHI_KEY_RE.fullmatch(inchi_key):
        return f"BDB_{inchi_key.replace('-', '_')}"
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:12]
    return f"BDB_{digest}"


def _load_skin_targets(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Skin target whitelist required and non-empty: {path}")
    out: set[str] = set()
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        value = line.strip().strip(",")
        if not value or value.lower().startswith("uniprot"):
            continue
        for token in _split_multi(value):
            token = token.strip().upper()
            if UNIPROT_ID_RE.fullmatch(token):
                out.add(token)
    if not out:
        raise SystemExit(f"No valid UniProt IDs loaded from {path}")
    return out


def build_panel(
    bindingdb_tsv: Path,
    *,
    out_csv: Path,
    max_rows: int,
    max_targets_per_compound: int,
    max_compounds: int,
    min_source_weight: float,
    default_source_weight: float,
    require_human: bool,
    skin_target_set: set[str] | None,
    max_affinity_nm: float | None,
    skip_no_smiles: bool,
    chain_max: int,
) -> tuple[int, int, int, int]:
    out_csv.unlink(missing_ok=True)
    out_csv.with_suffix(out_csv.suffix + ".tmp").unlink(missing_ok=True)
    compounds: DefaultDict[str, CompoundEvidence] = defaultdict(lambda: CompoundEvidence("", ""))
    rows_seen = 0

    with iter_bindingdb_rows(bindingdb_tsv) as (fieldnames, rows):
        chain_columns = _build_chain_columns(fieldnames, chain_max)
        _validate_bindingdb_headers(fieldnames, chain_columns)

        for row in rows:
            rows_seen += 1
            if max_rows > 0 and rows_seen > max_rows:
                break

            if require_human and not _is_human(row.get("Target Source Organism According to Curator or DataSource")):
                continue

            raw_smiles = str(row.get("Ligand SMILES") or "").strip()
            smiles = _normalize_smiles(raw_smiles)
            if smiles is None:
                if skip_no_smiles:
                    continue
                raise SystemExit(f"Invalid or missing ligand SMILES at line {rows_seen}")

            inchi_key = _normalize_inchi_key(row.get("Ligand InChI Key"))
            case_id_value = _case_id(inchi_key, smiles)

            raw_targets = _collect_chain_targets(row, chain_columns)
            if not raw_targets:
                continue

            if skin_target_set is not None:
                raw_targets = {
                    target_id: target_label
                    for target_id, target_label in raw_targets.items()
                    if target_id in skin_target_set
                }
                if not raw_targets:
                    continue

            affinity_nm = _best_affinity_value(row)
            if max_affinity_nm is not None:
                if affinity_nm is None or affinity_nm > max_affinity_nm:
                    continue

            source_weight = _weight_from_affinity(affinity_nm, default_source_weight)
            if source_weight < min_source_weight:
                continue

            evidence_note = _evidence_descriptor(row)
            compound = compounds[case_id_value]
            if not compound.case_id:
                compound.case_id = case_id_value
                compound.smiles = smiles
                compound.inchi_key = inchi_key or ""

            for target_id, target_label in raw_targets.items():
                previous = compound.targets.get(target_id)
                if previous is None or source_weight > previous.source_weight:
                    compound.targets[target_id] = TargetEvidence(
                        target_label=target_label,
                        source_weight=source_weight,
                        evidence_note=evidence_note,
                    )

    # Deterministic ordering and row materialization.
    grouped = sorted(compounds.values(), key=lambda item: item.case_id)
    out_rows = []
    for compound in grouped:
        if not compound.case_id:
            continue

        targets = sorted(
            compound.targets.items(),
            key=lambda item: (-item[1].source_weight, item[0]),
        )
        if max_targets_per_compound > 0:
            targets = targets[:max_targets_per_compound]
        if not targets:
            continue

        target_ids = [tid for tid, _ in targets]
        target_labels = [
            entry.target_label for _, entry in targets if entry.target_label
        ]
        target_weights = [entry.source_weight for _, entry in targets]
        target_evidence_notes = [entry.evidence_note for _, entry in targets]
        source_weight = max(entry.source_weight for _, entry in targets)

        out_rows.append(
            {
                "case_id": compound.case_id,
                "smiles": compound.smiles,
                "inchi_key": compound.inchi_key,
                "known_targets": ";".join(target_ids),
                "known_target_labels": ";".join(target_labels),
                "known_target_weights": ";".join(f"{weight:.6f}" for weight in target_weights),
                "known_target_evidence_notes": ";".join(target_evidence_notes),
                "source_weight": f"{source_weight:.6f}",
                "evidence_note": targets[0][1].evidence_note or "BindingDB evidence",
            }
        )

    if max_compounds > 0 and len(out_rows) > max_compounds:
        out_rows = out_rows[:max_compounds]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "smiles",
                "inchi_key",
                "known_targets",
                "known_target_labels",
                "known_target_weights",
                "known_target_evidence_notes",
                "source_weight",
                "evidence_note",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    tmp_csv.replace(out_csv)

    emitted = len(out_rows)
    compounds_seen = len([item for item in compounds.values() if item.case_id])
    target_count = sum(len(record.targets) for record in compounds.values() if record.case_id)
    return rows_seen, compounds_seen, emitted, target_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindingdb-tsv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--skin-target-uniprot", type=Path, default=None)
    parser.add_argument("--out-summary-json", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-targets-per-compound", type=int, default=25)
    parser.add_argument("--max-compounds", type=int, default=0)
    parser.add_argument("--min-source-weight", type=float, default=0.20)
    parser.add_argument("--default-source-weight", type=float, default=0.40)
    parser.add_argument("--max-affinity-nm", type=float, default=None)
    parser.add_argument("--chain-max", type=int, default=30)
    parser.add_argument("--include-non-human", action="store_true")
    parser.add_argument("--skip-missing-smiles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out_csv.unlink(missing_ok=True)
    args.out_csv.with_suffix(args.out_csv.suffix + ".tmp").unlink(missing_ok=True)
    if args.out_summary_json is not None:
        args.out_summary_json.unlink(missing_ok=True)
        args.out_summary_json.with_suffix(
            args.out_summary_json.suffix + ".tmp"
        ).unlink(missing_ok=True)

    if not args.bindingdb_tsv.exists():
        raise SystemExit(f"BindingDB TSV does not exist: {args.bindingdb_tsv}")
    if not math.isfinite(args.default_source_weight) or not 0.0 <= args.default_source_weight <= 1.0:
        raise SystemExit("--default-source-weight must be in [0, 1]")
    if not math.isfinite(args.min_source_weight) or not 0.0 <= args.min_source_weight <= 1.0:
        raise SystemExit("--min-source-weight must be in [0, 1]")
    if args.max_targets_per_compound < 0:
        raise SystemExit("--max-targets-per-compound must be >= 0")
    if args.max_compounds < 0:
        raise SystemExit("--max-compounds must be >= 0")
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be >= 0")
    if args.max_affinity_nm is not None and (
        not math.isfinite(args.max_affinity_nm) or args.max_affinity_nm <= 0
    ):
        raise SystemExit("--max-affinity-nm must be > 0 and finite when provided")
    if args.chain_max < 1 or args.chain_max > 100:
        raise SystemExit("--chain-max must be between 1 and 100")

    skin_targets = _load_skin_targets(args.skin_target_uniprot)
    rows_seen, compounds_seen, emitted, targets = build_panel(
        args.bindingdb_tsv,
        out_csv=args.out_csv,
        max_rows=args.max_rows,
        max_targets_per_compound=args.max_targets_per_compound,
        max_compounds=args.max_compounds,
        min_source_weight=args.min_source_weight,
        default_source_weight=args.default_source_weight,
        require_human=not args.include_non_human,
        skin_target_set=skin_targets,
        max_affinity_nm=args.max_affinity_nm,
        skip_no_smiles=args.skip_missing_smiles,
        chain_max=args.chain_max,
    )

    LOG.info(
        "wrote %d rows (%d compounds, %d target annotations) from %d BindingDB rows",
        emitted,
        compounds_seen,
        targets,
        rows_seen,
    )

    if args.out_summary_json is not None:
        summary = {
            "bindingdb_tsv": str(args.bindingdb_tsv),
            "out_csv": str(args.out_csv),
            "rows_seen": rows_seen,
            "compounds": compounds_seen,
            "out_rows": emitted,
            "target_annotations": targets,
            "max_rows": args.max_rows,
            "max_targets_per_compound": args.max_targets_per_compound,
            "max_compounds": args.max_compounds,
            "min_source_weight": args.min_source_weight,
            "default_source_weight": args.default_source_weight,
            "max_affinity_nm": args.max_affinity_nm,
            "require_human": not args.include_non_human,
            "skin_target_filter": bool(args.skin_target_uniprot),
        }
        _write_text_atomic(json.dumps(summary, indent=2, sort_keys=True) + "\n", args.out_summary_json)


if __name__ == "__main__":
    main()
