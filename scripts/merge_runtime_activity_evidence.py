#!/usr/bin/env python3
"""Merge BindingDB and GtoPdb evidence into the table a run retrieves against.

Stage 3 scores a query against `data/chembl37/human_activities.parquet` and the
fingerprints beside it. That is ChEMBL 37 only, so 547 targets the repository has
already downloaded are unreachable from a researcher's run - among them nothing
exotic, just proteins BindingDB and GtoPdb happen to cover and ChEMBL does not.

This projects those two sources into the run path's schema and writes a merged
table plus the fingerprints for the ligands it adds.

Licence tiers, because they are a decision and not a detail:

    permissive  BindingDB rows BindingDB curated itself (CC BY 4.0)      +255
    +gtopdb     ... and GtoPdb (ODbL 1.0, contents CC BY-SA 4.0)          +49
    all         ... and BindingDB's ChEMBL-derived rows (CC BY-SA 3.0)   +243

`LICENSE_POLICY.md` admits CC-BY as commercial-use-permissive. Share-alike is a
further obligation, so it is opt-in: --sources defaults to the permissive tier.

Ligand identity: BindingDB and GtoPdb have no ChEMBL id, and the run path keys
on `molecule_chembl_id`. Added ligands get `BDB:<id>` / `GTP:<id>` keys, which
cannot collide with a real `CHEMBL...` accession. Where an added ligand shares a
standard InChIKey with one ChEMBL already has, the ChEMBL row wins and the
duplicate is dropped - retrieval would otherwise score the same structure twice.

The `<id>` is the source *compound* identity (`source_compound_id`, BindingDB
MonomerID when the evidence carries it), never the Reactant_set/measurement id:
one compound measured in many assays must keep one run-path key while its
measurements stay separate rows. Evidence built before the namespace split only
has the legacy `ligand_id`, which may be a measurement id; the manifest records
which basis produced the keys instead of silently reinterpreting it.

Run:
    python scripts/merge_runtime_activity_evidence.py \\
        --out-dir data/chembl37_merged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CHEMBL_DIR = ROOT / "data" / "chembl37"
BINDINGDB = ROOT / "data" / "bindingdb" / "evidence_202608" / "activity_evidence.parquet"
GTOPDB = ROOT / "data" / "gtopdb" / "evidence_v1" / "activity_evidence.parquet"

SCHEMA_VERSION = "skinscout.runtime-activity-evidence.v1"
LOG = logging.getLogger("merge-runtime-evidence")

# The run path's retrieval only understands these four; anything else has no
# comparable pActivity and would sit in the table as an incomparable number.
SUPPORTED_AFFINITY = {"IC50", "EC50", "KI", "KD"}
AFFINITY_TO_CHEMBL = {"IC50": "IC50", "EC50": "EC50", "KI": "Ki", "KD": "Kd"}

SOURCE_TIERS: dict[str, tuple[str, ...]] = {
    "permissive": ("bindingdb_own",),
    "gtopdb": ("bindingdb_own", "gtopdb"),
    "all": ("bindingdb_own", "gtopdb", "bindingdb_chembl_derived"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publication_key(frame: pd.DataFrame) -> pd.Series:
    """A citation for every row, in decreasing specificity.

    The retrieval index counts distinct publications per edge, so a blank key
    would either drop the row or make unrelated measurements look like one
    repeated source.
    """
    blank = {"", "nan", "none", "<na>", "null"}

    def _clean(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series("", index=frame.index, dtype=object)
        text = frame[name].fillna("").map(lambda value: str(value).strip())
        text = text.str.replace(r"\.0$", "", regex=True)
        return text.mask(text.str.lower().isin(blank), "")

    keys = pd.Series("", index=frame.index, dtype=object)
    for name, prefix in (
        ("source_pmid", "pmid:"),
        ("source_doi", "doi:"),
        ("source_patent", "patent:"),
        ("source_article_id", "bdb-article:"),
    ):
        text = _clean(name)
        fill = (keys == "") & (text != "")
        keys.loc[fill] = prefix + text.loc[fill]
    assert not keys.isna().any(), "publication key must never be NaN"
    return keys


def _to_nanomolar(value: pd.Series, unit: pd.Series) -> pd.Series:
    """Both sources report nM in practice; anything else becomes NaN, not a guess."""
    numeric = pd.to_numeric(value, errors="coerce")
    normalized = unit.astype(str).str.strip().str.lower()
    return numeric.where(normalized.isin({"nm", "nmol/l"}))


def _pactivity(nanomolar: pd.Series) -> pd.Series:
    """pChEMBL convention: -log10(molar). Non-positive values have no logarithm."""
    molar = nanomolar / 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(
            np.where(molar > 0, -np.log10(molar.to_numpy(dtype=float)), np.nan),
            index=nanomolar.index,
        )


def _load_source(path: Path, label: str, keep_chembl_derived: bool | None) -> pd.DataFrame:
    columns = [
        "ligand_id",
        "source_compound_id",
        "measurement_id",
        "structure_id",
        "ligand_inchikey",
        "ligand_smiles",
        "uniprot",
        "affinity_type",
        "affinity_value",
        "affinity_unit",
        "relation",
        "censor",
        "source_pmid",
        "source_doi",
        "source_patent",
        "source_article_id",
        "evidence_date",
        "source_db",
        "source_release",
        "source_license",
        "chembl_derived_license_flag",
    ]
    available = set(pq.read_schema(path).names)
    frame = pd.read_parquet(path, columns=[c for c in columns if c in available])
    if keep_chembl_derived is not None:
        # Without this column the tier is unenforceable. Carrying on would put
        # CC BY-SA 3.0 rows into a table whose manifest declares CC BY 4.0, and
        # --sources all would read the same file twice and double every row.
        if "chembl_derived_license_flag" not in frame.columns:
            raise SystemExit(
                f"{label}: {path} has no chembl_derived_license_flag column, so the "
                "licence tier cannot be applied; rebuild the source evidence"
            )
        flags = frame["chembl_derived_license_flag"].astype(bool)
        frame = frame[flags if keep_chembl_derived else ~flags]
    LOG.info("%s: %d rows before normalisation", label, len(frame))
    return frame.reset_index(drop=True)


_BLANK_ID_VALUES = {"", "nan", "none", "<na>", "null"}


def _clean_id_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    values = frame[name].fillna("").map(lambda value: str(value).strip())
    return values.mask(values.str.lower().isin(_BLANK_ID_VALUES), "")


def _compound_identity(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the run-path ligand key and the basis it came from.

    The explicit compound id outranks a derived structure id, and both outrank
    the legacy `ligand_id` - which was measurement-first and may name an assay.
    The basis is carried out of here so the manifest can record the namespace
    instead of having the fallback go unnoticed.
    """
    identity = _clean_id_column(frame, "source_compound_id")
    basis = pd.Series("source_compound_id", index=frame.index, dtype=object)
    structure = _clean_id_column(frame, "structure_id")
    use_structure = identity.eq("") & structure.ne("")
    identity = identity.where(~use_structure, structure)
    basis = basis.where(~use_structure, "structure_id")
    legacy = _clean_id_column(frame, "ligand_id")
    use_legacy = identity.eq("") & legacy.ne("")
    identity = identity.where(~use_legacy, legacy)
    basis = basis.where(~use_legacy, "legacy_ligand_id")
    basis = basis.mask(identity.eq(""), "missing")
    return identity, basis


def _distinct_nonblank(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    values = frame[column].fillna("").astype(str).str.strip()
    values = values.mask(values.str.lower().isin(_BLANK_ID_VALUES), "")
    return int(values[values.ne("")].nunique())


def _project(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Project a source into the run path's column names, dropping what cannot map."""
    if frame.empty:
        return frame
    affinity = frame["affinity_type"].astype(str).str.strip().str.upper()
    keep = affinity.isin(SUPPORTED_AFFINITY)
    frame = frame[keep].copy()
    affinity = affinity[keep]

    nanomolar = _to_nanomolar(frame["affinity_value"], frame["affinity_unit"])
    pactivity = _pactivity(nanomolar)

    # A censored measurement has a bound, not a value. ">10000 nM" is "did not
    # bind up to 10 uM" and must not retrieve as an affinity.
    #
    # "<" is dropped too, and that is a real cost rather than an oversight:
    # 194,401 BindingDB rows carry it, median implied pActivity 7.0, and 19
    # targets are reachable only through them. They are left out to stay
    # consistent with ChEMBL, whose pchembl_value is null for censored rows -
    # admitting one side and not the other would make source_consensus6 and the
    # potency features mean different things per source. Recording the bound as a
    # conservative pActivity is the obvious extension; it needs its own
    # measurement before it changes any ranking.
    if "censor" in frame.columns:
        censored = frame["censor"].astype(str).str.strip().str.lower().isin({"true", "1"})
    else:
        censored = pd.Series(False, index=frame.index)
    if "relation" in frame.columns:
        censored = censored | ~frame["relation"].astype(str).str.strip().isin({"=", ""})

    compound_ids, key_basis = _compound_identity(frame)
    projected = pd.DataFrame(
        {
            "molecule_chembl_id": prefix + ":" + compound_ids,
            # Internal counters carried until after deduplication; dropped
            # before the table is written so the run path schema is unchanged.
            "_measurement_id": _clean_id_column(frame, "measurement_id"),
            "_structure_id": _clean_id_column(frame, "structure_id"),
            "_key_basis": key_basis,
            "uniprot": frame["uniprot"].astype(str).str.strip(),
            "smiles": frame["ligand_smiles"].astype(str).str.strip(),
            "standard_inchi_key": frame["ligand_inchikey"].astype(str).str.strip(),
            "act_type": affinity.map(AFFINITY_TO_CHEMBL).values,
            "act_value": nanomolar.values,
            "act_units": "nM",
            "pchembl": pactivity.where(~censored.values).values,
            "standard_relation": frame.get("relation", pd.Series("=", index=frame.index)),
            "source_db": frame.get("source_db", pd.Series(prefix, index=frame.index)),
            "source_release": frame.get("source_release", pd.Series("", index=frame.index)),
            "source_license": frame.get("source_license", pd.Series("", index=frame.index)),
            "evidence_date": frame.get("evidence_date", pd.Series("", index=frame.index)),
            "pubmed_id": frame.get("source_pmid", pd.Series("", index=frame.index)),
            # BindingDB cites by patent or its own article id far more often than
            # by PMID: of 1,416,982 rows without a PMID, all but 876 carry one of
            # these. Carrying only source_pmid dropped them downstream.
            "publication_key": _publication_key(frame),
        }
    )
    valid = (
        projected["uniprot"].str.len().gt(0)
        & projected["smiles"].str.len().gt(0)
        & projected["standard_inchi_key"].str.len().gt(0)
        & projected["act_value"].notna()
        & compound_ids.str.len().gt(0)
    )
    dropped = int((~valid).sum())
    if dropped:
        LOG.info("%s: dropped %d row(s) with no usable value or identity", prefix, dropped)
    return projected[valid].reset_index(drop=True)


def _canonical_structure(smiles: str) -> str | None:
    """The identity the retrieval index will collapse this SMILES onto."""
    from rdkit import Chem, RDLogger

    from build_activity_retrieval_index import _standardize_mol

    RDLogger.DisableLog("rdApp.*")
    try:
        return Chem.MolToSmiles(_standardize_mol(smiles), canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _structure_keys(smiles: pd.Series, workers: int) -> pd.Series:
    """Canonicalise once per distinct SMILES, not once per row.

    Deduplicating on the raw `standard_inchi_key` does not work: BindingDB
    computes its keys largely without stereo perception, so 859,200 of its
    897,444 rows carry the stereo-free UHFFFAOYSA block while ChEMBL's carry
    real stereo. The string comparison matched zero rows while 118,069 of them
    had a SMILES byte-identical to a ChEMBL row - and the index, which derives
    identity from the structure, then held both as separate sources for one
    measurement.
    """
    text = smiles.astype(str).str.strip()
    unique = list(dict.fromkeys(text))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(_canonical_structure, unique, chunksize=512))
    else:
        values = [_canonical_structure(value) for value in unique]
    return text.map(dict(zip(unique, values)))


def _run_path_publication_key(frame: pd.DataFrame) -> pd.Series:
    """Normalize the ChEMBL-side citation into the merged table's key format."""
    blank = {"", "nan", "none", "<na>", "null"}

    def clean(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series("", index=frame.index, dtype=object)
        values = frame[name].fillna("").map(lambda value: str(value).strip())
        values = values.str.replace(r"\.0$", "", regex=True)
        return values.mask(values.str.lower().isin(blank), "")

    result = clean("publication_key")
    for name, prefix in (
        ("pubmed_id", "pmid:"),
        ("doi", "doi:"),
        ("document_chembl_id", "chembl:"),
    ):
        values = clean(name)
        fill = result.eq("") & values.ne("")
        result.loc[fill] = prefix + values.loc[fill]
    return result


def _measurement_keys(
    frame: pd.DataFrame, *, structure_keys: pd.Series, publication_keys: pd.Series
) -> list[tuple[object, ...]]:
    """Identity of one measurement, excluding the database that mirrored it."""
    endpoint = frame["act_type"].astype(str).str.strip().str.upper()
    relation = frame.get(
        "standard_relation", pd.Series("=", index=frame.index)
    ).fillna("").astype(str).str.strip()
    values = pd.to_numeric(frame["act_value"], errors="coerce")
    return list(
        zip(
            structure_keys.fillna(""),
            frame["uniprot"].astype(str).str.strip(),
            endpoint,
            relation,
            values,
            publication_keys.fillna("").astype(str).str.strip(),
            strict=True,
        )
    )


def _fingerprints(ligands: pd.DataFrame, workers: int) -> pd.DataFrame:
    # Serial on purpose: it runs once over the added ligands and the process
    # pool's pickling cost outweighs the work per molecule.
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    rows = []
    failures = 0
    for record in ligands.itertuples(index=False):
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            failures += 1
            continue
        array = np.zeros(2048, dtype=np.uint8)
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
        words = np.packbits(array).view("<u8")
        rows.append(
            {
                "molecule_chembl_id": record.molecule_chembl_id,
                "smiles": record.smiles,
                "bitvec": [str(int(word)) for word in words],
            }
        )
    if failures:
        LOG.warning("%d ligand(s) could not be fingerprinted and were dropped", failures)
    # An empty result still has to carry the columns the caller selects on.
    return pd.DataFrame(rows, columns=["molecule_chembl_id", "smiles", "bitvec"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-dir", type=Path, default=CHEMBL_DIR)
    parser.add_argument("--bindingdb", type=Path, default=BINDINGDB)
    parser.add_argument("--gtopdb", type=Path, default=GTOPDB)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sources",
        choices=sorted(SOURCE_TIERS),
        default="permissive",
        help=(
            "permissive (default) adds only BindingDB's own CC BY 4.0 curation; "
            "gtopdb also adds GtoPdb (share-alike); all also adds BindingDB's "
            "ChEMBL-derived rows (share-alike, largely duplicate)"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be added without writing the merged tables",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tiers = SOURCE_TIERS[args.sources]
    base_path = args.chembl_dir / "human_activities.parquet"
    if not base_path.exists():
        raise SystemExit(f"ChEMBL run-path table is missing: {base_path}")
    # Fail on the missing input, not on a traceback three functions deep.
    if any(tier.startswith("bindingdb") for tier in tiers) and not args.bindingdb.exists():
        raise SystemExit(f"--sources {args.sources} needs BindingDB: {args.bindingdb}")
    if "gtopdb" in tiers and not args.gtopdb.exists():
        raise SystemExit(f"--sources {args.sources} needs GtoPdb: {args.gtopdb}")

    base = pd.read_parquet(base_path)
    base_targets = set(base["uniprot"].astype(str).str.strip())
    base_structures: set[str] | None = None
    LOG.info("run path today: %d rows, %d targets", len(base), len(base_targets))

    added: list[pd.DataFrame] = []
    if "bindingdb_own" in tiers:
        added.append(
            _project(_load_source(args.bindingdb, "BindingDB (own)", False), "BDB")
        )
    if "bindingdb_chembl_derived" in tiers:
        added.append(
            _project(_load_source(args.bindingdb, "BindingDB (ChEMBL-derived)", True), "BDB")
        )
    if "gtopdb" in tiers:
        added.append(_project(_load_source(args.gtopdb, "GtoPdb", None), "GTP"))

    extra = pd.concat(added, ignore_index=True) if added else pd.DataFrame()
    if extra.empty:
        raise SystemExit("no rows to add; check --sources and the input paths")

    # A measurement ChEMBL already carries must not enter twice: the index counts
    # distinct source databases per (ligand, target) edge, so a re-import reads as
    # independent corroboration. Measured before this was keyed correctly: 151,552
    # of 1,616,575 pairs carried both a ChEMBL and a BindingDB edge and 95.4% of
    # them agreed to within 0.01 pActivity - one measurement, two "sources".
    #
    # Independent measurements of the same structure-target pair remain useful,
    # so deduplication includes publication, endpoint, relation, and value.  A
    # pair-level key erased corroborating and conflicting measurements alike.
    #
    # Identity comes from the canonical structure rather than the raw InChIKey:
    # BindingDB computes its keys largely without stereo perception, so 859,200
    # of its 897,444 rows carry the stereo-free UHFFFAOYSA block while ChEMBL's
    # carry real stereo, and the string comparison matched nothing at all.
    LOG.info("canonicalising structures to deduplicate against ChEMBL")
    base_measurements = set(
        _measurement_keys(
            base,
            structure_keys=_structure_keys(base["smiles"], args.workers),
            publication_keys=_run_path_publication_key(base),
        )
    )
    before = len(extra)
    extra_measurements = _measurement_keys(
        extra,
        structure_keys=_structure_keys(extra["smiles"], args.workers),
        publication_keys=extra["publication_key"],
    )
    duplicate = pd.Series(
        [
            key in base_measurements and bool(key[0]) and bool(key[-1])
            for key in extra_measurements
        ],
        index=extra.index,
    )
    extra = extra[~duplicate].reset_index(drop=True)
    exact_mirror_rows = before - len(extra)
    LOG.info(
        "dropped %d exact mirrored measurement row(s) already present in ChEMBL",
        exact_mirror_rows,
    )

    new_targets = set(extra["uniprot"]) - base_targets
    print(f"sources          : {args.sources} -> {', '.join(tiers)}")
    print(f"run path today   : {len(base):>9,} rows  {len(base_targets):>5,} targets")

    if args.dry_run:
        # Fingerprinting 800k ligands is the expensive half; a dry run reports
        # the upper bound and says so rather than pretending to a final number.
        print(f"rows addable     : {len(extra):>9,}  (before fingerprinting)")
        print(f"targets addable  : {len(new_targets):>9,}")
        print("\n(dry run - nothing written)")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprint first, then keep only the rows whose ligand actually has one.
    # Retrieval scores against the fingerprint table, so a row without one is an
    # edge nothing can ever match - it would inflate the row count and the target
    # count while reaching neither.
    ligands = (
        extra[["molecule_chembl_id", "smiles"]]
        .drop_duplicates(subset=["molecule_chembl_id"])
        .reset_index(drop=True)
    )
    LOG.info("fingerprinting %d added ligand(s)", len(ligands))
    added_fp = _fingerprints(ligands, args.workers)
    fingerprinted = set(added_fp["molecule_chembl_id"])
    orphaned = extra[~extra["molecule_chembl_id"].isin(fingerprinted)]
    if not orphaned.empty:
        LOG.warning(
            "dropping %d row(s) for %d ligand(s) that could not be fingerprinted",
            len(orphaned),
            len(set(orphaned["molecule_chembl_id"])),
        )
        extra = extra[extra["molecule_chembl_id"].isin(fingerprinted)].reset_index(drop=True)
        new_targets = set(extra["uniprot"]) - base_targets

    # Identity counters come from rows that survived mirror and fingerprint
    # filtering, then the helpers are dropped so the run-path schema is
    # unchanged.
    identity_basis_counts = Counter(
        str(value)
        for value in extra.get("_key_basis", pd.Series("", index=extra.index))
        if str(value) not in {"", "missing"}
    )
    distinct_measurement_ids_added = _distinct_nonblank(extra, "_measurement_id")
    distinct_structure_ids_added = _distinct_nonblank(extra, "_structure_id")
    helper_columns = [
        name
        for name in ("_measurement_id", "_structure_id", "_key_basis")
        if name in extra.columns
    ]
    if helper_columns:
        extra = extra.drop(columns=helper_columns)

    merged = pd.concat([base, extra], ignore_index=True)
    merged_path = args.out_dir / "human_activities.parquet"
    merged.to_parquet(merged_path, index=False)
    merged.to_parquet(args.out_dir / "activity_evidence.parquet", index=False)

    base_fp = pd.read_parquet(args.chembl_dir / "fp_morgan2_2048.parquet")
    fingerprints = pd.concat([base_fp, added_fp], ignore_index=True)
    fp_path = args.out_dir / "fp_morgan2_2048.parquet"
    fingerprints.to_parquet(fp_path, index=False)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_tier": args.sources,
        "sources": list(tiers),
        "inputs": {
            "chembl_human_activities": {
                "path": str(base_path.resolve()),
                "sha256": _sha256(base_path),
                "rows": int(len(base)),
            }
        },
        "outputs": {
            "human_activities": {"path": str(merged_path.resolve()), "rows": int(len(merged))},
            "fingerprints": {"path": str(fp_path.resolve()), "rows": int(len(fingerprints))},
        },
        "counts": {
            "rows_added": int(len(extra)),
            "targets_before": len(base_targets),
            "targets_after": len(base_targets | set(extra["uniprot"])),
            "targets_added": len(new_targets),
            "ligands_added": int(len(added_fp)),
            "rows_dropped_without_fingerprint": int(len(orphaned)),
            "rows_dropped_as_exact_mirror": int(exact_mirror_rows),
            # F32: which namespace keyed molecule_chembl_id. A run whose keys
            # came from legacy_ligand_id is measurement-keyed evidence, not a
            # compound-keyed one, and consumers can see that here.
            "ligand_keys_by_identity_basis": dict(identity_basis_counts),
            "distinct_molecule_keys_added": (
                int(extra["molecule_chembl_id"].nunique()) if not extra.empty else 0
            ),
            "distinct_measurement_ids_added": distinct_measurement_ids_added,
            "distinct_structure_ids_added": distinct_structure_ids_added,
        },
        # D05: base(ChEMBL 등)의 라이선스 의무도 인덱스에 함께 기록한다.
        # extra만 넣으면 근거의 81.7%가 share-alike 의무를 숨긴 채 배포된다.
        "licences": sorted(
            (
                set(base["source_license"].astype(str))
                if "source_license" in base.columns
                else set()
            )
            | set(extra["source_license"].astype(str))
        ),
        "policy": {
            "affinity_types": sorted(SUPPORTED_AFFINITY),
            "censored_measurements": "kept as rows, pchembl left null",
            "duplicate_measurements": (
                "drop only exact canonical-structure/target/endpoint/relation/value/"
                "publication mirrors; preserve independent measurements"
            ),
            "ligand_key_namespaces": {
                "BDB": (
                    "BindingDB source compound id (source_compound_id; BindingDB "
                    "MonomerID when present); legacy ligand_id only for evidence "
                    "built before the F32 namespace split"
                ),
                "GTP": "GtoPdb ligand id",
            },
            "ligand_measurement_identity": (
                "molecule_chembl_id keys compound identity only; measurement_id "
                "and measurement-level fields stay per row, so repeated "
                "measurements of one compound share a key while remaining "
                "distinct measurements"
            ),
        },
    }
    for path, label in ((args.bindingdb, "bindingdb"), (args.gtopdb, "gtopdb")):
        if path.exists() and any(label in tier for tier in tiers):
            names = set(pq.read_schema(path).names)
            manifest["inputs"][label] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "id_namespaces_present": {
                    "source_compound_id": "source_compound_id" in names,
                    "measurement_id": "measurement_id" in names,
                    "structure_id": "structure_id" in names,
                },
            }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"rows added       : {len(extra):>9,}")
    print(f"targets added    : {len(new_targets):>9,}")
    if not orphaned.empty:
        print(f"  dropped        : {len(orphaned):>9,} rows with no fingerprint")
    print(f"merged total     : {len(merged):>9,} rows  "
          f"{len(base_targets | set(extra['uniprot'])):>5,} targets")
    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
