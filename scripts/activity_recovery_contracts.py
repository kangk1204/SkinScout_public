#!/usr/bin/env python3
"""Immutable contracts shared by activity-recovery builders and evaluators."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


# Amended 2026-08-26. v1 preregistered four compounds whose SMILES were a
# different molecule than the case_id named: farnesol filed as retinol, a
# saturated lactone as ascorbic acid, a five-membered furanone as kojic acid,
# and an EGCG isomer with the A- and B-ring hydroxyls transposed. Eleven of the
# panel's thirty-two pairs were measured against the wrong structure.
#
# The amendment corrects structures only - no case, target or threshold moved -
# and it is not outcome-driven: it made ascorbic acid (169 -> 703) and
# EGCG -> DNMT1 (52 -> 506) worse while making retinol (467 -> 25) better. The
# v1 digest stays recorded so a v1 artifact can still be identified.
# Evidence and full before/after: docs/PANEL_STRUCTURE_CORRECTION_20260826.md
SUPERSEDED_KNOWN_PANEL_CONTRACTS: dict[str, str] = {
    "skinscout.skin-known-target-panel.v1": (
        "1e0dd6763ea7f245839002bbe027a3a0cefb50137cf2959f178ee41d307a5017"
    ),
    # v2 held 15 cases and 32 pairs. It was replaced because it could not decide
    # what it was being used to decide: the Top30 difference that blocked
    # promoting `union_p6_consensus` was 0.125 of 32 pairs, i.e. four pairs, and
    # exact McNemar cannot put four pairs below p=0.05 under any arrangement of
    # the discordance. See docs/PANEL_ARBITRATION_20260831.md.
    "skinscout.skin-known-target-panel.v2": (
        "e76a265db2e8c56554378091b0dd7d06de0e678076e64ed93923803e572bf8ec"
    ),
    # The first v3 draft, superseded within the hour: its tranexamic acid entry
    # was the cis isomer. Recorded so an artifact built from it is identifiable
    # rather than mistaken for the corrected v3.
    "skinscout.skin-known-target-panel.v3-draft": (
        "62bedcedfb74462202dd257ba2a888d4ce204dd9081c78b191e37232bde16dd1"
    ),
}
FROZEN_KNOWN_PANEL_CONTRACT = "skinscout.skin-known-target-panel.v3"
FROZEN_KNOWN_PANEL_SHA256 = (
    "fa9fea0179b3e4c3305aeea157914ce4562d14d00f67092a5542c77d9cd249d0"
)
FROZEN_KNOWN_PANEL_CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Retinol": (
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/CO)C(C)(C)CCC1",
        ("P10276", "P10826", "P13631"),
    ),
    "Niacinamide": (
        "NC(=O)c1cccnc1",
        ("P40261", "Q96EB6", "P09874"),
    ),
    "Ascorbic_acid": (
        "O=C1O[C@H]([C@@H](O)CO)C(O)=C1O",
        ("P07237", "P13674", "O15460"),
    ),
    "alpha-Arbutin": (
        "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O",
        ("P14679",),
    ),
    "Kojic_acid": ("O=c1cc(CO)occ1O", ("P14679",)),
    "Hydroquinone": ("Oc1ccc(O)cc1", ("P14679",)),
    "Salicylic_acid": ("O=C(O)c1ccccc1O", ("P23219", "P35354")),
    "Resveratrol": (
        "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
        ("Q96EB6", "P35869"),
    ),
    "EGCG": (
        "O=C(O[C@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1",
        ("P14780", "P08253", "P26358", "P14679"),
    ),
    "Caffeine": (
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        ("P29274", "P29275", "P30542", "P0DMS8", "Q14432"),
    ),
    "Adenosine": (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
        ("P30542", "P29274", "P29275"),
    ),
    "Capsaicin": (
        "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
        ("Q8NER1",),
    ),
    "Cinnamaldehyde": ("O=CC=Cc1ccccc1", ("O75762",)),
    "Allyl_isothiocyanate": ("C=CCN=C=S", ("O75762",)),
    "Menthol": ("CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", ("Q7Z2W7",)),
    # --- v3 additions (2026-08-31) -----------------------------------------
    # Seven compounds chosen to make the panel able to arbitrate: 32 -> 46
    # pairs, which is past the 44 needed for a 0.125 effect to reach p<=0.05.
    # Selected on established topical use and literature-confirmed human
    # targets, on pharmacology the v2 panel did not cover, and explicitly
    # WITHOUT consulting how any recipe scores them.
    #
    # The first draft of this block had tranexamic acid as the cis isomer.
    # scripts/tests/test_panel_structures.py caught it by checking every added
    # structure against the ChEMBL mirror - the same class of error that made v1
    # wrong, found before it reached a measurement anybody quoted.
    "Hydrocortisone": (
        "C[C@]12C[C@H](O)[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@]2(O)C(=O)CO",
        ("P04150", "P08235"),
    ),
    "Finasteride": (
        "CC(C)(C)NC(=O)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4NC(=O)C=C[C@]4(C)[C@H]3CC[C@]12C",
        ("P31213", "P18405"),
    ),
    "Crisaborole": (
        "N#Cc1ccc(Oc2ccc3c(c2)COB3O)cc1",
        ("Q07343", "P27815", "Q08499"),
    ),
    "Ruxolitinib": (
        "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
        ("P23458", "O60674"),
    ),
    "Genistein": (
        "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12",
        ("P03372", "Q92731", "P00533"),
    ),
    "Diphenhydramine": ("CN(C)CCOC(c1ccccc1)c1ccccc1", ("P35367",)),
    "Tranexamic_acid": ("NC[C@H]1CC[C@H](C(=O)O)CC1", ("P00747",)),
}

ALPHAFOLD_HUMAN_V4_SOURCE: dict[str, str] = {
    "name": "AlphaFold Protein Structure Database human proteome",
    "version": "v4",
    "proteome_id": "UP000005640",
    "taxon_id": "9606",
    "archive_url": (
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/"
        "UP000005640_9606_HUMAN_v4.tar"
    ),
    "license": "CC BY 4.0",
    "license_url": "https://alphafold.ebi.ac.uk/faq",
}
EXPECTED_ALPHAFOLD_BASE_TARGET_COUNT = 20_171
EXPECTED_SCREENABLE_TARGET_COUNT = 20_204


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_known_panel(path: Path) -> dict[str, Any]:
    """Fail closed unless *path* is the exact preregistered 22-case panel."""

    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Frozen known-target panel is missing or empty: {path}")
    observed_sha = _sha256(path)
    if observed_sha != FROZEN_KNOWN_PANEL_SHA256:
        raise SystemExit(
            "Frozen known-target panel sha256 mismatch: "
            f"{observed_sha} != {FROZEN_KNOWN_PANEL_SHA256} "
            f"(expected {FROZEN_KNOWN_PANEL_CONTRACT}; observed digest is "
            # There is more than one superseded version now, so name the one seen
            # rather than reporting every old artifact as v1.
            + next(
                (
                    f"the superseded {name}"
                    for name, digest in SUPERSEDED_KNOWN_PANEL_CONTRACTS.items()
                    if digest == observed_sha
                ),
                "not a recognised panel version",
            )
            + ")"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "smiles", "known_targets"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise SystemExit(f"Frozen known-target panel missing columns: {missing}")
        rows = list(reader)
    if len(rows) != len(FROZEN_KNOWN_PANEL_CASES):
        raise SystemExit(
            "Frozen known-target panel row count changed: "
            f"{len(rows)} != {len(FROZEN_KNOWN_PANEL_CASES)}"
        )
    observed_cases: dict[str, tuple[str, tuple[str, ...]]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id or case_id in observed_cases:
            raise SystemExit("Frozen known-target panel case IDs must be unique and nonblank")
        smiles = str(row.get("smiles") or "").strip()
        targets = tuple(
            part.strip()
            for part in str(row.get("known_targets") or "").split(";")
            if part.strip()
        )
        observed_cases[case_id] = (smiles, targets)
    if observed_cases != FROZEN_KNOWN_PANEL_CASES:
        raise SystemExit(
            "Frozen known-target panel case/SMILES/target contract changed"
        )
    return {
        "contract": FROZEN_KNOWN_PANEL_CONTRACT,
        "sha256": observed_sha,
        "rows": len(rows),
        "case_ids": list(FROZEN_KNOWN_PANEL_CASES),
        "target_pair_count": sum(
            len(targets) for _, targets in FROZEN_KNOWN_PANEL_CASES.values()
        ),
    }
