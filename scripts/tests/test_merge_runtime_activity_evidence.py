"""Merging other sources into the run path must not corrupt what retrieval reads.

Stage 3 scores a query against `data/chembl37/human_activities.parquet`. Adding
BindingDB and GtoPdb to it reaches 248 targets a run cannot reach today - but the
same merge could quietly break retrieval in three ways, so each has a test:

* a censored measurement (">10 uM", i.e. did not bind) becoming a potency;
* a structure ChEMBL already has entering twice, so one molecule looks like two
  independent pieces of evidence for the same target;
* a ligand id colliding with a real CHEMBL accession.

The licence tier is also a decision, not a default: BindingDB's own curation is
CC BY 4.0, while GtoPdb and BindingDB's ChEMBL-derived half carry share-alike.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "merge_runtime_activity_evidence.py"


def _chembl_dir(tmp_path: Path) -> Path:
    """A miniature run-path table: one ligand, one target."""
    directory = tmp_path / "chembl"
    directory.mkdir()
    base = pd.DataFrame(
        [
            {
                "molecule_chembl_id": "CHEMBL1",
                "uniprot": "P11111",
                "smiles": "CCO",
                "standard_inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "act_type": "IC50",
                "act_value": 100.0,
                "act_units": "nM",
                "pchembl": 7.0,
                "pubmed_id": "123",
            }
        ]
    )
    base.to_parquet(directory / "human_activities.parquet", index=False)
    pd.DataFrame(
        [{"molecule_chembl_id": "CHEMBL1", "smiles": "CCO", "bitvec": ["0"] * 32}]
    ).to_parquet(directory / "fp_morgan2_2048.parquet", index=False)
    return directory


def _bindingdb(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "bindingdb.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _row(**overrides) -> dict:
    row = {
        "ligand_id": "500",
        "ligand_inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N",
        "ligand_smiles": "CCCO",
        "uniprot": "Q22222",
        "affinity_type": "IC50",
        "affinity_value": "250.0",
        "affinity_unit": "nM",
        "relation": "=",
        "censor": "False",
        "source_pmid": "123",
        "evidence_date": "2020-01-01",
        "source_db": "BindingDB",
        "source_release": "2026-08",
        "source_license": "CC BY 4.0 (BindingDB-curated)",
        "chembl_derived_license_flag": False,
    }
    row.update(overrides)
    return row


def _run(tmp_path: Path, chembl: Path, bindingdb: Path, *extra: str):
    out = tmp_path / "merged"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--chembl-dir",
            str(chembl),
            "--bindingdb",
            str(bindingdb),
            "--gtopdb",
            str(tmp_path / "absent.parquet"),
            "--out-dir",
            str(out),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, out


def test_a_new_target_is_reached(tmp_path: Path) -> None:
    chembl = _chembl_dir(tmp_path)
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, [_row()]))

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    assert set(merged["uniprot"]) == {"P11111", "Q22222"}
    # And the added ligand carries a key that cannot be mistaken for ChEMBL's.
    added = merged[merged["uniprot"] == "Q22222"]
    assert added["molecule_chembl_id"].iloc[0] == "BDB:500"


def test_a_censored_measurement_does_not_become_a_potency(tmp_path: Path) -> None:
    """">50000 nM" means it did not bind. Scoring it as 4.3 would invert the fact."""
    chembl = _chembl_dir(tmp_path)
    rows = [_row(relation=">", censor="True", affinity_value="50000.0")]
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    added = merged[merged["uniprot"] == "Q22222"]
    assert len(added) == 1
    assert pd.isna(added["pchembl"].iloc[0]), "a censored row must not carry a pActivity"


def test_a_measurement_chembl_already_has_is_not_added_twice(tmp_path: Path) -> None:
    """One measurement must not read as two independent sources.

    The index counts distinct source databases per (ligand, target) edge, so a
    row BindingDB re-imported from ChEMBL becomes corroboration of itself.
    Measured before this was keyed correctly: 151,552 pairs carried both a
    ChEMBL and a BindingDB edge and 95.4% agreed to within 0.01 pActivity.
    """
    chembl = _chembl_dir(tmp_path)
    # Same structure, same target as the ChEMBL fixture row (CCO / P11111).
    rows = [
        _row(ligand_smiles="CCO", uniprot="P11111", affinity_value="100.0")
    ]
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    assert len(merged) == 1, "the re-import must not survive"
    assert not merged["molecule_chembl_id"].astype(str).str.startswith("BDB:").any()


def test_the_same_structure_against_a_new_target_is_kept(tmp_path: Path) -> None:
    """The reason the key is the pair and not the ligand.

    Deduplicating on the structure alone also discards BindingDB measurements
    for targets ChEMBL never measured that ligand against - which is most of
    what merging BindingDB is for. It cost 10 reachable targets when tried.
    """
    chembl = _chembl_dir(tmp_path)
    rows = [_row(ligand_smiles="CCO", uniprot="Q33333")]
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    assert "Q33333" in set(merged["uniprot"])


def test_independent_publication_for_same_structure_target_is_kept(tmp_path: Path) -> None:
    chembl = _chembl_dir(tmp_path)
    rows = [
        _row(
            ligand_smiles="CCO",
            uniprot="P11111",
            source_pmid="999",
            affinity_value="250.0",
        )
    ]
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    assert len(merged) == 2
    assert merged["molecule_chembl_id"].astype(str).str.startswith("BDB:").any()


def test_a_stereo_blind_inchikey_does_not_defeat_the_deduplication(tmp_path: Path) -> None:
    """Why identity comes from the structure rather than the key string.

    BindingDB computes its InChIKeys largely without stereo perception: 859,200
    of its 897,444 rows carry the stereo-free UHFFFAOYSA block while ChEMBL's
    carry real stereo. Comparing the strings matched zero rows.
    """
    chembl = _chembl_dir(tmp_path)
    rows = [
        _row(
            ligand_inchikey="ZZZZZZZZZZZZZZ-UHFFFAOYSA-N",  # disagrees with ChEMBL's
            ligand_smiles="CCO",
            uniprot="P11111",
            affinity_value="100.0",
        )
    ]
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    assert len(merged) == 1, "structure identity must win over the key string"


def test_an_unsupported_endpoint_is_dropped_not_coerced(tmp_path: Path) -> None:
    """Retrieval compares pActivity; an MIC has no comparable value."""
    chembl = _chembl_dir(tmp_path)
    rows = [_row(affinity_type="MIC")]
    result, _ = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode != 0
    assert "no rows to add" in result.stderr


def test_a_unit_the_script_does_not_understand_is_dropped(tmp_path: Path) -> None:
    """Guessing a conversion would put a wrong number in the retrieval table."""
    chembl = _chembl_dir(tmp_path)
    rows = [_row(affinity_unit="ug/mL")]
    result, _ = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode != 0
    assert "no rows to add" in result.stderr


def test_the_default_tier_excludes_share_alike_sources(tmp_path: Path) -> None:
    """BindingDB's ChEMBL-derived half is CC BY-SA 3.0; it has to be asked for."""
    chembl = _chembl_dir(tmp_path)
    rows = [
        _row(
            ligand_id="900",
            uniprot="Q44444",
            chembl_derived_license_flag=True,
            source_license="CC BY-SA 3.0 (ChEMBL-derived)",
        )
    ]
    bindingdb = _bindingdb(tmp_path, rows)
    result, _ = _run(tmp_path, chembl, bindingdb)
    assert result.returncode != 0, "the share-alike row must not be added by default"
    assert "no rows to add" in result.stderr

    gtopdb = tmp_path / "absent.parquet"
    pd.DataFrame([_row(ligand_id="1", uniprot="Q55555", source_db="GtoPdb")]).to_parquet(
        gtopdb, index=False
    )
    result, out = _run(tmp_path, chembl, bindingdb, "--sources", "all")
    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    assert "Q44444" in set(merged["uniprot"])


def test_a_requested_source_that_is_missing_fails_with_a_readable_message(
    tmp_path: Path,
) -> None:
    chembl = _chembl_dir(tmp_path)
    result, _ = _run(
        tmp_path, chembl, _bindingdb(tmp_path, [_row()]), "--sources", "gtopdb"
    )

    assert result.returncode != 0
    assert "needs GtoPdb" in result.stderr


def test_the_manifest_records_the_licences_it_pulled_in(tmp_path: Path) -> None:
    chembl = _chembl_dir(tmp_path)
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, [_row()]))

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources_tier"] == "permissive"
    assert manifest["counts"]["targets_added"] == 1
    assert any("CC BY 4.0" in licence for licence in manifest["licences"])
    # Downstream has to be able to see what policy produced this table.
    assert manifest["policy"]["duplicate_measurements"]
    assert manifest["policy"]["censored_measurements"]


def test_added_ligands_get_fingerprints(tmp_path: Path) -> None:
    """Retrieval scores against the fingerprint table; a row with no fingerprint
    is a target the merge claims to add and then cannot reach."""
    chembl = _chembl_dir(tmp_path)
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, [_row()]))

    assert result.returncode == 0, result.stderr
    fingerprints = pd.read_parquet(out / "fp_morgan2_2048.parquet")
    assert set(fingerprints["molecule_chembl_id"]) == {"CHEMBL1", "BDB:500"}
    added = fingerprints[fingerprints["molecule_chembl_id"] == "BDB:500"]
    assert len(added["bitvec"].iloc[0]) == 32


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    chembl = _chembl_dir(tmp_path)
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, [_row()]), "--dry-run")

    assert result.returncode == 0, result.stderr
    # A dry run reports the upper bound and labels it, because fingerprinting -
    # the step that decides the final count - has not run yet.
    assert "targets addable" in result.stdout
    assert "before fingerprinting" in result.stdout
    assert not out.exists()


def test_the_workflow_can_point_retrieval_at_the_merged_table_but_does_not() -> None:
    """The switch exists and is off, and the reason is written down.

    Turning it on reaches 248 more targets and invalidates every recovery number
    in the guide, all of which were measured against ChEMBL alone. That trade has
    to be a decision someone makes, with the panel re-run, not a default that
    quietly moves the numbers the README quotes.
    """
    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    docking = config["docking"]

    assert docking["daina_evidence_dir"] == "data/chembl37", "must default to ChEMBL alone"

    rules = (ROOT / "workflow" / "rules" / "stage3b_fast.smk").read_text(encoding="utf-8")
    assert "DAINA_EVIDENCE_DIR" in rules
    assert "daina_evidence_dir" in rules
    # The caveat travels with the switch.
    assert "re-measured" in rules or "re-run" in rules


def test_a_row_whose_ligand_cannot_be_fingerprinted_is_dropped(tmp_path: Path) -> None:
    """Retrieval scores against the fingerprint table.

    A row whose ligand has no fingerprint is an edge nothing can ever match, so
    leaving it in inflates both the row count and the target count while reaching
    neither. The first real merge left 843 such rows behind before this.
    """
    chembl = _chembl_dir(tmp_path)
    rows = [
        _row(),
        _row(
            ligand_id="777",
            uniprot="Q66666",
            ligand_smiles="this is not a molecule",
            ligand_inchikey="AAAAAAAAAAAAAA-UHFFFAOYSA-N",
        ),
    ]
    result, out = _run(tmp_path, chembl, _bindingdb(tmp_path, rows))

    assert result.returncode == 0, result.stderr
    merged = pd.read_parquet(out / "human_activities.parquet")
    fingerprints = pd.read_parquet(out / "fp_morgan2_2048.parquet")

    assert "BDB:777" not in set(merged["molecule_chembl_id"])
    assert "Q66666" not in set(merged["uniprot"]), "must not claim an unreachable target"
    # Every molecule in the table has a fingerprint to be scored against.
    assert set(merged["molecule_chembl_id"]) <= set(fingerprints["molecule_chembl_id"])

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["rows_dropped_without_fingerprint"] == 1


# Two tiers get built here. The permissive one is what ships: BindingDB curates
# its own rows under CC BY 4.0, which LICENSE_POLICY.md admits. The other adds
# GtoPdb (ODbL 1.0 / CC BY-SA 4.0) for 33 further targets, and share-alike is an
# obligation nobody asked for, so it stays opt-in.
TIERS = (
    ROOT / "data" / "runtime_evidence_permissive_202608",
    ROOT / "data" / "runtime_evidence_merged_202608",
)
BUILT = [path for path in TIERS if (path / "human_activities.parquet").exists()]


@pytest.mark.skipif(not BUILT, reason="no merged evidence table is built here")
@pytest.mark.parametrize("merged", BUILT, ids=[path.name for path in BUILT])
def test_the_built_table_loads_through_the_run_paths_own_loaders(merged: Path) -> None:
    """The proof that matters: Stage 3's loaders accept it, not a hand check.

    `validate_reference_overlap` is the gate that would reject a table whose
    activity rows and fingerprints disagree, which is exactly the failure the
    orphaned-ligand bug produced.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from stage3_daina_zoete import (
        apply_quality_policy,
        load_activities,
        load_fingerprints,
        validate_reference_overlap,
    )

    fingerprints = load_fingerprints(merged / "fp_morgan2_2048.parquet")
    activities = load_activities(
        merged / "human_activities.parquet",
        quality_policy="legacy",
        evidence_mode="retrieval",
    )
    activities, _ = apply_quality_policy(activities, "legacy")
    validate_reference_overlap(activities, fingerprints, merged / "human_activities.parquet")

    # Against the manifest rather than a literal: the number moves with the tier,
    # and a stale literal would either fail on the wrong tier or stop checking.
    manifest = json.loads((merged / "manifest.json").read_text(encoding="utf-8"))
    assert activities["uniprot"].nunique() == manifest["counts"]["targets_after"]
    # And the merge is only worth doing if it actually moves.
    assert manifest["counts"]["targets_after"] > manifest["counts"]["targets_before"]


@pytest.mark.skipif(
    not (TIERS[0] / "manifest.json").exists(), reason="permissive tier is not built here"
)
def test_the_tier_that_ships_adds_only_permissive_rows_and_discloses_the_base() -> None:
    """`LICENSE_POLICY.md` admits CC-BY as commercial-use-permissive and says
    nothing about share-alike. The index a run reads by default is built from
    this tier, so a share-alike licence for an *added* source would put an
    obligation into every run without anyone choosing it.

    D05: the shipped manifest must also disclose the licence of the ChEMBL base
    it merges into, because that base is 81.7% of the evidence and its CC BY-SA
    obligation exists whether or not the manifest names it. So the union records
    CC BY-SA 3.0 for ChEMBL; what the tier guards is the added source, which
    stays `bindingdb_own` with its own CC BY 4.0 rows.
    """
    manifest = json.loads((TIERS[0] / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["sources_tier"] == "permissive"
    assert manifest["sources"] == ["bindingdb_own"]
    assert "CC BY-SA 3.0" in manifest["licences"], (
        "the base ChEMBL share-alike obligation must be disclosed, not hidden"
    )
    added = [licence for licence in manifest["licences"] if "bindingdb" in licence.lower()]
    assert added == ["CC BY 4.0 (BindingDB-curated)"]
    for licence in added:
        assert "SA" not in licence and "ODbL" not in licence, licence
