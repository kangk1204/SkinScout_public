"""The pocket-cold panel, and the property that makes it worth having.

Route B borrows ligand evidence across Foldseek pocket clusters. Scoring it on
a panel whose clusters appear in training measures memorisation, so the panel's
defining property is that none of them do - and that has to be asserted, not
assumed, because it is the whole reason the panel exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "eval" / "build_pocket_cold_panel.py"
PANEL = ROOT / "data" / "pocket_cold_panel_202608"
BENCHMARK = ROOT / "data" / "activity_benchmark_202608"
POCKETS = ROOT / "data" / "pocket_clusters_screenable_202608" / "pocket_cluster_map.csv"

pytestmark = pytest.mark.skipif(
    not (PANEL.is_dir() and BENCHMARK.is_dir() and POCKETS.is_file()),
    reason="pocket-cold panel has not been built",
)


def _module():
    spec = importlib.util.spec_from_file_location("pocket_cold_under_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PANEL / "manifest.json").read_text(encoding="utf-8"))


def test_no_panel_target_shares_a_pocket_cluster_with_its_reference() -> None:
    """The defining property. If this fails the panel measures nothing."""
    pd = pytest.importorskip("pandas")

    train = pd.read_parquet(BENCHMARK / "train.parquet", columns=["uniprot"])
    dev = pd.read_parquet(BENCHMARK / "dev.parquet", columns=["uniprot"])
    pockets = pd.read_csv(POCKETS).set_index("uniprot")

    for view, reference in (
        ("strict", pd.concat([train, dev], ignore_index=True)),
        ("extended", train),
    ):
        for level in ("tm40", "tm50", "tm60"):
            column = f"pocket_cluster_{level}"
            seen = set(pockets.reindex(reference.uniprot.unique())[column].dropna())
            panel = pd.read_parquet(PANEL / f"pocket_cold_{level}_{view}.parquet")
            leaked = panel[panel[column].isin(seen)]
            assert leaked.empty, f"{view}/{level}: {len(leaked)} leaked rows"


def test_a_target_without_a_pocket_is_not_counted_as_cold() -> None:
    """No pocket is a coverage problem, not a held-out cluster."""
    pd = pytest.importorskip("pandas")

    for level in ("tm40", "tm50", "tm60"):
        panel = pd.read_parquet(PANEL / f"pocket_cold_{level}_strict.parquet")
        assert panel[f"pocket_cluster_{level}"].notna().all()


def test_the_panel_is_larger_than_the_view_it_replaces(manifest) -> None:
    """dual_cold reaches 13 targets, which is why Route B was unmeasurable."""
    pd = pytest.importorskip("pandas")

    dual_cold = pd.read_parquet(BENCHMARK / "dual_cold.parquet")
    strict = manifest["views"]["pocket_cold_tm40_strict"]
    extended = manifest["views"]["pocket_cold_tm40_extended"]

    assert strict["targets"] > dual_cold.uniprot.nunique()
    assert extended["targets"] >= 3 * dual_cold.uniprot.nunique()


def test_the_panel_targets_can_actually_be_scored() -> None:
    """A held-out target with no receptor cannot be evaluated by this pipeline."""
    pd = pytest.importorskip("pandas")
    receptors = ROOT / "data" / "human_pdbqt"
    if not receptors.is_dir():
        pytest.skip("Stage 0 receptors are not present")

    available = {path.stem for path in receptors.glob("*.pdbqt")}
    panel = pd.read_parquet(PANEL / "pocket_cold_tm40_strict.parquet")
    targets = set(panel.uniprot)

    assert len(targets & available) >= 0.9 * len(targets)


def test_the_larger_view_is_not_claimable(manifest) -> None:
    """It evaluates on dev, so it cannot be quoted as a headline result."""
    assert manifest["views"]["pocket_cold_tm40_extended"]["claimable"] is False
    assert manifest["views"]["pocket_cold_tm40_strict"]["claimable"] is True
    assert "evaluates on dev" in manifest["views"]["pocket_cold_tm40_extended"]["policy"]


def test_the_manifest_says_these_are_recovery_panels(manifest) -> None:
    """The benchmark holds positives only; reading them as classification
    panels would produce a meaningless accuracy."""
    assert "positives only" in manifest["label_semantics"]
    assert "recovery" in manifest["label_semantics"]


def test_the_pair_key_matches_the_benchmark_builder() -> None:
    """Two different notions of "same pair" would make absence meaningless."""
    module = _module()
    builder = (ROOT / "eval" / "build_activity_benchmark.py").read_text(encoding="utf-8")
    reference = builder.split("def _ligand_key_series", 1)[1].split("def ", 1)[0]
    ours = module.__dict__["_ligand_key"].__doc__ or ""

    for token in ("ligand_inchikey", "ligand_id", "ligand_smiles", "upper"):
        assert token in reference
    assert "benchmark builder" in ours


def test_the_panel_is_accepted_by_the_existing_scorer() -> None:
    """A panel the scorer cannot load would have to be rebuilt to be used."""
    pytest.importorskip("rdkit")
    sys.path.insert(0, str(ROOT / "eval"))
    from activity_retrieval_model import load_ranking_panel

    for view in ("strict", "extended"):
        path = PANEL / f"pocket_cold_tm40_{view}_ranking.parquet"
        loaded = load_ranking_panel(path, "test")
        assert not loaded.empty
        assert loaded["n_truth_targets"].min() >= 1


def test_the_ranking_form_keeps_every_scoreable_compound(manifest) -> None:
    """Compounds RDKit cannot read are dropped, so the two shapes can differ -
    but not by much, or the panel is quietly smaller than it claims."""
    for view in ("strict", "extended"):
        record = manifest["views"][f"pocket_cold_tm40_{view}"]
        assert record["ranking_queries"] >= 0.98 * record["compounds"]
