"""`daina_recipe_scoring: true` has to reach the command line, not just the file.

The flag and `daina_recipe_index_dir` sat in `workflow/config.yaml` for a day
with a comment saying Stage 3 scored with the promoted recipe. Nothing read
them: `grep -rn recipe scripts/stage3_daina_zoete.py` returned nothing, and the
only caller of `stage3_recipe_scoring` was the measurement harness. A promoted
recipe no run applies is the same failure as no promotion at all, and it is
invisible unless something asserts the wiring end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / "scripts" / "stage3_daina_zoete.py"
RULES = ROOT / "workflow" / "rules" / "stage3b_fast.smk"
CONFIG = ROOT / "workflow" / "config.yaml"


def _docking() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["docking"]


def _sdf(tmp_path: Path, smiles: str) -> Path:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=20260831)
    path = tmp_path / "ligand.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STAGE3), *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def test_the_rule_passes_the_recipe_when_the_config_turns_it_on() -> None:
    """The contract that was missing. Reading the flag is not using it."""
    text = RULES.read_text(encoding="utf-8")

    assert "DAINA_RECIPE_SCORING" in text
    assert 'DOCKING.get("daina_recipe_scoring"' in text
    # The flag has to become an argument...
    assert "--recipe " in text and "--recipe-index-dir " in text
    assert "--operational-gate " in text
    # ...and that argument has to reach the shell block.
    shell = text.split("scripts/stage3_daina_zoete.py", 1)[1].split('"""', 1)[0]
    assert "{params.recipe_arg}" in shell


def test_the_index_is_a_declared_input_so_the_dag_sees_it() -> None:
    """Otherwise Snakemake would run Stage 3 against an index that a concurrent
    rebuild is halfway through writing."""
    text = RULES.read_text(encoding="utf-8")

    assert "DAINA_RECIPE_INPUTS" in text
    assert "recipe_files = DAINA_RECIPE_INPUTS" in text
    for name in ("ligands.parquet", "edges.parquet", "manifest.json"):
        assert name in text, name


def test_the_configured_recipe_and_index_actually_exist() -> None:
    docking = _docking()
    if not docking.get("daina_recipe_scoring"):
        pytest.skip("recipe scoring is off")

    recipe = ROOT / docking["daina_recipe_path"]
    index = ROOT / docking["daina_recipe_index_dir"]
    assert recipe.exists(), recipe
    for name in ("ligands.parquet", "edges.parquet", "manifest.json"):
        assert (index / name).exists(), index / name


def test_half_the_switch_is_refused(tmp_path: Path) -> None:
    """A recipe with no index would silently fall back to nearest neighbour."""
    result = _run(
        "--ligand-sdf", str(tmp_path / "absent.sdf"),
        "--chembl-fp", str(tmp_path / "absent.parquet"),
        "--out-scores", str(tmp_path / "out.tsv"),
        "--recipe", str(tmp_path / "recipe.json"),
    )
    assert result.returncode != 0
    assert "must be given together" in result.stderr


def test_a_contradictory_scoring_method_is_refused(tmp_path: Path) -> None:
    """`--scoring-method` aggregates the mirror; the recipe replaces that step.
    Accepting both would leave a run recording a method it did not use."""
    result = _run(
        "--ligand-sdf", str(tmp_path / "absent.sdf"),
        "--chembl-fp", str(tmp_path / "absent.parquet"),
        "--out-scores", str(tmp_path / "out.tsv"),
        "--recipe", str(tmp_path / "recipe.json"),
        "--recipe-index-dir", str(tmp_path),
        "--scoring-method", "quality-hybrid",
    )
    assert result.returncode != 0
    assert "would be ignored" in result.stderr


def test_an_evaluation_index_is_refused(tmp_path: Path) -> None:
    """The mistake that cost alpha-arbutin its tyrosinase hit. An evaluation
    index is built from the benchmark's train split, so a production run against
    it sees strictly less than the mirror it replaced."""
    index = tmp_path / "index"
    index.mkdir()
    for name in ("ligands.parquet", "edges.parquet"):
        (index / name).touch()
    (index / "manifest.json").write_text(
        json.dumps({"schema_version": "x", "index_role": "evaluation"}), encoding="utf-8"
    )
    result = _run(
        "--ligand-sdf", str(_sdf(tmp_path, "CCO")),
        "--chembl-fp", str(tmp_path / "absent.parquet"),
        "--out-scores", str(tmp_path / "out.tsv"),
        "--recipe", str(tmp_path / "recipe.json"),
        "--recipe-index-dir", str(index),
        "--operational-gate", str(ROOT / "data/manifests/activity_retrieval_operational_gate.flag"),
    )
    assert result.returncode != 0
    assert "must be a production index" in result.stderr
    assert "train-only" in result.stderr


def test_an_incomplete_index_names_the_missing_file(tmp_path: Path) -> None:
    index = tmp_path / "index"
    index.mkdir()
    result = _run(
        "--ligand-sdf", str(_sdf(tmp_path, "CCO")),
        "--chembl-fp", str(tmp_path / "absent.parquet"),
        "--out-scores", str(tmp_path / "out.tsv"),
        "--recipe", str(tmp_path / "recipe.json"),
        "--recipe-index-dir", str(index),
        "--operational-gate", str(ROOT / "data/manifests/activity_retrieval_operational_gate.flag"),
    )
    assert result.returncode != 0
    assert "index is incomplete" in result.stderr
    assert "ligands.parquet" in result.stderr


def test_retrieval_mode_excludes_nothing(tmp_path: Path) -> None:
    """The semantic that must not change when the recipe goes in.

    Stage 3's `retrieval` mode keeps a compound's own measurements in view, and
    the overlay labels them as lookup. The recipe scorer takes a similarity
    threshold, and the nearest legal value - 1.0 - still drops an exact
    self-match, so 'exclude nothing' has to be its own state rather than a
    number. A run that quietly stopped telling a researcher their ingredient is
    already known would look like a ranking change.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    sys.path.insert(0, str(ROOT / "scripts"))
    from activity_retrieval_scoring import score_query_features

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    class _Reference:
        pass

    reference = _Reference()
    reference.fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles("CCO"))]
    reference.edge_ligand = np.array([0])
    reference.edge_target = np.array([0])
    reference.target_ids = np.array(["P11111"])
    reference.edge_max_pactivity = np.array([7.0])
    reference.edge_positive_count = np.array([1.0])
    reference.edge_negative_count = np.array([0.0])
    reference.edge_chembl_max = np.array([7.0])
    reference.edge_bindingdb_max = np.array([-np.inf])
    reference.edge_gtopdb_max = np.array([-np.inf])
    reference.edge_publication_count = np.array([1.0])
    reference.edge_source_count = np.array([1.0])
    reference.edge_measurement_count = np.array([1.0])

    query = generator.GetFingerprint(Chem.MolFromSmiles("CCO"))  # identical to the reference

    kept, _ = score_query_features(reference, query, exclude_reference_similarity=None)
    dropped, _ = score_query_features(reference, query, exclude_reference_similarity=1.0)

    assert kept["max_union_any"][0] == pytest.approx(1.0)
    assert dropped["max_union_any"][0] == 0.0, "1.0 still filters the self-match"


def test_a_real_run_produces_what_the_next_rule_consumes(tmp_path: Path) -> None:
    """End to end on the configured index, because the failure this guards was a
    schema mismatch nobody would notice until a run was half done."""
    import pandas as pd

    docking = _docking()
    if not docking.get("daina_recipe_scoring"):
        pytest.skip("recipe scoring is off")
    index = ROOT / docking["daina_recipe_index_dir"]
    recipe = ROOT / docking["daina_recipe_path"]
    gate = ROOT / "data/manifests/activity_retrieval_operational_gate.flag"
    if not (index / "edges.parquet").exists() or not gate.exists():
        pytest.skip("the configured production retrieval artifacts are not built here")
    scores = tmp_path / "daina.tsv"
    result = _run(
        "--ligand-sdf", str(_sdf(tmp_path, "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O")),
        "--chembl-fp", str(ROOT / "data" / "chembl37" / "fp_morgan2_2048.parquet"),
        "--out-scores", str(scores),
        "--out-metadata-json", str(tmp_path / "meta.json"),
        "--recipe", str(recipe),
        "--recipe-index-dir", str(index),
        "--operational-gate", str(gate),
    )
    assert result.returncode == 0, result.stderr

    frame = pd.read_csv(scores, sep="\t")
    assert {"target_id", "max_tanimoto", "evidence_count", "supporting_molecule_id", "score"} <= set(
        frame.columns
    )
    # alpha-arbutin is in the reference set, and `retrieval` keeps it there, so
    # its own measurement has to come back at 1.0. Anything less means the query
    # and the references are not the same kind of molecule again.
    assert frame["max_tanimoto"].max() == pytest.approx(1.0)
    assert frame["scoring_method"].iloc[0].startswith("recipe:")

    # A reader looks this up by hand; the index's internal "<key>#SMILES-<hash>"
    # is not something to hand them.
    supporting = frame["supporting_molecule_id"].iloc[0]
    assert "#" not in supporting, supporting
    assert len(supporting) == 27 and supporting.count("-") == 2, supporting

    # And the next rule in the DAG has to accept the file unchanged.
    select = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "stage3_select_daina.py"),
            "--daina-scores", str(scores),
            "--daina-metadata", str(tmp_path / "meta.json"),
            "--top-n", "256",
            "--out-csv", str(tmp_path / "selected.csv"),
            "--out-compat-csv", str(tmp_path / "compat.csv"),
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert select.returncode == 0, select.stderr
    assert len(pd.read_csv(tmp_path / "selected.csv")) == 256


def test_the_run_says_whether_its_nearest_evidence_was_active(tmp_path: Path) -> None:
    """The rank weights similarity, not what the analogue's measurement said.

    9.9% of index pairs carry only measurements at or below the activity
    threshold, so a target can rank high on an analogue whose own number was
    weak. Excluding those from the score measured worse on the panel
    (docs/RECIPE_RUNPATH_MEASURED_20260831.md section 7), so the artifact has to
    carry the label instead of the ranking silently absorbing it.

    alpha-arbutin is the case that makes it concrete: tyrosinase is rank 1 and
    its own measurement sits below the threshold. That is a real, weak
    inhibition - which is why the label is named for the threshold and not
    "inactive".
    """
    import pandas as pd

    docking = _docking()
    index = ROOT / docking["daina_recipe_index_dir"]
    gate = ROOT / "data/manifests/activity_retrieval_operational_gate.flag"
    if not docking.get("daina_recipe_scoring") or not (index / "edges.parquet").exists() or not gate.exists():
        pytest.skip("the configured production index is not built here")
    scores = tmp_path / "daina.tsv"
    result = _run(
        "--ligand-sdf", str(_sdf(tmp_path, "OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O")),
        "--chembl-fp", str(ROOT / "data" / "chembl37" / "fp_morgan2_2048.parquet"),
        "--out-scores", str(scores),
        "--out-metadata-json", str(tmp_path / "meta.json"),
        "--recipe", str(ROOT / docking["daina_recipe_path"]),
        "--recipe-index-dir", str(index),
        "--operational-gate", str(gate),
    )
    assert result.returncode == 0, result.stderr

    frame = pd.read_csv(scores, sep="\t")
    assert "supporting_evidence" in frame.columns
    assert set(frame["supporting_evidence"]) <= {
        "at_or_above_threshold", "between_thresholds", "below_threshold", "none",
    }
    # All three states occur, so the column is a real signal and not a constant.
    assert frame["supporting_evidence"].nunique() >= 3

    # And it survives the next rule, which is where a reader meets it.
    select = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "stage3_select_daina.py"),
            "--daina-scores", str(scores),
            "--daina-metadata", str(tmp_path / "meta.json"),
            "--top-n", "256",
            "--out-csv", str(tmp_path / "selected.csv"),
            "--out-compat-csv", str(tmp_path / "compat.csv"),
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert select.returncode == 0, select.stderr
    selected = pd.read_csv(tmp_path / "selected.csv")
    assert "daina_supporting_evidence" in selected.columns
    assert selected["daina_supporting_evidence"].iloc[0] != "unknown"


def test_the_mirror_path_reports_unknown_rather_than_implying_it_checked() -> None:
    """The ChEMBL mirror carries no labels. Defaulting to anything else would
    say the evidence was checked and found positive."""
    source = (ROOT / "scripts" / "stage3_select_daina.py").read_text(encoding="utf-8")

    block = source.split("daina_supporting_evidence", 1)[1][:400]
    assert '"unknown"' in block, block[:200]
