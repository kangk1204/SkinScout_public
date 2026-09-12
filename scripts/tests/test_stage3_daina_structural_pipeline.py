from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, *args: object, env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_daina_selection_is_fixed_and_not_a_probability(tmp_path: Path) -> None:
    scores = tmp_path / "daina.tsv"
    pd.DataFrame(
        {
            "target_id": [f"P{index:03d}" for index in range(300)],
            "max_tanimoto": [1.0 - index / 1000 for index in range(300)],
            "evidence_count": [index + 1 for index in range(300)],
            "score": [1.0 - index / 1000 for index in range(300)],
            "supporting_molecule_id": [f"CHEMBL{index}" for index in range(300)],
            "quality_policy": ["legacy"] * 300,
        }
    ).to_csv(scores, sep="\t", index=False)
    metadata = tmp_path / "daina.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.daina-run.v1",
                "evidence_mode": "retrieval",
                "quality_policy": "legacy",
                "scoring_method": "max-similarity",
                "score_is_calibrated_probability": False,
            }
        )
    )
    selected = tmp_path / "selected.csv"
    compat = tmp_path / "compat.csv"

    result = _run(
        "stage3_select_daina.py",
        "--daina-scores",
        scores,
        "--daina-metadata",
        metadata,
        "--top-n",
        256,
        "--out-csv",
        selected,
        "--out-compat-csv",
        compat,
    )

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(selected)
    assert len(frame) == 256
    assert frame["target_id"].iloc[[0, -1]].tolist() == ["P000", "P255"]
    assert frame["daina_rank"].tolist() == list(range(1, 257))
    assert set(frame["daina_score_is_probability"].astype(str).str.lower()) == {
        "false"
    }
    projected = pd.read_csv(compat)
    assert projected["target_id"].tolist() == frame["target_id"].tolist()
    assert set(projected["source_count"]) == {1}
    assert set(projected["sources"]) == {"daina"}


def test_daina_cli_rejects_noncanonical_target_count(tmp_path: Path) -> None:
    result = _run(
        "stage3_select_daina.py",
        "--daina-scores",
        tmp_path / "unused.tsv",
        "--top-n",
        2,
        "--out-csv",
        tmp_path / "selected.csv",
    )

    assert result.returncode != 0
    assert "frozen public contract and must equal 256" in result.stderr


def test_autogrid_manifest_preserves_unavailable_and_failed_rows_and_reuses_cache(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.csv"
    pd.DataFrame(
        {
            "target_id": ["P1", "P2", "P3", "P4"],
            "daina_rank": [1, 2, 3, 4],
            "daina_score": [0.9, 0.8, 0.7, 0.6],
        }
    ).to_csv(selected, index=False)
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("ATOM 1 C\n")
    receptor_dir = tmp_path / "receptors"
    box_dir = tmp_path / "boxes"
    receptor_dir.mkdir()
    box_dir.mkdir()
    for target in ("P1", "P4"):
        (receptor_dir / f"{target}.pdbqt").write_text("ATOM 1 C\n")
        (box_dir / f"{target}.box.txt").write_text(
            "center_x=0\ncenter_y=0\ncenter_z=0\n"
            "size_x=10\nsize_y=10\nsize_z=10\n"
        )
    no_pocket = tmp_path / "no_pocket.list"
    no_pocket.write_text("P2\n")
    parameter = tmp_path / "AD4.1_bound.dat"
    parameter.write_text("parameter\n")
    fail_marker = tmp_path / "fail-if-called"
    fake_autogrid = tmp_path / "autogrid4"
    fake_autogrid.write_text(
        "#!/bin/sh\n"
        f"test -f {fail_marker} && exit 91\n"
        "gpf=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-p' ]; then shift; gpf=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "case \"$gpf\" in P4.gpf) exit 7 ;; esac\n"
        "while read -r kind file rest; do\n"
        "  case \"$kind\" in\n"
        "    gridfld|map|elecmap|dsolvmap) printf 'map\\n' > \"$file\" ;;\n"
        "  esac\n"
        "done < \"$gpf\"\n"
    )
    fake_autogrid.chmod(0o755)
    cache = tmp_path / "cache"

    def run_once(name: str):
        run_dir = tmp_path / name
        run_dir.mkdir()
        return _run(
            "stage3_autogrid_maps.py",
            "--selected-csv",
            selected,
            "--ligand-pdbqt",
            ligand,
            "--receptor-dir",
            receptor_dir,
            "--box-dir",
            box_dir,
            "--no-pocket-list",
            no_pocket,
            "--autogrid-bin",
            fake_autogrid,
            "--parameter-file",
            parameter,
            "--cache-dir",
            cache,
            "--out-map-dir",
            run_dir / "maps",
            "--out-manifest",
            run_dir / "map_manifest.json",
        ), run_dir

    first, first_dir = run_once("first")
    assert first.returncode == 0, first.stderr
    payload = json.loads((first_dir / "map_manifest.json").read_text())
    statuses = {row["target_id"]: row["status"] for row in payload["targets"]}
    assert statuses == {
        "P1": "map_ready",
        "P2": "structural_unavailable_no_pocket",
        "P3": "structural_unavailable_prep",
        "P4": "structure_failed_map",
    }
    p1 = next(row for row in payload["targets"] if row["target_id"] == "P1")
    assert _sha256(first_dir / p1["map_fld"]) == p1["map_fld_sha256"]
    assert payload["autogrid_version"] == "4.2.8+6d2847b"
    assert p1["pocket_id"] == "P1:p2rank_top1"
    assert p1["pocket_source"] == "P2Rank-top1"
    assert p1["structure_source"] == "AlphaFold-human-v4"

    fail_marker.write_text("the cached P1 map must avoid another binary call\n")
    second, second_dir = run_once("second")
    assert second.returncode == 0, second.stderr
    second_payload = json.loads((second_dir / "map_manifest.json").read_text())
    second_statuses = {
        row["target_id"]: row["status"] for row in second_payload["targets"]
    }
    assert second_statuses["P1"] == "map_ready"
    assert second_statuses["P4"] == "structure_failed_map"


def test_gnina_pose_mode_scores_the_exported_pose_with_gpu(tmp_path: Path) -> None:
    top = tmp_path / "autodock.tsv"
    top.write_text("target_id\tvina_score\tneg_vina_score\nP1\t-7.0\t7.0\n")
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    pose = pose_dir / "P1.sdf"
    pose.write_text("P1\n  SkinScout\n\n  0  0  0  0  0  0            999 V2000\nM  END\n$$$$\n")
    pose_manifest = pose_dir / "pose_manifest.json"
    pose_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.docking_pose_manifest.v1",
                "target_count": 1,
                "targets": [
                    {
                        "target_id": "P1",
                        "pose_file": "P1.sdf",
                        "pose_sha256": _sha256(pose),
                    }
                ],
            }
        )
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "P1_clean.pdb").write_text("ATOM\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_log = tmp_path / "gnina.args"
    gnina = fake_bin / "gnina"
    gnina.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {args_log}\n"
        "printf 'CNNaffinity: 2.25\\n'\n"
    )
    gnina.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    scores = tmp_path / "gnina.tsv"
    status = tmp_path / "gnina.json"

    result = _run(
        "stage3_gnina_rescore.py",
        "--top-csv",
        top,
        "--pose-manifest",
        pose_manifest,
        "--pose-dir",
        pose_dir,
        "--clean-dir",
        clean,
        "--use-gpu",
        "--allow-partial-structure",
        "--out-scores",
        scores,
        "--out-status-manifest",
        status,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    arguments = args_log.read_text().splitlines()
    assert "--no_gpu" not in arguments
    assert str(pose) in arguments
    frame = pd.read_csv(scores, sep="\t")
    assert frame.loc[0, "scored_actual_docked_pose"]
    assert frame.loc[0, "pose_sha256"] == _sha256(pose)
    payload = json.loads(status.read_text())
    assert payload["gpu_enabled"] is True
    assert payload["targets"][0]["status"] == "structure_supported"


def test_overlay_keeps_all_daina_rows_and_primary_order(tmp_path: Path) -> None:
    selected = tmp_path / "selected.csv"
    pd.DataFrame(
        {
            "target_id": ["P1", "P2", "P3"],
            "daina_rank": [1, 2, 3],
            "daina_score": [0.91, 0.85, 0.70],
            "daina_score_is_probability": [False, False, False],
            "daina_evidence_mode": ["retrieval"] * 3,
            "daina_quality_policy": ["legacy"] * 3,
            "daina_scoring_method": ["max-similarity"] * 3,
            "daina_max_tanimoto": [0.91, 0.85, 0.70],
            "daina_known_ligand_count": [12, 7, 3],
            "daina_supporting_molecule_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        }
    ).to_csv(selected, index=False)
    map_manifest = tmp_path / "maps.json"
    map_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.autogrid-map-manifest.v1",
                "target_count": 3,
                "targets": [
                    {
                        "target_id": "P1",
                        "status": "map_ready",
                        "map_cache_key": "a",
                        "pocket_id": "P1:p2rank_top1",
                        "pocket_source": "P2Rank-top1",
                        "structure_source": "AlphaFold-human-v4",
                    },
                    {"target_id": "P2", "status": "structural_unavailable_no_pocket"},
                    {"target_id": "P3", "status": "map_ready", "map_cache_key": "c"},
                ],
            }
        )
    )
    docking_status = tmp_path / "docking.json"
    docking_status.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.docking-status-manifest.v1",
                "target_count": 3,
                "targets": [
                    {"target_id": "P1", "status": "docked"},
                    {"target_id": "P2", "status": "structural_unavailable_no_pocket"},
                    {"target_id": "P3", "status": "structure_failed_docking"},
                ],
            }
        )
    )
    docking_scores = tmp_path / "autodock.tsv"
    docking_scores.write_text("target_id\tvina_score\nP1\t-7.2\n")
    gnina_status = tmp_path / "gnina.json"
    gnina_status.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.gnina-pose-status.v1",
                "target_count": 1,
                "targets": [
                    {
                        "target_id": "P1",
                        "status": "structure_supported",
                        "pose_file": "P1.sdf",
                        "pose_sha256": "f" * 64,
                    }
                ],
            }
        )
    )
    gnina_scores = tmp_path / "gnina.tsv"
    gnina_scores.write_text("target_id\tcnn_affinity\nP1\t2.3\n")
    skin = tmp_path / "skin.tsv"
    skin.write_text(
        "uniprot\tskin_score\ttier\tcell_type_preferred\n"
        "P1\t0.8\thigh\tmelanocytes\n"
        "P2\t0.1\tlow\tunknown\n"
        "P3\t0.5\tmedium\tkeratinocytes\n"
    )
    priors = tmp_path / "priors.csv"
    priors.write_text(
        "target_id,prior_score,source_case_id,source_panel,evidence_count,"
        "evidence_case_ids,evidence_panels\n"
        "P2,1.0,case,known,1,case,known\n"
    )
    graph = nx.DiGraph()
    graph.add_node("gene:P1", type="Gene")
    graph.add_node("category:barrier", type="EfficacyCategory", name="barrier")
    graph.add_edge(
        "gene:P1",
        "category:barrier",
        n_papers=2,
        skin_effect_direction="beneficial",
    )
    kg = tmp_path / "kg.graphml"
    nx.write_graphml(graph, kg)
    canonical = tmp_path / "canonical.csv"
    top50 = tmp_path / "top50.csv"

    result = _run(
        "stage3_daina_structural_overlay.py",
        "--selected-csv",
        selected,
        "--map-manifest",
        map_manifest,
        "--docking-scores",
        docking_scores,
        "--docking-status-manifest",
        docking_status,
        "--gnina-scores",
        gnina_scores,
        "--gnina-status-manifest",
        gnina_status,
        "--skin-tsv",
        skin,
        "--skin-kg",
        kg,
        "--known-target-priors",
        priors,
        "--out-csv",
        canonical,
        "--out-top50",
        top50,
    )

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(canonical)
    assert frame["target_id"].tolist() == ["P1", "P2", "P3"]
    assert frame["daina_rank"].tolist() == [1, 2, 3]
    assert frame["final_score"].tolist() == [0.91, 0.85, 0.70]
    assert frame["structural_status"].tolist() == [
        "structure_supported",
        "structural_unavailable_no_pocket",
        "structure_failed_docking",
    ]
    assert frame.loc[0, "skin_kg_supported"]
    assert frame.loc[1, "experimental_supported"]
    assert frame.loc[0, "pocket_id"] == "P1:p2rank_top1"
    structure = json.loads(frame.loc[0, "structure_annotation_json"])
    assert structure["pocket_source"] == "P2Rank-top1"
    assert structure["structure_source"] == "AlphaFold-human-v4"
    assert pd.read_csv(top50)["target_id"].tolist() == ["P1", "P2", "P3"]


def _overlay_fixture(
    tmp_path: Path,
    *,
    scoring_method: str = "max-similarity",
    map_targets: list[dict[str, object]],
    docking_targets: list[dict[str, object]],
    docking_rows: str,
    gnina_targets: list[dict[str, object]],
    gnina_rows: str,
) -> dict[str, Path]:
    """Write a minimal single-target overlay input set."""
    selected = tmp_path / "selected.csv"
    pd.DataFrame(
        {
            "target_id": ["P1"],
            "daina_rank": [1],
            "daina_score": [0.91],
            "daina_score_is_probability": ["false"],
            "daina_scoring_method": [scoring_method],
            "daina_max_tanimoto": [0.91],
            "daina_known_ligand_count": [12],
            "daina_supporting_molecule_id": ["CHEMBL1"],
        }
    ).to_csv(selected, index=False)
    map_manifest = tmp_path / "maps.json"
    map_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.autogrid-map-manifest.v1",
                "target_count": len(map_targets),
                "targets": map_targets,
            }
        )
    )
    docking_status = tmp_path / "docking.json"
    docking_status.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.docking-status-manifest.v1",
                "target_count": len(docking_targets),
                "targets": docking_targets,
            }
        )
    )
    docking_scores = tmp_path / "autodock.tsv"
    docking_scores.write_text(docking_rows)
    gnina_status = tmp_path / "gnina.json"
    gnina_status.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.gnina-pose-status.v1",
                "target_count": len(gnina_targets),
                "targets": gnina_targets,
            }
        )
    )
    gnina_scores = tmp_path / "gnina.tsv"
    gnina_scores.write_text(gnina_rows)
    skin = tmp_path / "skin.tsv"
    skin.write_text(
        "uniprot\tskin_score\ttier\tcell_type_preferred\nP1\t0.8\thigh\tmelanocytes\n"
    )
    priors = tmp_path / "priors.csv"
    priors.write_text("target_id,prior_score\n")
    graph = nx.DiGraph()
    graph.add_node("gene:P1", type="Gene")
    kg = tmp_path / "kg.graphml"
    nx.write_graphml(graph, kg)
    return {
        "selected": selected,
        "map_manifest": map_manifest,
        "docking_status": docking_status,
        "docking_scores": docking_scores,
        "gnina_status": gnina_status,
        "gnina_scores": gnina_scores,
        "skin": skin,
        "priors": priors,
        "kg": kg,
        "out": tmp_path / "canonical.csv",
        "top50": tmp_path / "top50.csv",
    }


def _run_overlay(paths: dict[str, Path]):
    return _run(
        "stage3_daina_structural_overlay.py",
        "--selected-csv",
        paths["selected"],
        "--map-manifest",
        paths["map_manifest"],
        "--docking-scores",
        paths["docking_scores"],
        "--docking-status-manifest",
        paths["docking_status"],
        "--gnina-scores",
        paths["gnina_scores"],
        "--gnina-status-manifest",
        paths["gnina_status"],
        "--skin-tsv",
        paths["skin"],
        "--skin-kg",
        paths["kg"],
        "--known-target-priors",
        paths["priors"],
        "--out-csv",
        paths["out"],
        "--out-top50",
        paths["top50"],
    )


def test_overlay_never_writes_the_literal_string_none(tmp_path: Path) -> None:
    paths = _overlay_fixture(
        tmp_path,
        map_targets=[
            {
                "target_id": "P1",
                "status": "map_ready",
                "map_cache_key": None,
                "pocket_id": None,
                "pocket_source": None,
                "structure_source": None,
            }
        ],
        docking_targets=[{"target_id": "P1", "status": "docked"}],
        docking_rows="target_id\tvina_score\nP1\t-7.2\n",
        gnina_targets=[
            {
                "target_id": "P1",
                "status": "structure_supported",
                "pose_file": None,
                "pose_sha256": None,
            }
        ],
        gnina_rows="target_id\tcnn_affinity\nP1\t2.3\n",
    )

    result = _run_overlay(paths)

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(paths["out"], keep_default_na=False)
    for column in ("pocket_id", "map_cache_key", "pose_file", "pose_sha256"):
        assert frame.loc[0, column] == "", column
    structure = json.loads(frame.loc[0, "structure_annotation_json"])
    for key in ("pocket_id", "pocket_source", "structure_source", "pose_sha256"):
        assert structure[key] is None, key
    assert "None" not in paths["out"].read_text()


def test_overlay_rejects_failed_status_that_still_carries_a_docking_score(
    tmp_path: Path,
) -> None:
    paths = _overlay_fixture(
        tmp_path,
        map_targets=[{"target_id": "P1", "status": "map_ready", "map_cache_key": "a"}],
        docking_targets=[{"target_id": "P1", "status": "structure_failed_docking"}],
        docking_rows="target_id\tvina_score\nP1\t-7.2\n",
        gnina_targets=[{"target_id": "P1", "status": "structure_supported"}],
        gnina_rows="target_id\tcnn_affinity\n",
    )

    result = _run_overlay(paths)

    assert result.returncode != 0
    assert "still carries a docking or GNINA score" in result.stderr
    assert not paths["out"].exists()


def test_overlay_rejects_docked_target_without_an_autodock_score(
    tmp_path: Path,
) -> None:
    paths = _overlay_fixture(
        tmp_path,
        map_targets=[{"target_id": "P1", "status": "map_ready", "map_cache_key": "a"}],
        docking_targets=[{"target_id": "P1", "status": "docked"}],
        docking_rows="target_id\tvina_score\n",
        gnina_targets=[],
        gnina_rows="target_id\tcnn_affinity\n",
    )

    result = _run_overlay(paths)

    assert result.returncode != 0
    assert "marked docked but has no AutoDock score row" in result.stderr


def test_overlay_rejects_structure_supported_without_a_gnina_score(
    tmp_path: Path,
) -> None:
    paths = _overlay_fixture(
        tmp_path,
        map_targets=[{"target_id": "P1", "status": "map_ready", "map_cache_key": "a"}],
        docking_targets=[{"target_id": "P1", "status": "docked"}],
        docking_rows="target_id\tvina_score\nP1\t-7.2\n",
        gnina_targets=[{"target_id": "P1", "status": "structure_supported"}],
        gnina_rows="target_id\tcnn_affinity\n",
    )

    result = _run_overlay(paths)

    assert result.returncode != 0
    assert "missing an AutoDock or GNINA score" in result.stderr


def test_daina_selection_requires_max_tanimoto(tmp_path: Path) -> None:
    scores = tmp_path / "daina.tsv"
    pd.DataFrame(
        {
            "target_id": [f"P{index:03d}" for index in range(300)],
            "evidence_count": [index + 1 for index in range(300)],
            "score": [1.0 - index / 1000 for index in range(300)],
        }
    ).to_csv(scores, sep="\t", index=False)

    result = _run(
        "stage3_select_daina.py",
        "--daina-scores",
        scores,
        "--top-n",
        256,
        "--out-csv",
        tmp_path / "selected.csv",
    )

    assert result.returncode != 0
    assert "missing required column(s): max_tanimoto" in result.stderr


def test_daina_selection_rejects_out_of_range_max_tanimoto(tmp_path: Path) -> None:
    scores = tmp_path / "daina.tsv"
    frame = pd.DataFrame(
        {
            "target_id": [f"P{index:03d}" for index in range(300)],
            "max_tanimoto": [1.0 - index / 1000 for index in range(300)],
            "evidence_count": [index + 1 for index in range(300)],
            "supporting_molecule_id": [f"CHEMBL{index}" for index in range(300)],
            "score": [1.0 - index / 1000 for index in range(300)],
        }
    )
    frame.loc[5, "max_tanimoto"] = 1.5
    frame.to_csv(scores, sep="\t", index=False)

    result = _run(
        "stage3_select_daina.py",
        "--daina-scores",
        scores,
        "--top-n",
        256,
        "--out-csv",
        tmp_path / "selected.csv",
    )

    assert result.returncode != 0
    assert "max_tanimoto outside [0, 1]" in result.stderr


def test_daina_known_ligand_count_comes_from_evidence_count(tmp_path: Path) -> None:
    """The count read a column the scorer never emits, so it was always 0.

    stage3_daina_zoete writes `evidence_count`; the old lookup asked for
    `n_known_ligands`, matched nothing, and fell through to the zero default.
    """
    scores = tmp_path / "daina.tsv"
    pd.DataFrame(
        {
            "target_id": [f"P{index:03d}" for index in range(300)],
            "max_tanimoto": [1.0 - index / 1000 for index in range(300)],
            "evidence_count": [(index + 1) * 7 for index in range(300)],
            "supporting_molecule_id": [f"CHEMBL{index}" for index in range(300)],
            "score": [1.0 - index / 1000 for index in range(300)],
        }
    ).to_csv(scores, sep="\t", index=False)
    selected = tmp_path / "selected.csv"

    result = _run(
        "stage3_select_daina.py",
        "--daina-scores", scores,
        "--top-n", 256,
        "--out-csv", selected,
    )

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(selected)
    assert frame["daina_known_ligand_count"].tolist()[:3] == [7, 14, 21]
    assert (frame["daina_known_ligand_count"] > 0).all()


def test_daina_selection_requires_evidence_count(tmp_path: Path) -> None:
    scores = tmp_path / "daina.tsv"
    pd.DataFrame(
        {
            "target_id": [f"P{index:03d}" for index in range(300)],
            "max_tanimoto": [1.0 - index / 1000 for index in range(300)],
            "score": [1.0 - index / 1000 for index in range(300)],
        }
    ).to_csv(scores, sep="\t", index=False)

    result = _run(
        "stage3_select_daina.py",
        "--daina-scores", scores,
        "--top-n", 256,
        "--out-csv", tmp_path / "selected.csv",
    )

    assert result.returncode != 0
    assert "missing required column(s): evidence_count" in result.stderr


def _supported_overlay(tmp_path: Path, *, scoring_method: str = "max-similarity") -> dict[str, Path]:
    return _overlay_fixture(
        tmp_path,
        scoring_method=scoring_method,
        map_targets=[
            {
                "target_id": "P1",
                "status": "map_ready",
                "map_cache_key": None,
                "pocket_id": None,
                "pocket_source": None,
                "structure_source": None,
            }
        ],
        docking_targets=[{"target_id": "P1", "status": "docked"}],
        docking_rows="target_id\tvina_score\nP1\t-7.2\n",
        gnina_targets=[
            {
                "target_id": "P1",
                "status": "structure_supported",
                "pose_file": None,
                "pose_sha256": None,
            }
        ],
        gnina_rows="target_id\tcnn_affinity\nP1\t2.3\n",
    )


def test_the_retrieval_evidence_reaches_the_published_overlay(tmp_path: Path) -> None:
    """The overlay used to drop every quantity a reader needs to judge a hit.

    `daina_top256.csv` carried max Tanimoto, the known-ligand count and the
    supporting molecule, and none of the three appeared in the final CSV.
    """
    paths = _supported_overlay(tmp_path)

    result = _run_overlay(paths)

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(paths["out"])
    assert frame.loc[0, "daina_max_tanimoto"] == pytest.approx(0.91)
    assert int(frame.loc[0, "daina_known_ligand_count"]) == 12
    assert frame.loc[0, "daina_supporting_molecule_id"] == "CHEMBL1"


def test_an_identical_reference_ligand_is_reported_as_a_known_interaction(
    tmp_path: Path,
) -> None:
    """A Tanimoto of 1.0 is retrieval, not prediction.

    Well-known cosmetic ingredients are themselves measured ligands in ChEMBL,
    so this is the common case rather than an edge case, and the artifact
    presented it identically to an extrapolated hit.
    """
    paths = _supported_overlay(tmp_path)
    selected = pd.read_csv(paths["selected"])
    selected.loc[0, "daina_max_tanimoto"] = 1.0
    selected.to_csv(paths["selected"], index=False)

    result = _run_overlay(paths)

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(paths["out"])
    assert bool(frame.loc[0, "daina_is_self_match"]) is True


def test_an_extrapolated_hit_is_not_reported_as_a_known_interaction(
    tmp_path: Path,
) -> None:
    paths = _supported_overlay(tmp_path)

    result = _run_overlay(paths)

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(paths["out"])
    assert bool(frame.loc[0, "daina_is_self_match"]) is False


def test_the_artifact_states_which_quantity_its_order_came_from(
    tmp_path: Path,
) -> None:
    """`docking_rrf` and `final_score` carry no docking information at all.

    They are copies of the similarity score, so the column names alone imply a
    fusion the overlay never performs.
    """
    paths = _supported_overlay(tmp_path)

    result = _run_overlay(paths)

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(paths["out"])
    assert frame.loc[0, "ranking_basis"] == "daina_max_tanimoto"
    assert frame.loc[0, "final_score"] == pytest.approx(frame.loc[0, "daina_score"])
    assert frame.loc[0, "docking_rrf"] == pytest.approx(frame.loc[0, "daina_score"])


def test_a_recipe_scored_run_does_not_claim_the_similarity_ordered_it(
    tmp_path: Path,
) -> None:
    """Under the promoted recipe the score and the similarity are different
    numbers - on alpha-arbutin the two orderings disagree at 253 of the top 256
    positions. Stamping `daina_max_tanimoto` on those rows sends a reader to
    sort by a column that cannot reproduce the rank."""
    paths = _supported_overlay(tmp_path, scoring_method="recipe:union_any_consensus")

    result = _run_overlay(paths)

    assert result.returncode == 0, result.stderr
    basis = pd.read_csv(paths["out"]).loc[0, "ranking_basis"]
    assert basis != "daina_max_tanimoto"
    assert "union_any_consensus" in basis, basis


def test_a_selection_without_the_evidence_columns_is_refused(tmp_path: Path) -> None:
    paths = _supported_overlay(tmp_path)
    selected = pd.read_csv(paths["selected"]).drop(columns=["daina_max_tanimoto"])
    selected.to_csv(paths["selected"], index=False)

    result = _run_overlay(paths)

    assert result.returncode == 1
    assert "daina_max_tanimoto" in result.stderr


def test_selection_carries_the_molecule_behind_the_similarity_score(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "daina.tsv"
    pd.DataFrame(
        {
            "target_id": [f"P{index:03d}" for index in range(300)],
            "max_tanimoto": [1.0 - index / 1000 for index in range(300)],
            "evidence_count": [index + 1 for index in range(300)],
            "supporting_molecule_id": [f"CHEMBL{index}" for index in range(300)],
            "score": [1.0 - index / 1000 for index in range(300)],
        }
    ).to_csv(scores, sep="\t", index=False)
    selected = tmp_path / "selected.csv"

    result = _run(
        "stage3_select_daina.py",
        "--daina-scores",
        scores,
        "--top-n",
        256,
        "--out-csv",
        selected,
    )

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(selected)
    assert frame.loc[0, "daina_supporting_molecule_id"] == "CHEMBL0"


def test_a_blank_supporting_molecule_is_refused(tmp_path: Path) -> None:
    scores = tmp_path / "daina.tsv"
    frame = pd.DataFrame(
        {
            "target_id": [f"P{index:03d}" for index in range(300)],
            "max_tanimoto": [1.0 - index / 1000 for index in range(300)],
            "evidence_count": [index + 1 for index in range(300)],
            "supporting_molecule_id": [f"CHEMBL{index}" for index in range(300)],
            "score": [1.0 - index / 1000 for index in range(300)],
        }
    )
    frame.loc[3, "supporting_molecule_id"] = "   "
    frame.to_csv(scores, sep="\t", index=False)

    result = _run(
        "stage3_select_daina.py",
        "--daina-scores",
        scores,
        "--top-n",
        256,
        "--out-csv",
        tmp_path / "selected.csv",
    )

    assert result.returncode == 1
    assert "supporting_molecule_id" in result.stderr


def _pose_fixture(tmp_path: Path, target_ids: list[str]) -> tuple[Path, Path, Path]:
    """A pose directory, its manifest, and a matching clean-structure directory."""
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir(exist_ok=True)
    clean = tmp_path / "clean"
    clean.mkdir(exist_ok=True)
    records = []
    for target_id in target_ids:
        pose = pose_dir / f"{target_id}.sdf"
        pose.write_text(
            f"{target_id}\n  SkinScout\n\n"
            "  0  0  0  0  0  0            999 V2000\nM  END\n$$$$\n"
        )
        (clean / f"{target_id}_clean.pdb").write_text("ATOM\n")
        records.append(
            {
                "target_id": target_id,
                "pose_file": f"{target_id}.sdf",
                "pose_sha256": _sha256(pose),
            }
        )
    manifest = pose_dir / "pose_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.docking_pose_manifest.v1",
                "target_count": len(records),
                "targets": records,
            }
        )
    )
    return pose_dir, manifest, clean


def _stub_gnina(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    gnina = fake_bin / "gnina"
    gnina.write_text("#!/bin/sh\nprintf 'CNNaffinity: 2.25\\n'\n")
    gnina.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    return env


def _gnina_subset_run(tmp_path: Path, *extra: object):
    pose_dir, manifest, clean = _pose_fixture(tmp_path, ["P1", "P2"])
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\n")
    scores = tmp_path / "gnina.tsv"
    return (
        _run(
            "stage3_gnina_rescore.py",
            "--top-csv", top,
            "--pose-manifest", manifest,
            "--pose-dir", pose_dir,
            "--clean-dir", clean,
            "--out-scores", scores,
            *extra,
            env=_stub_gnina(tmp_path),
        ),
        scores,
    )


def test_unscored_poses_are_refused_unless_the_caller_asks_for_a_subset(
    tmp_path: Path,
) -> None:
    """Exact parity is the right default: an unexpected extra pose is a wiring bug."""
    result, _ = _gnina_subset_run(tmp_path)

    assert result.returncode == 1
    assert "--allow-unscored-poses" in result.stderr


def test_rescoring_a_subset_of_docked_poses_is_allowed_explicitly(
    tmp_path: Path,
) -> None:
    """The comprehensive path docks the proteome and rescores a top percentage.

    GNINA there used to read the free ligand conformer instead of the docked
    pose, and pointing it at the poses means the manifest is deliberately a
    superset of the score table.
    """
    result, scores = _gnina_subset_run(tmp_path, "--allow-unscored-poses")

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(scores, sep="\t")
    assert frame["target_id"].tolist() == ["P1"]
    assert bool(frame.loc[0, "scored_actual_docked_pose"])


def test_a_scored_target_without_a_pose_stays_fatal(tmp_path: Path) -> None:
    """The guard that matters is never relaxed by asking for a subset."""
    pose_dir, manifest, clean = _pose_fixture(tmp_path, ["P1"])
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\nP2,0.9\n")
    (clean / "P2_clean.pdb").write_text("ATOM\n")

    result = _run(
        "stage3_gnina_rescore.py",
        "--top-csv", top,
        "--pose-manifest", manifest,
        "--pose-dir", pose_dir,
        "--clean-dir", clean,
        "--allow-unscored-poses",
        "--out-scores", tmp_path / "gnina.tsv",
        env=_stub_gnina(tmp_path),
    )

    assert result.returncode == 1
    assert "missing_pose" in result.stderr
    assert "P2" in result.stderr


def test_the_subset_flag_requires_pose_mode(tmp_path: Path) -> None:
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("x\n")
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\n")

    result = _run(
        "stage3_gnina_rescore.py",
        "--top-csv", top,
        "--ligand-sdf", ligand,
        "--clean-dir", tmp_path,
        "--allow-unscored-poses",
        "--out-scores", tmp_path / "gnina.tsv",
    )

    assert result.returncode == 1
    assert "requires pose-manifest mode" in result.stderr


def _rtmscore_subset_run(tmp_path: Path, *extra: object):
    pose_dir, manifest, clean = _pose_fixture(tmp_path, ["P1", "P2"])
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\n")
    scores = tmp_path / "rtm.tsv"
    return (
        _run(
            "stage3_rtmscore.py",
            "--top-csv", top,
            "--pose-manifest", manifest,
            "--pose-dir", pose_dir,
            "--clean-dir", clean,
            "--out-scores", scores,
            *extra,
        ),
        scores,
    )


def test_rtmscore_refuses_unscored_poses_unless_asked(tmp_path: Path) -> None:
    result, _ = _rtmscore_subset_run(tmp_path)

    assert result.returncode == 1
    assert "--allow-unscored-poses" in result.stderr


def test_rtmscore_requires_a_pose_for_every_scored_target(tmp_path: Path) -> None:
    pose_dir, manifest, clean = _pose_fixture(tmp_path, ["P1"])
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\nP2,0.9\n")
    (clean / "P2_clean.pdb").write_text("ATOM\n")

    result = _run(
        "stage3_rtmscore.py",
        "--top-csv", top,
        "--pose-manifest", manifest,
        "--pose-dir", pose_dir,
        "--clean-dir", clean,
        "--allow-unscored-poses",
        "--out-scores", tmp_path / "rtm.tsv",
    )

    assert result.returncode == 1
    assert "missing_pose" in result.stderr
    assert "P2" in result.stderr


def test_rtmscore_rejects_a_ligand_sdf_combined_with_poses(tmp_path: Path) -> None:
    """The two modes score different geometries and must never be mixed."""
    pose_dir, manifest, clean = _pose_fixture(tmp_path, ["P1"])
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("x\n")
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\n")

    result = _run(
        "stage3_rtmscore.py",
        "--top-csv", top,
        "--pose-manifest", manifest,
        "--pose-dir", pose_dir,
        "--ligand-sdf", ligand,
        "--clean-dir", clean,
        "--out-scores", tmp_path / "rtm.tsv",
    )

    assert result.returncode == 1
    assert "cannot be combined" in result.stderr


def test_rtmscore_status_manifest_requires_pose_mode(tmp_path: Path) -> None:
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("x\n")
    top = tmp_path / "top.csv"
    top.write_text("target_id,score\nP1,1.0\n")

    result = _run(
        "stage3_rtmscore.py",
        "--top-csv", top,
        "--ligand-sdf", ligand,
        "--clean-dir", tmp_path,
        "--out-scores", tmp_path / "rtm.tsv",
        "--out-status-manifest", tmp_path / "rtm.json",
    )

    assert result.returncode == 1
    assert "requires pose-manifest mode" in result.stderr


def test_both_rescorers_share_one_pose_manifest_reader() -> None:
    """Two copies would drift into disagreeing about what a valid pose is."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import docking_pose_manifest
    import stage3_gnina_rescore
    import stage3_rtmscore

    assert (
        stage3_gnina_rescore.read_pose_manifest
        is docking_pose_manifest.read_pose_manifest
    )
    assert (
        stage3_rtmscore.read_pose_manifest is docking_pose_manifest.read_pose_manifest
    )
    assert (
        stage3_gnina_rescore.check_pose_parity is stage3_rtmscore.check_pose_parity
    )
