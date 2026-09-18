"""Regression tests for Stage 5.5 pose-supported interaction atom gates."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


def load_prolif_module():
    spec = importlib.util.spec_from_file_location(
        "stage5_5_prolif_under_test",
        ROOT / "scripts" / "stage5_5_prolif.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plip_module():
    spec = importlib.util.spec_from_file_location(
        "stage5_5_plip_under_test",
        ROOT / "scripts" / "stage5_5_plip.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_boltz_complex(tmp_path: Path) -> Path:
    target = tmp_path / "boltz" / "P12345"
    target.mkdir(parents=True)
    (target / "complex.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(target / "ligand.sdf"))
    writer.write(mol)
    writer.close()
    return tmp_path / "boltz"


def run_script(
    script: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def consensus_args(tmp_path: Path, plip: Path, prolif: Path) -> list[str]:
    return [
        "--plip",
        str(plip),
        "--prolif",
        str(prolif),
        "--out-json",
        str(tmp_path / "consensus.json"),
    ]


def test_plip_missing_fails_without_empty_xml(tmp_path: Path) -> None:
    boltz_dir = write_boltz_complex(tmp_path)
    env = os.environ.copy()
    env["PATH"] = ""

    res = run_script(
        "stage5_5_plip.py",
        ["--boltz-dir", str(boltz_dir), "--out-xml", str(tmp_path / "plip.xml")],
        env=env,
    )

    assert res.returncode != 0
    assert "PLIP is required" in res.stderr
    assert not (tmp_path / "plip.xml").exists()


def test_plip_missing_allow_empty_writes_degraded_xml(tmp_path: Path) -> None:
    boltz_dir = write_boltz_complex(tmp_path)
    env = os.environ.copy()
    env["PATH"] = ""

    res = run_script(
        "stage5_5_plip.py",
        [
            "--boltz-dir",
            str(boltz_dir),
            "--out-xml",
            str(tmp_path / "plip.xml"),
            "--allow-empty",
        ],
        env=env,
    )

    assert res.returncode == 0, res.stderr
    assert "<plip_report" in (tmp_path / "plip.xml").read_text()


def test_plip_normalizes_nonsequential_pdb_serials_to_ligand_order(
    tmp_path: Path,
) -> None:
    module = load_plip_module()
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "HETATM  105  C1  LIG B   1       2.000   2.000   3.000  1.00 90.00           C\n"
        "HETATM  205  O1  LIG B   1       3.000   2.000   3.000  1.00 90.00           O\n"
        "HETATM  999  N1  LIG B   1       4.000   2.000   3.000  1.00 90.00           N\n"
        "END\n"
    )
    report = tmp_path / "report.xml"
    report.write_text(
        """<report><bindingsite><interactions>
        <hydrophobic_interactions><hydrophobic_interaction>
          <ligcarbonidx>205</ligcarbonidx>
        </hydrophobic_interaction></hydrophobic_interactions>
        <hydrogen_bonds>
          <hydrogen_bond><protisdon>True</protisdon><acceptoridx>105</acceptoridx></hydrogen_bond>
          <hydrogen_bond><protisdon>False</protisdon><donoridx>999</donoridx></hydrogen_bond>
        </hydrogen_bonds>
        <salt_bridges><salt_bridge><lig_idx_list><idx>999</idx><idx>105</idx></lig_idx_list></salt_bridge></salt_bridges>
        </interactions></bindingsite></report>"""
    )

    assert module.normalized_ligand_indices(report, complex_pdb) == {0, 1, 2}


def test_plip_rejects_interaction_serial_outside_ligand_records(
    tmp_path: Path,
) -> None:
    module = load_plip_module()
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "ATOM      7  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "HETATM  105  C1  LIG B   1       2.000   2.000   3.000  1.00 90.00           C\n"
        "END\n"
    )
    report = tmp_path / "report.xml"
    report.write_text(
        "<report><hydrophobic_interaction>"
        "<ligcarbonidx>7</ligcarbonidx>"
        "</hydrophobic_interaction></report>"
    )

    try:
        module.normalized_ligand_indices(report, complex_pdb)
    except SystemExit as exc:
        assert "is absent from Boltz HETATM records" in str(exc)
    else:
        raise AssertionError("expected non-ligand PLIP serial to be rejected")


def test_prolif_missing_dependency_fails_without_empty_csv(tmp_path: Path) -> None:
    boltz_dir = write_boltz_complex(tmp_path)
    out_csv = tmp_path / "prolif.csv"
    out_csv.write_text("stale\n")
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "prolif.py").write_text("raise ImportError('forced missing')\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shim_dir)

    res = run_script(
        "stage5_5_prolif.py",
        ["--boltz-dir", str(boltz_dir), "--out-csv", str(out_csv)],
        env=env,
    )

    assert res.returncode != 0
    assert "ProLIF and MDAnalysis are required" in res.stderr
    assert not out_csv.exists()


def test_prolif_prefers_parent_ligand_indices_over_residue_local_indices() -> None:
    module = load_prolif_module()

    class FakeFingerprint:
        ifp = {
            0: {
                ("LIG1", "ASP42"): {
                    "HBAcceptor": (
                        {
                            "indices": {"ligand": [0], "protein": [10]},
                            "parent_indices": {"ligand": [7, 8], "protein": [110]},
                        },
                    ),
                },
            },
        }

    assert module.ligand_atom_indices_from_fingerprint(FakeFingerprint()) == {7, 8}


def test_prolif_falls_back_to_local_ligand_indices() -> None:
    module = load_prolif_module()

    class FakeFingerprint:
        ifp = {
            0: {
                ("LIG1", "ASP42"): {
                    "Hydrophobic": (
                        {
                            "indices": {"ligand": [2], "protein": [10]},
                        },
                    ),
                },
            },
        }

    assert module.ligand_atom_indices_from_fingerprint(FakeFingerprint()) == {2}


def test_prolif_rejects_parent_indices_outside_ligand_atom_count() -> None:
    module = load_prolif_module()

    class FakeFingerprint:
        ifp = {
            0: {
                ("LIG1", "ASP42"): {
                    "Hydrophobic": (
                        {
                            "parent_indices": {"ligand": [3], "protein": [10]},
                        },
                    ),
                },
            },
        }

    try:
        module.ligand_atom_indices_from_fingerprint(
            FakeFingerprint(),
            ligand_atom_count=3,
        )
    except ValueError as exc:
        assert "exceed the bound ligand atom count" in str(exc)
    else:
        raise AssertionError("out-of-range ProLIF parent index was accepted")


def test_prolif_uses_bound_complex_coordinates_not_separate_sdf() -> None:
    source = (ROOT / "scripts" / "stage5_5_prolif.py").read_text()

    assert 'select_atoms("protein")' in source
    assert 'select_atoms("not protein")' in source
    assert "Molecule.from_mda" in source
    assert "generate(ligand, protein, metadata=True)" in source
    assert "sdf_supplier" not in source
    assert "run_from_iterable" not in source


def test_invalid_claim_source_scripts_are_removed_from_stage5_5_workflow() -> None:
    workflow = (ROOT / "workflow" / "rules" / "stage5_5_pharmacophore.smk").read_text()

    assert not (ROOT / "scripts" / "stage5_5_gnina_attr.py").exists()
    assert not (ROOT / "scripts" / "stage5_5_boltz_attention.py").exists()
    assert not (ROOT / "scripts" / "stage5_5_functional_groups.py").exists()
    assert not (ROOT / "scripts" / "stage5_5_pharmer_query.py").exists()
    assert "gnina_atom_attribution" not in workflow
    assert "boltz2_attention" not in workflow
    assert "--gnina" not in workflow
    assert "--boltz-attn" not in workflow
    assert "functional_group_label" not in workflow
    assert "pharmer_query" not in workflow
    assert "out-smarts" not in workflow


def test_consensus_empty_sources_fail_by_default(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text("<?xml version='1.0'?><plip_report />")
    prolif.write_text("target_id,atom_idx\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "Pose-supported interaction atom evidence is empty" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_malformed_plip_xml(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text("<plip_report><target id='T1'><bindingsite>")
    prolif.write_text("target_id,atom_idx\nT1,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "PLIP XML is malformed" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_requires_confirmed_atoms(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,2\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "No pose-supported interaction atoms reached" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_requires_complete_source_coverage_per_target(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT2,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script(
        "stage5_5_consensus.py",
        [*consensus_args(tmp_path, plip, prolif), "--min-votes", "1"],
    )

    assert res.returncode != 0
    assert "complete same-target PLIP and ProLIF coverage" in res.stderr
    assert "plip=T1" in res.stderr
    assert "prolif=T2" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_min_votes_above_available_sources(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1\n")

    res = run_script(
        "stage5_5_consensus.py",
        [*consensus_args(tmp_path, plip, prolif), "--min-votes", "3"],
    )

    assert res.returncode != 0
    assert "--min-votes cannot exceed the 2 available Stage 5.5 evidence sources" in res.stderr


def test_consensus_rejects_duplicate_prolif_atom_evidence(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1\nT1,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "ProLIF evidence contains duplicate target_id/atom_idx rows" in res.stderr
    assert "T1:1" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_invalid_prolif_atom_index(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,not-an-int\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "ProLIF atom_idx must be an integer" in res.stderr
    assert "not-an-int" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_fractional_prolif_atom_index(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1.5\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "ProLIF atom_idx must be an integer" in res.stderr
    assert "T1:1.5" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_negative_prolif_atom_index(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,-1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "ProLIF atom_idx must be a non-negative integer" in res.stderr
    assert "T1:-1" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_prolif_missing_required_columns(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id\nT1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "ProLIF CSV missing required column(s): atom_idx" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_blank_prolif_target_id(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\n,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "ProLIF target_id must be non-empty" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_invalid_plip_atom_index(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /><ligand_atom idx='not-an-int' />"
        "</bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "PLIP ligand_atom idx must be an integer" in res.stderr
    assert "T1:not-an-int" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_negative_plip_atom_index(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='-1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "PLIP ligand_atom idx must be a non-negative integer" in res.stderr
    assert "T1:-1" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_duplicate_normalized_plip_atom_index(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /><ligand_atom idx='01' />"
        "</bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "PLIP evidence contains duplicate ligand_atom idx values" in res.stderr
    assert "T1:1" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_plip_target_without_id(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nunknown,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "PLIP target missing required id" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_rejects_duplicate_normalized_plip_target_id(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report>"
        "<target id='T1'><bindingsite><ligand_atom idx='1' /></bindingsite></target>"
        "<target id=' T1 '><bindingsite><ligand_atom idx='0' /></bindingsite></target>"
        "</plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1\n")
    (tmp_path / "consensus.json").write_text("stale\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode != 0
    assert "PLIP evidence contains duplicate target id values" in res.stderr
    assert "T1" in res.stderr
    assert not (tmp_path / "consensus.json").exists()


def test_consensus_writes_outputs_when_plip_and_prolif_agree(tmp_path: Path) -> None:
    plip = tmp_path / "plip.xml"
    prolif = tmp_path / "prolif.csv"
    plip.write_text(
        "<?xml version='1.0'?><plip_report><target id='T1'><bindingsite>"
        "<ligand_atom idx='1' /></bindingsite></target></plip_report>"
    )
    prolif.write_text("target_id,atom_idx\nT1,1\n")

    res = run_script("stage5_5_consensus.py", consensus_args(tmp_path, plip, prolif))

    assert res.returncode == 0, res.stderr
    payload = json.loads((tmp_path / "consensus.json").read_text())
    assert payload["T1"]["confirmed_atoms"] == [1]
    assert payload["T1"]["pose_supported_interaction_atoms"] == [1]
    assert payload["T1"]["evidence_label"] == "pose-supported interaction atoms"
    assert payload["T1"]["coordinate_system"] == "boltz_complex_ligand_atom_order_0_based"
    assert payload["T1"]["degraded"] is False
    assert payload["T1"]["claim_eligible"] is True
    assert set(payload["T1"]["votes"]) == {"plip", "prolif"}
