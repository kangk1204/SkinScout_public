"""F17 regression: the Boltz-2 length cap is actually applied to the input.

The `--max-residues`/`--crop-radius` options used to be validated and then
ignored, so a 4,000-residue receptor ran unbounded on a 12 GB host. Now an
over-cap receptor is either cropped around the Stage 4 pocket with the original
residue mapping recorded, or rejected with an explanatory error. The sequence
is never silently truncated.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import boltz2_runner as runner  # noqa: E402


def write_receptor(path: Path, n_residues: int, *, chain: str = "A") -> None:
    lines = []
    for index in range(1, n_residues + 1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA {chain}{index:4d}    "
            f"{float(index):8.3f}   0.000   0.000  1.00 90.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_pocket_box_json(path: Path, center: tuple[float, float, float]) -> None:
    path.write_text(
        json.dumps(
            {
                "pockets": [{"rank": 1, "score": None, "center": list(center), "radius": 12.0}],
                "source": "test",
            }
        )
        + "\n"
    )


def write_box_text(path: Path, center: tuple[float, float, float]) -> None:
    path.write_text(
        f"center_x = {center[0]}\ncenter_y = {center[1]}\ncenter_z = {center[2]}\n"
        "size_x = 30\nsize_y = 30\nsize_z = 30\n"
    )


def write_ligand_sdf(path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert mol is not None
    AllChem.EmbedMolecule(mol, randomSeed=7)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def test_short_receptor_is_left_unchanged(tmp_path: Path) -> None:
    receptor = tmp_path / "short.pdb"
    write_receptor(receptor, 5)
    pocket = tmp_path / "pocket_box.json"
    write_pocket_box_json(pocket, (3.0, 0.0, 0.0))

    model, meta = runner.prepare_receptor_for_boltz(
        receptor,
        max_residues=700,
        crop_radius=20.0,
        out_dir=tmp_path / "out",
        label="P1",
        pocket_path=pocket,
    )

    assert model == receptor
    assert meta["crop_mode"] == "not_needed"
    assert meta["input_residues"] == 5
    assert meta["model_residues"] == 5
    assert meta["crop_map"] == ""
    assert not (tmp_path / "out" / "receptor_cropped.pdb").exists()


def test_long_receptor_without_pocket_fails_explainably(tmp_path: Path) -> None:
    receptor = tmp_path / "long.pdb"
    write_receptor(receptor, 701)

    with pytest.raises(runner.ReceptorLengthError) as excinfo:
        runner.prepare_receptor_for_boltz(
            receptor,
            max_residues=700,
            crop_radius=20.0,
            out_dir=tmp_path / "out",
            label="P1",
        )

    message = str(excinfo.value)
    assert "701 residues" in message
    assert "--max-residues 700" in message
    assert "no pocket box" in message
    assert not (tmp_path / "out" / "receptor_cropped.pdb").exists()


def test_long_receptor_is_cropped_around_pocket_with_mapping(tmp_path: Path) -> None:
    receptor = tmp_path / "long.pdb"
    write_receptor(receptor, 701)
    pocket = tmp_path / "pocket_box.json"
    write_pocket_box_json(pocket, (350.0, 0.0, 0.0))

    model, meta = runner.prepare_receptor_for_boltz(
        receptor,
        max_residues=700,
        crop_radius=20.0,
        out_dir=tmp_path / "out",
        label="P1",
        pocket_path=pocket,
    )

    assert meta["crop_mode"] == "pocket_crop"
    assert meta["input_residues"] == 701
    assert 0 < meta["model_residues"] <= 700
    sequence = runner.pdb_sequence(model)
    assert len(sequence) == meta["model_residues"]
    mapping_path = Path(meta["crop_map"])
    assert mapping_path.is_file()
    rows = list(csv.DictReader(mapping_path.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == len(sequence)
    original = [int(row["original_resseq"]) for row in rows]
    # +-20 A around residue 350 keeps 330..370 only; the pocket is preserved.
    assert original == list(range(330, 371))
    assert [row["crop_index"] for row in rows] == [
        str(index) for index in range(1, len(rows) + 1)
    ]
    assert rows[0]["original_chain"] == "A"
    assert rows[0]["original_resname"] == "ALA"


def test_long_receptor_crops_from_docking_box_text(tmp_path: Path) -> None:
    receptor = tmp_path / "long.pdb"
    write_receptor(receptor, 705)
    box = tmp_path / "P1.box.txt"
    write_box_text(box, (350.0, 0.0, 0.0))

    model, meta = runner.prepare_receptor_for_boltz(
        receptor,
        max_residues=700,
        crop_radius=20.0,
        out_dir=tmp_path / "out",
        label="P1",
        pocket_path=box,
    )

    assert meta["crop_mode"] == "pocket_crop"
    assert meta["input_residues"] == 705
    assert Path(meta["crop_map"]).is_file()


def test_crop_without_any_residue_in_radius_fails(tmp_path: Path) -> None:
    receptor = tmp_path / "long.pdb"
    write_receptor(receptor, 701)
    pocket = tmp_path / "pocket_box.json"
    write_pocket_box_json(pocket, (5000.0, 0.0, 0.0))

    with pytest.raises(runner.ReceptorLengthError) as excinfo:
        runner.prepare_receptor_for_boltz(
            receptor,
            max_residues=700,
            crop_radius=20.0,
            out_dir=tmp_path / "out",
            label="P1",
            pocket_path=pocket,
        )

    assert "no residue lies within" in str(excinfo.value)


def test_stage3_affinity_crops_with_box_and_rejects_without(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stage3_boltz2_affinity as stage3

    receptor = tmp_path / "P1_clean.pdb"
    write_receptor(receptor, 701)
    ligand = tmp_path / "ligand.sdf"
    write_ligand_sdf(ligand)
    box = tmp_path / "P1.box.txt"
    write_box_text(box, (350.0, 0.0, 0.0))
    monkeypatch.setattr(stage3.shutil, "which", lambda name: "/usr/bin/boltz")
    captured: dict[str, str] = {}

    def fake_predict(input_yaml: Path, *_args, **_kwargs):
        captured["yaml"] = input_yaml.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(["boltz"], 0, "", "")

    monkeypatch.setattr(stage3, "run_boltz_predict", fake_predict)
    monkeypatch.setattr(
        stage3, "load_affinity_payload", lambda out_dir: {"affinity_pred_value": 1.2}
    )

    score = stage3.call_boltz(
        receptor, ligand, tmp_path / "work", 700, 20.0, pocket_path=box
    )

    assert score == pytest.approx(-1.2)
    sequence_line = next(
        line for line in captured["yaml"].splitlines() if "sequence:" in line
    )
    assert 0 < sequence_line.count("A") <= 700

    with pytest.raises(SystemExit, match="receptor length cap exceeded"):
        stage3.call_boltz(receptor, ligand, tmp_path / "work2", 700, 20.0)


def _write_fake_boltz(bin_dir: Path) -> None:
    bin_dir.mkdir()
    fake = bin_dir / "boltz"
    fake.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--out_dir' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out/predictions/input\"\n"
        "printf '{\"iptm\":0.8,\"complex_plddt\":90,\"affinity_log_uM\":1.2,"
        "\"posebusters_valid\":true}' > \"$out/report.json\"\n"
        "printf 'ATOM protein\\nHETATM ligand\\n' > \"$out/predictions/input/input_model_0.pdb\"\n"
    )
    fake.chmod(0o755)


def _env_with_fake_boltz(bin_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', os.defpath)}"
    return env


def _run_stage5(tmp_path: Path, manifest: Path, ligand: Path, report: Path):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage5_boltz2.py"),
            "--manifest", str(manifest),
            "--ligand-sdf", str(ligand),
            "--struct-dir", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--out-report", str(report),
        ],
        capture_output=True,
        text=True,
        env=_env_with_fake_boltz(tmp_path / "bin"),
        check=False,
    )


def test_stage5_cli_crops_long_receptor_and_records_mapping(tmp_path: Path) -> None:
    receptor = tmp_path / "P1_input.pdb"
    write_receptor(receptor, 701)
    pocket = tmp_path / "pocket_box.json"
    write_pocket_box_json(pocket, (350.0, 0.0, 0.0))
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{
            "target_id": "P1",
            "source": "alphafold_cleaned",
            "input_pdb": str(receptor),
            "pocket_box_json": str(pocket),
        }]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_ligand_sdf(ligand)
    _write_fake_boltz(tmp_path / "bin")
    report = tmp_path / "report.tsv"

    res = _run_stage5(tmp_path, manifest, ligand, report)

    assert res.returncode == 0, res.stderr
    rows = pd.read_csv(report, sep="\t")
    assert rows["crop_mode"].tolist() == ["pocket_crop"]
    assert rows["n_input_residues"].tolist() == [701]
    assert int(rows.iloc[0]["n_residues"]) <= 700
    mapping = Path(rows.iloc[0]["crop_map"])
    assert mapping.is_file()
    assert mapping.read_text().splitlines()[0].split("\t") == [
        "crop_index",
        "original_chain",
        "original_resseq",
        "original_icode",
        "original_resname",
    ]
    # The YAML actually handed to Boltz carries the cropped sequence, not the
    # 701-residue receptor sequence.
    yaml_text = (tmp_path / "out" / "P1" / "input.yaml").read_text(encoding="utf-8")
    sequence_line = next(line for line in yaml_text.splitlines() if "sequence:" in line)
    assert sequence_line.count("A") == int(rows.iloc[0]["n_residues"]) <= 700


def test_stage5_cli_rejects_over_cap_receptor_without_pocket(tmp_path: Path) -> None:
    receptor = tmp_path / "P1_input.pdb"
    write_receptor(receptor, 701)
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{"target_id": "P1", "source": "alphafold_cleaned", "input_pdb": str(receptor)}]
    ).to_csv(manifest, sep="\t", index=False)
    ligand = tmp_path / "ligand.sdf"
    write_ligand_sdf(ligand)
    _write_fake_boltz(tmp_path / "bin")
    report = tmp_path / "report.tsv"

    res = _run_stage5(tmp_path, manifest, ligand, report)

    assert res.returncode != 0
    assert "Boltz-2 receptor length cap exceeded" in res.stderr
    assert "no pocket box" in res.stderr
    assert not report.exists()
