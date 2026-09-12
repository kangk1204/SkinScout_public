"""Regression tests for keeping generated Stage 0 data out of Git."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


GENERATED_STAGE0_PATHS = {
    "data/cosing/cosing.parquet",
    "data/drug_avoidance/drugs.parquet",
    "data/drug_avoidance/scaffolds.parquet",
    "data/skin_efficacy_kg/skin_efficacy.graphml",
    "data/skin_proteome/.ingest_complete",
    "data/skin_proteome/skin_proteome.tsv",
}

IGNORED_STAGE0_DIRS = {
    "data/chembl37/",
    "data/evidence_splits/",
    "data/cosing/",
    "data/drug_avoidance/",
    "data/skin_efficacy_kg/",
    "data/skin_expression/",
    "data/skin_proteome/",
    "data/pharmacophore_smarts/",
}


def test_generated_stage0_data_paths_are_not_tracked() -> None:
    res = subprocess.run(
        ["git", "ls-files", "data"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    tracked = set(res.stdout.splitlines())

    assert GENERATED_STAGE0_PATHS.isdisjoint(tracked)
    assert "data/holo_transplant/P14679_box.txt" in tracked
    assert "data/holo_transplant/P14679_with_Cu.pdb" in tracked


def test_generated_stage0_dirs_are_ignored() -> None:
    ignored = set()
    for line in (ROOT / ".gitignore").read_text().splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            ignored.add(text)

    assert IGNORED_STAGE0_DIRS <= ignored
