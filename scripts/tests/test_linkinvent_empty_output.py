"""F22 regression: LinkInvent with zero candidates is an explainable empty result.

``linkinvent_filter.py`` derived its output fieldnames from ``rows[0]``, so a
header-only ``linked.smi`` produced ``[preselect] 0`` and then an IndexError.
A seed with no pharmacophore features divided by zero the same way.

The filter now always writes fixed-schema, header-only outputs plus a manifest
that distinguishes empty input, all-filtered input, and a zero-feature seed.
The normal path keeps its previous output schemas and values.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analog_funnel"))

SEED = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
PASSING = "O=C(O)c1ccc(-c2ccccc2)cc1"
FILTERED_FIELDS = ["smiles", "qed", "sa", "mw", "recall"]
CANDIDATE_FIELDS = [
    "gene", "uniprot", "smiles", "qed", "sa", "mw", "feature_recall", "skin_reaction",
]


class _FakeTable:
    def column(self, _name: str) -> SimpleNamespace:
        return SimpleNamespace(to_pylist=lambda: [])


def _empty_pq() -> SimpleNamespace:
    return SimpleNamespace(read_table=lambda path, columns=None: _FakeTable())


class _FakeADMETModel:
    def predict(self, smiles: str) -> dict:
        return {"Skin_Reaction": 0.1}


def _fake_admet_ai() -> ModuleType:
    module = ModuleType("admet_ai")
    module.ADMETModel = _FakeADMETModel
    return module


@pytest.fixture()
def filter_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module("linkinvent_filter")
    monkeypatch.setattr(module, "P", tmp_path)
    monkeypatch.setattr(module, "SEED", SEED)
    monkeypatch.setattr(module, "pq", _empty_pq())
    return module


def _write_linked(tmp_path: Path, rows: list[str]) -> None:
    with (tmp_path / "linked.smi").open("w", encoding="utf-8") as fh:
        fh.write("SMILES,name\n")
        for smi in rows:
            fh.write(f"{smi},name\n")


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _manifest(tmp_path: Path) -> dict:
    return json.loads(
        (tmp_path / "linkinvent_filter_manifest.json").read_text(encoding="utf-8")
    )


def _assert_header_only(tmp_path: Path, name: str, fields: list[str]) -> None:
    lines = (tmp_path / name).read_text(encoding="utf-8-sig").strip().splitlines()
    assert lines and lines[0].split(",") == fields
    assert _read(tmp_path / name) == []


def test_header_only_input_is_explainable(
    tmp_path: Path, filter_module
) -> None:
    _write_linked(tmp_path, [])

    assert filter_module.main() == 0

    _assert_header_only(tmp_path, "filtered_all.csv", FILTERED_FIELDS)
    _assert_header_only(tmp_path, "TGF-B1_candidates.csv", CANDIDATE_FIELDS)
    payload = _manifest(tmp_path)
    assert payload["status"] == "empty_input"
    assert payload["reason"]
    assert payload["input"] == payload["parsed"] == payload["filtered"] == payload["kept"] == 0


def test_all_filtered_input_is_explainable(
    tmp_path: Path, filter_module
) -> None:
    _write_linked(tmp_path, ["CCO"])

    assert filter_module.main() == 0

    _assert_header_only(tmp_path, "filtered_all.csv", FILTERED_FIELDS)
    _assert_header_only(tmp_path, "TGF-B1_candidates.csv", CANDIDATE_FIELDS)
    payload = _manifest(tmp_path)
    assert payload["status"] == "all_filtered"
    assert payload["reason"]
    assert payload["input"] == 1 and payload["parsed"] == 1
    assert payload["filtered"] == payload["kept"] == 0


def test_zero_feature_seed_is_explainable(
    tmp_path: Path, filter_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(filter_module, "SEED", "C")
    _write_linked(tmp_path, ["CCO"])

    assert filter_module.main() == 2

    _assert_header_only(tmp_path, "filtered_all.csv", FILTERED_FIELDS)
    _assert_header_only(tmp_path, "TGF-B1_candidates.csv", CANDIDATE_FIELDS)
    payload = _manifest(tmp_path)
    assert payload["status"] == "seed_feature_zero"
    assert payload["reason"]
    assert payload["filtered"] == payload["kept"] == 0


def test_normal_case_keeps_outputs_and_schemas(
    tmp_path: Path, filter_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "admet_ai", _fake_admet_ai())
    _write_linked(tmp_path, [PASSING])

    assert filter_module.main() == 0

    filtered = _read(tmp_path / "filtered_all.csv")
    assert len(filtered) == 1
    assert list(filtered[0]) == FILTERED_FIELDS
    assert filtered[0]["smiles"] == PASSING
    assert filtered[0]["qed"] and filtered[0]["recall"]

    kept = _read(tmp_path / "TGF-B1_candidates.csv")
    assert len(kept) == 1
    assert list(kept[0]) == CANDIDATE_FIELDS
    assert kept[0]["gene"] == "TGF-B1" and kept[0]["uniprot"] == "P01137"
    assert kept[0]["feature_recall"] and kept[0]["skin_reaction"] == "0.1"

    payload = _manifest(tmp_path)
    assert payload["status"] == "ok"
    assert payload["input"] == payload["parsed"] == payload["filtered"] == 1
    assert payload["kept"] == 1
