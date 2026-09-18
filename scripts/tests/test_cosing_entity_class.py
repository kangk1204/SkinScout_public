"""CosIng entity-class contract (audit F11).

PubChem name search returns a first hit even for extracts and mixtures, so a
botanical name would silently become the exact reference of a marker compound.
Extracts/mixtures/botanicals must stay unresolved (with a curation record) and
only verified single compounds may reach ``cosing.parquet``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage0_cosing as cosing  # noqa: E402


def _row(name: str, cas: str | None = None) -> cosing.CosingRow:
    return cosing.CosingRow(
        inci_name=name, cas=cas, einecs=None, functions=[], smiles=None
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NIACINAMIDE", cosing.ENTITY_SINGLE_COMPOUND),
        ("GLYCERIN", cosing.ENTITY_SINGLE_COMPOUND),
        ("ALOE BARBADENSIS EXTRACT", cosing.ENTITY_EXTRACT),
        ("CAMELLIA SINENSIS LEAF EXTRACT", cosing.ENTITY_EXTRACT),
        ("A 2:1 MIXTURE OF: RESORCINOL AND SULFONATE", cosing.ENTITY_MIXTURE),
        ("LAVENDER OIL", cosing.ENTITY_MIXTURE),
        ("CAMELLIA SINENSIS FLOWER WATER", cosing.ENTITY_MIXTURE),
        ("ALLIUM SATIVUM BULB POWDER", cosing.ENTITY_BOTANICAL),
        ("SANTALUM ALBUM WOOD POWDER", cosing.ENTITY_BOTANICAL),
        ("WATER", cosing.ENTITY_SINGLE_COMPOUND),
        ("ALUMINUM POWDER", cosing.ENTITY_SINGLE_COMPOUND),
    ],
)
def test_entity_classification(name: str, expected: str) -> None:
    assert cosing.classify_entity(name) == expected


def test_extract_name_hit_stays_unresolved() -> None:
    row = _row("ALOE BARBADENSIS EXTRACT", "85507-69-3")
    first_hit = "CC1=CC=C(C=C1)C(=O)/C=C/C2=CC(=CC=C2)[N+](=O)[O-]"
    cache = {"smiles_for:85507-69-3": first_hit}

    records, unresolved, invalid, suspects = cosing.build_records([row], cache)

    assert records == []
    assert invalid == []
    assert unresolved[0]["entity_class"] == cosing.ENTITY_EXTRACT
    assert unresolved[0]["reason"] == "extract_requires_verified_structure_evidence"
    assert suspects[0]["first_hit_smiles"] == first_hit
    assert suspects[0]["fingerprint_sha256"]
    assert suspects[0]["reason"] == unresolved[0]["reason"]


def test_garlic_variants_remain_distinct_entities() -> None:
    names = [
        "ALLIUM SATIVUM BULB EXTRACT",
        "ALLIUM SATIVUM BULB JUICE",
        "ALLIUM SATIVUM BULB POWDER",
        "ALLIUM SATIVUM ROOT EXTRACT",
        "ALLIUM SATIVUM STEM EXTRACT",
    ]
    diallyl_trisulfide = "C=CCSSSCC=C"
    cache = {f"smiles_for:{name}": diallyl_trisulfide for name in names}
    cache["smiles_for:8008-99-9"] = diallyl_trisulfide

    records, unresolved, _, suspects = cosing.build_records(
        [_row(name, "8008-99-9") for name in names], cache
    )

    assert records == [], "garlic materials must not collapse into one compound"
    assert len(unresolved) == len(names)
    assert len({row["fingerprint_sha256"] for row in suspects}) == len(names)
    assert {row["entity_class"] for row in suspects} == {
        cosing.ENTITY_EXTRACT,
        cosing.ENTITY_BOTANICAL,
    }


def test_verified_single_compound_control_resolves() -> None:
    row = _row("NIACINAMIDE", "98-92-0")
    cache = {"smiles_for:98-92-0": "NC(=O)c1cccnc1"}

    records, unresolved, invalid, suspects = cosing.build_records([row], cache)

    assert len(records) == 1
    assert records[0]["entity_class"] == cosing.ENTITY_SINGLE_COMPOUND
    assert records[0]["smiles"] == "NC(=O)c1cccnc1"
    assert records[0]["inchikey"]
    assert unresolved == [] and invalid == [] and suspects == []


def test_non_single_entities_are_excluded_from_pubchem_resolution() -> None:
    rows = [
        _row("ALOE BARBADENSIS EXTRACT", "85507-69-3"),
        _row("NIACINAMIDE", "98-92-0"),
    ]

    resolvable = cosing.single_compound_rows(rows)

    assert [row.inci_name for row in resolvable] == ["NIACINAMIDE"]


def test_curation_record_lists_fingerprint_and_reason(tmp_path: Path) -> None:
    row = _row("ALOE BARBADENSIS EXTRACT", "85507-69-3")
    cache = {"smiles_for:85507-69-3": "CCO"}
    _, _, _, suspects = cosing.build_records([row], cache)
    curation_dir = tmp_path / "curation"

    record = cosing.write_curation_record(suspects, curation_dir)

    path = curation_dir / cosing.CURATION_RECORD_NAME
    assert path.exists()
    assert record["rows"] == 1
    assert record["sha256"] == cosing._sha256(path)
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(cosing.CURATION_RECORD_COLUMNS)
    assert frame.loc[0, "fingerprint_sha256"] == suspects[0]["fingerprint_sha256"]
    assert "requires_verified_structure_evidence" in frame.loc[0, "reason"]


def test_main_keeps_extracts_out_of_cosing_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "cosing"
    out_dir.mkdir()
    (out_dir / "cosing.csv").write_text(
        "inci name,cas,function\n"
        "ALOE BARBADENSIS EXTRACT,85507-69-3,Skin Conditioning\n"
        "NIACINAMIDE,98-92-0,Skin Conditioning\n",
        encoding="utf-8",
    )
    (out_dir / ".pubchem_cache.json").write_text(
        json.dumps(
            {
                "smiles_for:85507-69-3": (
                    "O=[N+]([O-])c1ccc(/C=C/C(=O)c2ccc(O)cc2)cc1"
                )
            }
        )
    )
    requested: list[str] = []

    def fake_fetch(identifier: str) -> str | None:
        requested.append(identifier)
        return "NC(=O)c1cccnc1" if identifier == "98-92-0" else None

    monkeypatch.setattr(cosing, "_fetch_smiles_from_identifier", fake_fetch)
    monkeypatch.setattr(cosing.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cosing.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage0_cosing.py",
            "--out-dir",
            str(out_dir),
            "--min-resolved-count",
            "1",
            "--min-resolution-fraction",
            "0",
            "--pubchem-workers",
            "1",
            "--progress-every",
            "0",
        ],
    )

    cosing.main()

    frame = pd.read_parquet(out_dir / "cosing.parquet")
    assert list(frame["inci_name"]) == ["NIACINAMIDE"]
    assert "85507-69-3" not in requested
    assert "ALOE BARBADENSIS EXTRACT" not in requested
    curation = pd.read_csv(tmp_path / "curation" / cosing.CURATION_RECORD_NAME)
    assert list(curation["inci_name"]) == ["ALOE BARBADENSIS EXTRACT"]
    assert curation.loc[0, "entity_class"] == cosing.ENTITY_EXTRACT
    assert curation.loc[0, "first_hit_smiles"]
    manifest = json.loads(
        (out_dir / "cosing_ingest_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["entity_class_counts"] == {
        cosing.ENTITY_EXTRACT: 1,
        cosing.ENTITY_SINGLE_COMPOUND: 1,
    }
    assert manifest["suspect_row_count"] == 1
    assert manifest["curation_record"]["rows"] == 1


def test_default_curation_dir_is_a_sibling_curation_directory(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "library"

    assert cosing.curation_dir_for(out_dir) == tmp_path / "curation"
    assert cosing.curation_dir_for(out_dir, tmp_path / "custom") == tmp_path / "custom"
