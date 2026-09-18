"""Tests for provenance-bound RCSB ligand-contact pocket fragments."""

from __future__ import annotations

import gzip
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_rcsb_contact_pocket_fragments as builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gzip_bytes(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"), mtime=0)


def _atom(
    serial: int,
    atom: str,
    comp: str,
    asym: str,
    auth_asym: str,
    seq: int,
    auth_seq: int,
    x: float,
) -> str:
    return (
        f"ATOM {serial} {atom[0]} {atom} . {comp} {asym} 1 {seq} ? {auth_seq} "
        f"{auth_asym} {atom} {comp} 1 {x:.3f} 0.000 0.000 1.00 20.00\n"
    )


def _mmcif(entry: str = "1ABC", residues: int = 5, chain: str = "T") -> str:
    lines = [
        f"data_{entry}",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.auth_comp_id",
        "_atom_site.pdbx_PDB_model_num",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    ]
    serial = 1
    for seq in range(1, residues + 1):
        for atom in ("N", "CA", "C"):
            lines.append(_atom(serial, atom, "ALA", chain, "A", seq, 100 + seq, float(seq)))
            serial += 1
    lines.append("#")
    return "\n".join(lines) + "\n"


def _write_inputs(
    tmp_path: Path,
    rows: list[dict[str, object]] | None = None,
    *,
    manifest_rows: int | None = None,
) -> tuple[Path, Path]:
    pair_rows = rows or [
        {
            "pair_id": "pair-1",
            "entry_id": "1ABC",
            "component_id": "LIG",
            "instance_id": "1ABC.L",
            "uniprot": "P11111",
            "ligand_key": "ligand-1",
            "contacted_residue_count": 1,
            "contacted_residues": ["T|103|3|ALA"],
            "is_dual_cold": True,
        }
    ]
    pairs = tmp_path / "pairs.parquet"
    pd.DataFrame(pair_rows).to_parquet(pairs, index=False)
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.rcsb-holo-contact-snapshot.v1",
                "source_license": {
                    "name": "CC0",
                    "url": "https://www.rcsb.org/pages/policies",
                },
                "usage_contract": {
                    "positive_only_direct_contact_evaluation_source": True,
                    "never_training_or_calibration": True,
                },
            }
        )
        + "\n"
    )
    manifest = tmp_path / "panel_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "skinscout.rcsb-holo-direct-contact-panel.v1",
                "inputs": {
                    "source_manifest": {
                        "path": str(source_manifest.resolve()),
                        "sha256": _sha256(source_manifest),
                        "schema_version": "skinscout.rcsb-holo-contact-snapshot.v1",
                    }
                },
                "outputs": {
                    "pairs": {
                        "path": str(pairs.resolve()),
                        "sha256": _sha256(pairs),
                        "rows": len(pair_rows) if manifest_rows is None else manifest_rows,
                    }
                },
                "contract": {
                    "positive_only": True,
                    "no_inferred_negatives": True,
                    "no_affinities_or_calibration": True,
                    "never_training_or_model_selection": True,
                    "all_ranking_rows_dual_cold": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return pairs, manifest


class _Response:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _mock_download(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes]) -> list[str]:
    urls: list[str] = []

    class Session:
        def get(self, url: str, timeout: float) -> _Response:
            del timeout
            urls.append(url)
            entry = url.rsplit("/", 1)[-1].split(".", 1)[0]
            return _Response(payloads[entry])

    monkeypatch.setattr(builder.requests, "Session", Session)
    return urls


def _args(tmp_path: Path, pairs: Path, manifest: Path) -> Namespace:
    return Namespace(
        pairs_parquet=pairs,
        panel_manifest=manifest,
        coordinate_dir=tmp_path / "coordinates",
        fragment_dir=tmp_path / "fragments",
        out_index=tmp_path / "index.csv",
        out_exclusions=tmp_path / "exclusions.csv",
        out_manifest=tmp_path / "manifest.json",
        context_residues=1,
        retries=1,
        timeout=1.0,
        workers=1,
    )


def test_actual_mmcif_parsing_download_and_contract_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, manifest = _write_inputs(tmp_path)
    urls = _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif())})

    builder.build_fragments(_args(tmp_path, pairs, manifest))

    index = pd.read_csv(tmp_path / "index.csv")
    assert len(index) == 1
    assert index.loc[0, "source_cif_gz_sha256"] == _sha256(tmp_path / "coordinates/1ABC.cif.gz")
    fragment = tmp_path / "fragments" / index.loc[0, "fragment_path"]
    assert index.loc[0, "fragment_sha256"] == _sha256(fragment)
    text = fragment.read_text()
    assert " A   2" in text
    assert " A   4" in text
    assert " A   1" not in text
    assert json.loads(index.loc[0, "chain_id_map_json"]) == {"T": "A"}
    out_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert out_manifest["schema_version"] == "skinscout.rcsb-contact-pocket-fragments.v1"
    assert out_manifest["contract"]["purpose"] == "evaluation leakage audit only"
    assert out_manifest["contract"]["never_training"] is True
    assert out_manifest["source_license"]["url"] == "https://www.rcsb.org/pages/policies"
    assert Path(out_manifest["retrievals"][0]["path"]) == (
        tmp_path / "coordinates/1ABC.cif.gz"
    ).resolve()
    assert Path(out_manifest["retrievals"][0]["path"]).exists()
    assert out_manifest["counts"] == {
        "strict_dual_cold_pairs": 1,
        "indexed": 1,
        "excluded": 0,
        "unique_entries": 1,
    }
    assert urls == ["https://files.rcsb.org/download/1ABC.cif.gz"]


def test_panel_provenance_tamper_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, manifest = _write_inputs(tmp_path, manifest_rows=99)
    _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif())})

    with pytest.raises(builder.FragmentError, match="rows does not match"):
        builder.build_fragments(_args(tmp_path, pairs, manifest))
    assert not (tmp_path / "index.csv").exists()
    assert not (tmp_path / "fragments").exists()


def test_missing_seed_is_audited_exclusion_and_accounted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, manifest = _write_inputs(tmp_path)
    _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif(residues=2))})

    builder.build_fragments(_args(tmp_path, pairs, manifest))

    assert pd.read_csv(tmp_path / "index.csv").empty
    exclusions = pd.read_csv(tmp_path / "exclusions.csv")
    assert exclusions.to_dict("records")[0]["reason"] == "contact_residues_not_extractable"
    out_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert out_manifest["counts"]["strict_dual_cold_pairs"] == 1
    assert out_manifest["counts"]["excluded"] == 1


def test_duplicate_pair_id_fails_before_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "pair_id": "dup",
            "entry_id": "1ABC",
            "component_id": "LIG",
            "instance_id": "1ABC.L",
            "uniprot": "P11111",
            "ligand_key": "ligand-1",
            "contacted_residue_count": 1,
            "contacted_residues": ["T|103|3|ALA"],
            "is_dual_cold": True,
        },
        {
            "pair_id": "dup",
            "entry_id": "1ABC",
            "component_id": "LIG",
            "instance_id": "1ABC.M",
            "uniprot": "P22222",
            "ligand_key": "ligand-2",
            "contacted_residue_count": 1,
            "contacted_residues": ["T|104|4|ALA"],
            "is_dual_cold": True,
        },
    ]
    pairs, manifest = _write_inputs(tmp_path, rows)
    _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif())})

    with pytest.raises(builder.FragmentError, match="duplicate pair_id"):
        builder.build_fragments(_args(tmp_path, pairs, manifest))
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("bad_flag", ["False", 1, None])
def test_non_boolean_dual_cold_flag_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_flag: object
) -> None:
    rows = [
        {
            "pair_id": "pair-1",
            "entry_id": "1ABC",
            "component_id": "LIG",
            "instance_id": "1ABC.L",
            "uniprot": "P11111",
            "ligand_key": "ligand-1",
            "contacted_residue_count": 1,
            "contacted_residues": ["T|103|3|ALA"],
            "is_dual_cold": bad_flag,
        }
    ]
    pairs, manifest = _write_inputs(tmp_path, rows)
    _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif())})

    with pytest.raises(builder.FragmentError, match="non-boolean is_dual_cold"):
        builder.build_fragments(_args(tmp_path, pairs, manifest))


def test_contact_count_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "pair_id": "pair-1",
            "entry_id": "1ABC",
            "component_id": "LIG",
            "instance_id": "1ABC.L",
            "uniprot": "P11111",
            "ligand_key": "ligand-1",
            "contacted_residue_count": 2,
            "contacted_residues": ["T|103|3|ALA"],
            "is_dual_cold": True,
        }
    ]
    pairs, manifest = _write_inputs(tmp_path, rows)
    _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif())})

    with pytest.raises(builder.FragmentError, match="contacted_residue_count does not match"):
        builder.build_fragments(_args(tmp_path, pairs, manifest))


def test_deterministic_fragment_and_tree_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, manifest = _write_inputs(tmp_path)
    _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif())})

    args = _args(tmp_path, pairs, manifest)
    builder.build_fragments(args)
    first_index = pd.read_csv(tmp_path / "index.csv")
    first_manifest = json.loads((tmp_path / "manifest.json").read_text())
    first_fragment_sha = first_index.loc[0, "fragment_sha256"]
    first_tree_sha = first_manifest["artifacts"]["fragment_dir"]["tree_sha256"]

    builder.build_fragments(args)
    second_index = pd.read_csv(tmp_path / "index.csv")
    second_manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert second_index.loc[0, "fragment_sha256"] == first_fragment_sha
    assert second_manifest["artifacts"]["fragment_dir"]["tree_sha256"] == first_tree_sha


def test_parallel_entry_extraction_matches_serial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "pair_id": f"pair-{index}",
            "entry_id": entry,
            "component_id": "LIG",
            "instance_id": f"{entry}.L",
            "uniprot": f"P1111{index}",
            "ligand_key": f"ligand-{index}",
            "contacted_residue_count": 1,
            "contacted_residues": ["T|103|3|ALA"],
            "is_dual_cold": True,
        }
        for index, entry in enumerate(("1ABC", "2DEF"), start=1)
    ]
    pairs, manifest = _write_inputs(tmp_path, rows)
    _mock_download(
        monkeypatch,
        {
            "1ABC": _gzip_bytes(_mmcif("1ABC")),
            "2DEF": _gzip_bytes(_mmcif("2DEF")),
        },
    )
    args = _args(tmp_path, pairs, manifest)

    builder.build_fragments(args)
    serial_index = pd.read_csv(tmp_path / "index.csv")
    serial_manifest = json.loads((tmp_path / "manifest.json").read_text())

    args.workers = 2
    builder.build_fragments(args)
    parallel_index = pd.read_csv(tmp_path / "index.csv")
    parallel_manifest = json.loads((tmp_path / "manifest.json").read_text())

    pd.testing.assert_frame_equal(parallel_index, serial_index)
    assert parallel_manifest["artifacts"]["fragment_dir"]["tree_sha256"] == (
        serial_manifest["artifacts"]["fragment_dir"]["tree_sha256"]
    )


def test_worker_count_must_be_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs, manifest = _write_inputs(tmp_path)
    _mock_download(monkeypatch, {"1ABC": _gzip_bytes(_mmcif())})
    args = _args(tmp_path, pairs, manifest)
    args.workers = 0

    with pytest.raises(builder.FragmentError, match="--workers must be >= 1"):
        builder.build_fragments(args)
