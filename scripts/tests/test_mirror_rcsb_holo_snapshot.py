from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import requests


ROOT = Path(__file__).resolve().parents[2]
MIRROR = ROOT / "scripts/mirror_rcsb_holo_snapshot.py"


def load_mirror_module():
    spec = importlib.util.spec_from_file_location("mirror_rcsb_holo_snapshot", MIRROR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[dict[str, Any] | Exception | FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append((url, json, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected HTTP call to {url}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, FakeResponse):
            return response
        return FakeResponse(response)


def search_page(total: int, ids: list[str]) -> dict[str, Any]:
    return {
        "total_count": total,
        "result_set": [{"identifier": entry_id} for entry_id in ids],
    }


def entry(entry_id: str, *, release: str = "2025-03-01") -> dict[str, Any]:
    return {
        "rcsb_id": entry_id,
        "rcsb_accession_info": {"initial_release_date": release},
        "exptl": [{"method": "X-RAY DIFFRACTION"}],
        "rcsb_entry_info": {"resolution_combined": [1.8]},
        "rcsb_primary_citation": {
            "pdbx_database_id_DOI": "10.1000/test",
            "pdbx_database_id_PubMed": "12345",
            "title": "Test structure",
            "year": 2025,
        },
        "polymer_entities": [
            {
                "rcsb_id": "1",
                "rcsb_entity_source_organism": [
                    {
                        "ncbi_taxonomy_id": 9606,
                        "scientific_name": "Homo sapiens",
                        "taxonomy_lineage": [{"id": 9606, "name": "Homo sapiens"}],
                    }
                ],
                "rcsb_polymer_entity_container_identifiers": {
                    "asym_ids": ["A"],
                    "auth_asym_ids": ["A"],
                    "uniprot_ids": ["P12345"],
                    "reference_sequence_identifiers": [
                        {
                            "database_accession": "P12345",
                            "database_name": "UniProt",
                            "provenance_source": "SIFTS",
                        }
                    ],
                },
                "polymer_entity_instances": [
                    {
                        "rcsb_id": f"{entry_id}.A",
                        "rcsb_polymer_entity_instance_container_identifiers": {
                            "asym_id": "A",
                            "auth_asym_id": "A",
                        },
                    }
                ],
            }
        ],
        "nonpolymer_entities": [
            {
                "rcsb_id": "2",
                "rcsb_nonpolymer_entity_container_identifiers": {
                    "auth_asym_ids": ["B"],
                    "asym_ids": ["B"],
                    "entity_id": "2",
                    "nonpolymer_comp_id": "LIG",
                },
                "nonpolymer_comp": {
                    "chem_comp": {
                        "id": "LIG",
                        "name": "Ligand",
                        "formula": "C10 H12 O3",
                        "formula_weight": 180.2,
                        "type": "NON-POLYMER",
                    },
                    "rcsb_chem_comp_descriptor": {
                        "SMILES": "CCO",
                        "InChI": "InChI=1S/test",
                        "InChIKey": "TEST",
                        "SMILES_stereo": "CCO",
                    },
                },
                "rcsb_nonpolymer_entity_annotation": [
                    {
                        "annotation_id": "ann-1",
                        "type": "SUBJECT_OF_INVESTIGATION",
                        "name": "subject",
                        "description": "PDB-native annotation",
                        "provenance_source": "PDB",
                    }
                ],
                "nonpolymer_entity_instances": [
                    {
                        "rcsb_id": f"{entry_id}.B",
                        "rcsb_nonpolymer_entity_instance_container_identifiers": {
                            "asym_id": "B",
                            "auth_asym_id": "B",
                            "auth_seq_id": "1",
                            "comp_id": "LIG",
                            "entity_id": "2",
                        },
                        "rcsb_nonpolymer_instance_annotation": [
                            {
                                "annotation_id": "ann-2",
                                "type": "SUBJECT_OF_INVESTIGATION",
                                "name": "subject",
                                "description": "PDB-native annotation",
                                "provenance_source": "PDB",
                            }
                        ],
                        "rcsb_target_neighbors": [
                            {
                                "atom_id": "C1",
                                "comp_id": "LIG",
                                "distance": 3.2,
                                "target_asym_id": "A",
                                "target_atom_id": "NE2",
                                "target_auth_seq_id": "10",
                                "target_comp_id": "HIS",
                                "target_entity_id": "1",
                                "target_is_bound": True,
                                "target_seq_id": 10,
                            }
                        ],
                        "rcsb_nonpolymer_struct_conn": [
                            {
                                "connect_partner": {
                                    "label_alt_id": None,
                                    "label_asym_id": "B",
                                    "label_atom_id": "C1",
                                    "label_comp_id": "LIG",
                                    "label_seq_id": None,
                                    "symmetry": "1_555",
                                },
                                "connect_target": {
                                    "auth_asym_id": "A",
                                    "auth_seq_id": "10",
                                    "label_alt_id": None,
                                    "label_asym_id": "A",
                                    "label_atom_id": "NE2",
                                    "label_comp_id": "HIS",
                                    "label_seq_id": "10",
                                    "symmetry": "1_555",
                                },
                                "connect_type": "metalc",
                                "description": "contact",
                                "dist_value": 2.1,
                                "id": "conn1",
                                "role": "ligand",
                                "value_order": "sing",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def data_page(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"data": {"entries": entries}}


def read_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_mirror_happy_path_writes_sorted_jsonl_and_manifest(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    raw = tmp_path / "raw.jsonl.gz"
    manifest_path = tmp_path / "manifest.json"
    session = FakeSession(
        [
            search_page(2, ["2BBB", "1AAA"]),
            data_page([entry("1AAA"), entry("2BBB")]),
        ]
    )

    manifest = mirror.mirror_rcsb_holo_snapshot(
        raw_jsonl_gz=raw,
        manifest=manifest_path,
        release_cutoff="2025-01-01",
        batch_size=2,
        timeout_s=5.0,
        max_retries=0,
        session=session,
    )

    records = read_records(raw)
    assert [record["entry_id"] for record in records] == ["1AAA", "2BBB"]
    assert records[0]["human_polymer_uniprot_asym_mappings"] == [
        {
            "entity_id": "1",
            "instance_id": "1AAA.A",
            "asym_id": "A",
            "auth_asym_id": "A",
            "uniprot_id": "P12345",
            "reference_database": "UniProt",
            "reference_provenance_source": "SIFTS",
        }
    ]
    assert records[0]["rcsb_target_neighbors"][0]["neighbors"][0]["target_asym_id"] == "A"
    assert manifest["schema_version"] == mirror.SCHEMA_VERSION
    assert manifest["candidate_count"] == 2
    assert manifest["raw_jsonl_gz"]["rows"] == 2
    assert manifest["raw_jsonl_gz"]["bytes"] == raw.stat().st_size
    assert manifest["raw_jsonl_gz"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert manifest["source_license"]["name"] == "CC0"
    assert manifest["usage_contract"]["never_training_or_calibration"] is True
    assert manifest["filters"]["nonpolymer_molecular_weight_gt"] == 50.0
    assert "DrugBank" in manifest["usage_contract"]["forbidden_external_mappings"]
    assert json.loads(manifest_path.read_text())["query_sha256s"] == manifest["query_sha256s"]
    assert manifest["candidate_record_counts"] == {
        "with_human_uniprot_asym_mapping": 2,
        "without_human_uniprot_asym_mapping": 0,
        "with_primary_citation": 2,
        "without_primary_citation": 0,
    }


def test_search_pagination_and_graphql_batching(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    session = FakeSession(
        [
            search_page(3, ["1AAA", "2BBB"]),
            search_page(3, ["3CCC"]),
            data_page([entry("1AAA"), entry("2BBB")]),
            data_page([entry("3CCC")]),
        ]
    )

    mirror.mirror_rcsb_holo_snapshot(
        raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
        manifest=tmp_path / "manifest.json",
        batch_size=2,
        max_retries=0,
        retry_backoff_s=0,
        session=session,
    )

    search_calls = [call for call in session.calls if call[0] == mirror.SEARCH_API_URL]
    graphql_calls = [call for call in session.calls if call[0] == mirror.DATA_API_GRAPHQL_URL]
    assert [call[1]["request_options"]["paginate"]["start"] for call in search_calls] == [0, 2]
    assert [call[1]["variables"]["ids"] for call in graphql_calls] == [
        ["1AAA", "2BBB"],
        ["3CCC"],
    ]


def test_transient_http_failures_are_retried(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    session = FakeSession(
        [
            requests.Timeout("slow"),
            search_page(1, ["1AAA"]),
            FakeResponse({}, status_code=503, text="try later"),
            data_page([entry("1AAA")]),
        ]
    )

    manifest = mirror.mirror_rcsb_holo_snapshot(
        raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
        manifest=tmp_path / "manifest.json",
        batch_size=1,
        max_retries=1,
        retry_backoff_s=0,
        session=session,
    )

    assert manifest["raw_jsonl_gz"]["rows"] == 1
    assert len(session.calls) == 4


@pytest.mark.parametrize(
    ("bad_payload", "message"),
    [
        ({"result_set": []}, "missing valid total_count"),
        ({"total_count": 1}, "missing result_set"),
    ],
)
def test_search_malformed_payloads_fail_closed(
    tmp_path: Path,
    bad_payload: dict[str, Any],
    message: str,
) -> None:
    mirror = load_mirror_module()

    with pytest.raises(mirror.MirrorError, match=message):
        mirror.mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
            manifest=tmp_path / "manifest.json",
            batch_size=2,
            max_retries=0,
            session=FakeSession([bad_payload]),
        )


def test_graphql_incomplete_response_fails_closed(tmp_path: Path) -> None:
    mirror = load_mirror_module()

    with pytest.raises(mirror.MirrorError, match="batch completeness mismatch"):
        mirror.mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
            manifest=tmp_path / "manifest.json",
            batch_size=2,
            max_retries=0,
            session=FakeSession(
                [
                    search_page(2, ["1AAA", "2BBB"]),
                    data_page([entry("1AAA")]),
                ]
            ),
        )


def test_candidate_without_uniprot_mapping_is_preserved_for_panel_audit(
    tmp_path: Path,
) -> None:
    mirror = load_mirror_module()
    record = entry("1AAA")
    record["polymer_entities"][0][
        "rcsb_polymer_entity_container_identifiers"
    ]["reference_sequence_identifiers"] = []

    manifest = mirror.mirror_rcsb_holo_snapshot(
        raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
        manifest=tmp_path / "manifest.json",
        batch_size=1,
        max_retries=0,
        session=FakeSession(
            [
                search_page(1, ["1AAA"]),
                data_page([record]),
            ]
        ),
    )

    mirrored = read_records(tmp_path / "raw.jsonl.gz")
    assert mirrored[0]["human_polymer_uniprot_asym_mappings"] == []
    assert manifest["candidate_record_counts"]["without_human_uniprot_asym_mapping"] == 1


def test_non_pdb_subject_annotation_fails_closed(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    record = entry("1AAA")
    record["nonpolymer_entities"][0]["rcsb_nonpolymer_entity_annotation"][0][
        "provenance_source"
    ] = "DrugBank"
    record["nonpolymer_entities"][0]["nonpolymer_entity_instances"][0][
        "rcsb_nonpolymer_instance_annotation"
    ][0]["provenance_source"] = "DrugBank"

    with pytest.raises(mirror.MirrorError, match="no SUBJECT_OF_INVESTIGATION"):
        mirror.mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
            manifest=tmp_path / "manifest.json",
            batch_size=1,
            max_retries=0,
            session=FakeSession(
                [
                    search_page(1, ["1AAA"]),
                    data_page([record]),
                ]
            ),
        )


def test_entry_before_requested_release_cutoff_fails_closed(tmp_path: Path) -> None:
    mirror = load_mirror_module()

    with pytest.raises(mirror.MirrorError, match="predates release cutoff"):
        mirror.mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
            manifest=tmp_path / "manifest.json",
            release_cutoff="2025-01-01",
            batch_size=1,
            max_retries=0,
            session=FakeSession(
                [
                    search_page(1, ["1AAA"]),
                    data_page([entry("1AAA", release="2024-12-31")]),
                ]
            ),
        )


def test_search_duplicate_ids_fail_closed(tmp_path: Path) -> None:
    mirror = load_mirror_module()

    with pytest.raises(mirror.MirrorError, match="duplicate entry IDs"):
        mirror.mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=tmp_path / "raw.jsonl.gz",
            manifest=tmp_path / "manifest.json",
            batch_size=2,
            max_retries=0,
            session=FakeSession(
                [
                    search_page(3, ["1AAA", "2BBB"]),
                    search_page(3, ["2BBB"]),
                ]
            ),
        )


def test_failure_removes_stale_outputs_and_does_not_leave_temp_files(
    tmp_path: Path,
) -> None:
    mirror = load_mirror_module()
    raw = tmp_path / "raw.jsonl.gz"
    manifest = tmp_path / "manifest.json"
    raw.write_text("stale", encoding="utf-8")
    manifest.write_text('{"stale": true}', encoding="utf-8")

    with pytest.raises(mirror.MirrorError, match="missing entries list"):
        mirror.mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=raw,
            manifest=manifest,
            batch_size=1,
            max_retries=0,
            session=FakeSession(
                [
                    search_page(1, ["1AAA"]),
                    {"data": {"entries": None}},
                ]
            ),
        )

    assert not raw.exists()
    assert not manifest.exists()
    assert not (tmp_path / ".raw.jsonl.gz.tmp").exists()
    assert not (tmp_path / ".manifest.json.tmp").exists()


def test_manifest_hashes_bind_raw_and_queries(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    raw = tmp_path / "raw.jsonl.gz"
    manifest_path = tmp_path / "manifest.json"

    manifest = mirror.mirror_rcsb_holo_snapshot(
        raw_jsonl_gz=raw,
        manifest=manifest_path,
        release_cutoff="2026-02-03",
        min_nonpolymer_mw=75.0,
        batch_size=1,
        max_retries=0,
        session=FakeSession(
            [
                search_page(1, ["1AAA"]),
                data_page([entry("1AAA", release="2026-02-04")]),
            ]
        ),
    )

    search_contract = mirror._search_payload(
        "2026-02-03", min_nonpolymer_mw=75.0, start=0, rows=1
    )
    assert manifest["release_cutoff"] == "2026-02-03"
    assert manifest["filters"]["nonpolymer_molecular_weight_gt"] == 75.0
    assert manifest["query_sha256s"]["search"] == mirror._sha256_text(
        mirror._canonical_json(search_contract)
    )
    assert manifest["query_sha256s"]["graphql"] == mirror._sha256_text(
        mirror.DATA_GRAPHQL_QUERY
    )
    assert manifest["raw_jsonl_gz"] == {
        "path": str(raw.resolve()),
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "bytes": raw.stat().st_size,
        "rows": 1,
    }


def test_manifest_normalizes_relative_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = load_mirror_module()
    monkeypatch.chdir(tmp_path)
    raw = Path("snapshot/raw.jsonl.gz")
    manifest_path = Path("snapshot/manifest.json")

    payload = mirror.mirror_rcsb_holo_snapshot(
        raw_jsonl_gz=raw,
        manifest=manifest_path,
        release_cutoff="2026-02-03",
        batch_size=1,
        max_retries=0,
        session=FakeSession(
            [
                search_page(1, ["1AAA"]),
                data_page([entry("1AAA", release="2026-02-04")]),
            ]
        ),
    )

    assert payload["raw_jsonl_gz"]["path"] == str(raw.resolve())


def test_invalid_min_nonpolymer_mw_fails_before_network_or_outputs(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    raw = tmp_path / "raw.jsonl.gz"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(mirror.MirrorError, match="min-nonpolymer-mw"):
        mirror.mirror_rcsb_holo_snapshot(
            raw_jsonl_gz=raw,
            manifest=manifest,
            min_nonpolymer_mw=float("nan"),
            session=FakeSession([]),
        )

    assert not raw.exists()
    assert not manifest.exists()
