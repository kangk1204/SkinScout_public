"""Regression tests for the EC CosIng API mirror."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
MIRROR = ROOT / "scripts/stage0_mirror_cosing.py"


def load_mirror_module():
    spec = importlib.util.spec_from_file_location("stage0_mirror_cosing", MIRROR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(
        self,
        responses: dict[
            tuple[str, int, int],
            dict[str, Any] | list[dict[str, Any]],
        ],
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int, int]] = []

    def post(
        self,
        _api_url: str,
        *,
        params: dict[str, str],
        files: dict[str, tuple[str, str, str]],
        timeout: float,
    ) -> FakeResponse:
        assert params["apiKey"] == "test-key"
        assert timeout == 5.0
        assert "inciName.exact" in files["query"][1]
        key = (
            params["text"],
            int(params["pageSize"]),
            int(params["pageNumber"]),
        )
        self.calls.append(key)
        if key not in self.responses:
            raise AssertionError(f"unexpected CosIng API call: {key}")
        response = self.responses[key]
        if isinstance(response, list):
            if not response:
                raise AssertionError(
                    f"no fake responses left for CosIng API call: {key}"
                )
            return FakeResponse(response.pop(0))
        return FakeResponse(response)


def result(
    *,
    reference: str,
    substance_id: str,
    inci: str,
    cas: str = "-",
    ec: str = "-",
    functions: list[str] | None = None,
    search_reference: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "metadata": {
            "reference": [reference],
            "REFERENCE": [reference],
            "substanceId": [substance_id],
            "inciName": [inci],
            "casNo": [cas],
            "ecNo": [ec],
            "functionName": functions or ["SKIN CONDITIONING"],
            "itemType": ["ingredient"],
            "currentVersion": ["1"],
        }
    }
    if search_reference is not None:
        item["reference"] = search_reference
    return item


def payload(
    total: int,
    results: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "totalResults": total,
        "results": results or [],
        "warnings": warnings or [],
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_mirror_writes_complete_csv_and_manifest(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(2),
        ("A*", 1, 1): payload(1),
        ("B*", 1, 1): payload(1),
        ("A*", 2, 1): payload(
            1,
            [
                result(
                    reference="ref-a",
                    substance_id="1",
                    inci="Alpha",
                    cas="111-11-1",
                    ec="200-001-1",
                    functions=["HUMECTANT", "SKIN CONDITIONING"],
                )
            ],
        ),
        ("B*", 2, 1): payload(
            1,
            [result(reference="ref-b", substance_id="2", inci="Beta")],
        ),
    }

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        manifest=tmp_path / "manifest.json",
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        prefix_alphabet="AB",
        timeout_s=5.0,
        sleep_s=0,
        session=FakeSession(responses),
    )

    rows = read_csv_rows(tmp_path / "cosing.csv")
    assert [row["INCI name"] for row in rows] == ["Alpha", "Beta"]
    assert rows[0]["CAS"] == "111-11-1"
    assert rows[0]["Function"] == "HUMECTANT;SKIN CONDITIONING"
    assert manifest["expected_total"] == 2
    assert manifest["unique_records"] == 2
    assert (tmp_path / "manifest.json").exists()


def test_mirror_preserves_distinct_display_rows_under_same_reference(
    tmp_path: Path,
) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(2),
        ("A*", 1, 1): payload(2),
        ("A*", 2, 1): payload(
            2,
            [
                result(
                    reference="ref-shared",
                    substance_id="1",
                    inci="Alpha",
                    functions=["HUMECTANT"],
                ),
                result(
                    reference="ref-shared",
                    substance_id="1",
                    inci="Alpha",
                    functions=["SKIN CONDITIONING"],
                ),
            ],
        ),
    }

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        prefix_alphabet="A",
        timeout_s=5.0,
        sleep_s=0,
        session=FakeSession(responses),
    )

    rows = read_csv_rows(tmp_path / "cosing.csv")
    assert [row["Function"] for row in rows] == [
        "HUMECTANT",
        "SKIN CONDITIONING",
    ]
    assert manifest["unique_records"] == 2


def test_mirror_preserves_distinct_backend_references_under_same_display_row(
    tmp_path: Path,
) -> None:
    mirror = load_mirror_module()
    first = result(reference="ref-shared", substance_id="1", inci="Alpha")
    second = result(reference="ref-shared", substance_id="1", inci="Alpha")
    first["metadata"]["REFERENCE"] = ["backend-a"]
    second["metadata"]["REFERENCE"] = ["backend-b"]
    responses = {
        ("***", 1, 1): payload(2),
        ("A*", 1, 1): payload(2),
        ("A*", 2, 1): payload(2, [first, second]),
    }

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        prefix_alphabet="A",
        timeout_s=5.0,
        sleep_s=0,
        session=FakeSession(responses),
    )

    rows = read_csv_rows(tmp_path / "cosing.csv")
    assert [row["Reference"] for row in rows] == ["ref-shared", "ref-shared"]
    assert manifest["unique_records"] == 2


def test_mirror_preserves_distinct_search_documents_with_same_metadata(
    tmp_path: Path,
) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(2),
        ("A*", 1, 1): payload(2),
        ("A*", 2, 1): payload(
            2,
            [
                result(
                    reference="ref-shared",
                    search_reference="search-doc-1",
                    substance_id="1",
                    inci="Alpha",
                ),
                result(
                    reference="ref-shared",
                    search_reference="search-doc-2",
                    substance_id="1",
                    inci="Alpha",
                ),
            ],
        ),
    }

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        prefix_alphabet="A",
        timeout_s=5.0,
        sleep_s=0,
        session=FakeSession(responses),
    )

    rows = read_csv_rows(tmp_path / "cosing.csv")
    assert [row["INCI name"] for row in rows] == ["Alpha", "Alpha"]
    assert manifest["unique_records"] == 2


def test_mirror_splits_large_prefix_windows_and_deduplicates(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(3),
        ("A*", 1, 1): payload(3),
        ("AA*", 1, 1): payload(2),
        ("AB*", 1, 1): payload(2),
        ("B*", 1, 1): payload(0),
        ("AA*", 2, 1): payload(
            2,
            [
                result(reference="ref-a", substance_id="1", inci="Alpha"),
                result(reference="ref-b", substance_id="2", inci="Alpha Beta"),
            ],
        ),
        ("AB*", 2, 1): payload(
            2,
            [
                result(reference="ref-b", substance_id="2", inci="Alpha Beta"),
                result(reference="ref-c", substance_id="3", inci="Beta Alpha"),
            ],
        ),
    }
    session = FakeSession(responses)

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=2,
        prefix_alphabet="AB",
        timeout_s=5.0,
        sleep_s=0,
        session=session,
    )

    assert sorted(row["Reference"] for row in read_csv_rows(tmp_path / "cosing.csv")) == [
        "ref-a",
        "ref-b",
        "ref-c",
    ]
    assert [stat["query_text"] for stat in manifest["prefix_windows"]] == [
        "AA*",
        "AB*",
    ]
    assert ("A*", 2, 1) not in session.calls


def test_mirror_residual_full_query_covers_non_prefix_records(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(2),
        ("A*", 1, 1): payload(1),
        ("A*", 2, 1): payload(
            1,
            [result(reference="ref-a", substance_id="1", inci="Alpha")],
        ),
        ("***", 2, 1): payload(
            2,
            [
                result(reference="ref-a", substance_id="1", inci="Alpha"),
                result(reference="ref-symbol", substance_id="2", inci="(Beta)"),
            ],
        ),
    }

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        prefix_alphabet="A",
        timeout_s=5.0,
        sleep_s=0,
        session=FakeSession(responses),
    )

    rows = read_csv_rows(tmp_path / "cosing.csv")
    assert sorted(row["Reference"] for row in rows) == ["ref-a", "ref-symbol"]
    assert manifest["residual_full_query_fetched_rows"] == 2
    assert manifest["residual_full_query_new_unique_rows"] == 1


def test_mirror_suffix_residual_covers_full_query_limit(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(3),
        ("A*", 1, 1): payload(1),
        ("A*", 2, 1): payload(
            1,
            [result(reference="ref-a", substance_id="1", inci="Alpha")],
        ),
        ("***", 2, 1): payload(
            3,
            [result(reference="ref-a", substance_id="1", inci="Alpha")],
        ),
        ("***", 2, 2): payload(
            3,
            warnings=["The number of returned results exceed the search limit"],
        ),
        ("*Z", 1, 1): payload(2),
        ("*Z", 2, 1): payload(
            2,
            [
                result(reference="ref-symbol", substance_id="2", inci="(Symbolz"),
                result(reference="ref-dash", substance_id="3", inci="-Dashz"),
            ],
        ),
    }

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        max_suffix_depth=1,
        prefix_alphabet="A",
        residual_suffix_alphabet="Z",
        timeout_s=5.0,
        sleep_s=0,
        max_retries=0,
        session=FakeSession(responses),
    )

    rows = read_csv_rows(tmp_path / "cosing.csv")
    assert sorted(row["Reference"] for row in rows) == [
        "ref-a",
        "ref-dash",
        "ref-symbol",
    ]
    assert "search limit" in manifest["residual_full_query_error"]
    assert manifest["residual_suffix_windows"][0]["query_text"] == "*Z"
    assert manifest["residual_suffix_windows"][0]["new_unique_rows"] == 2


def test_default_suffix_residual_covers_greek_token_rows(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(2),
        ("A*", 1, 1): payload(1),
        ("A*", 2, 1): payload(
            1,
            [result(reference="ref-a", substance_id="1", inci="Alpha")],
        ),
        ("***", 2, 1): payload(
            2,
            [result(reference="ref-a", substance_id="1", inci="Alpha")],
        ),
        ("***", 2, 2): payload(
            2,
            warnings=["The number of returned results exceed the search limit"],
        ),
        ("*Α", 2, 1): payload(
            1,
            [
                result(
                    reference="ref-greek",
                    substance_id="2",
                    inci="[3R-(3α,3aβ,7β,8aα)] Example",
                )
            ],
        ),
    }
    for char in mirror.DEFAULT_RESIDUAL_SUFFIX_ALPHABET:
        responses.setdefault((f"*{char}", 1, 1), payload(0))
    responses[("*Α", 1, 1)] = payload(1)

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        max_suffix_depth=1,
        prefix_alphabet="A",
        timeout_s=5.0,
        sleep_s=0,
        max_retries=0,
        session=FakeSession(responses),
    )

    rows = read_csv_rows(tmp_path / "cosing.csv")
    assert sorted(row["Reference"] for row in rows) == ["ref-a", "ref-greek"]
    assert manifest["residual_suffix_windows"][0]["query_text"] == "*Α"
    assert manifest["residual_suffix_windows"][0]["new_unique_rows"] == 1


def test_mirror_fails_closed_when_unique_count_is_incomplete(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    (tmp_path / "cosing.csv").write_text("stale\n")
    (tmp_path / "cosing_api_manifest.json").write_text("stale\n")
    responses = {
        ("***", 1, 1): payload(2),
        ("A*", 1, 1): payload(1),
        ("B*", 1, 1): payload(0),
        ("A*", 2, 1): payload(
            1,
            [result(reference="ref-a", substance_id="1", inci="Alpha")],
        ),
        ("***", 2, 1): payload(
            2,
            [result(reference="ref-a", substance_id="1", inci="Alpha")],
        ),
    }

    with pytest.raises(mirror.MirrorError, match="mirror incomplete"):
        mirror.mirror_cosing_api(
            out_dir=tmp_path,
            api_url="https://example.test/search",
            api_key="test-key",
            page_size=2,
            max_window_results=2,
            max_prefix_depth=1,
            prefix_alphabet="AB",
            residual_suffix_alphabet="",
            timeout_s=5.0,
            sleep_s=0,
            session=FakeSession(responses),
        )

    assert not (tmp_path / "cosing.csv").exists()
    assert not (tmp_path / "cosing_api_manifest.json").exists()


def test_mirror_fails_closed_on_api_warnings(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(1),
        ("A*", 1, 1): payload(1),
        ("A*", 2, 1): payload(1, warnings=["The number of returned results exceed"]),
    }

    with pytest.raises(mirror.MirrorError, match="returned warnings"):
        mirror.mirror_cosing_api(
            out_dir=tmp_path,
            api_url="https://example.test/search",
            api_key="test-key",
            page_size=2,
            max_window_results=2,
            max_prefix_depth=1,
            prefix_alphabet="A",
            timeout_s=5.0,
            sleep_s=0,
            max_retries=0,
            session=FakeSession(responses),
        )

    assert not (tmp_path / "cosing.csv").exists()


def test_mirror_retries_transient_api_warnings(tmp_path: Path) -> None:
    mirror = load_mirror_module()
    responses = {
        ("***", 1, 1): payload(1),
        ("A*", 1, 1): payload(1),
        ("A*", 2, 1): [
            payload(1, warnings=["ingestion exception"]),
            payload(
                1,
                [result(reference="ref-a", substance_id="1", inci="Alpha")],
            ),
        ],
    }
    session = FakeSession(responses)

    manifest = mirror.mirror_cosing_api(
        out_dir=tmp_path,
        api_url="https://example.test/search",
        api_key="test-key",
        page_size=2,
        max_window_results=2,
        max_prefix_depth=1,
        prefix_alphabet="A",
        timeout_s=5.0,
        sleep_s=0,
        max_retries=1,
        retry_backoff_s=0,
        session=session,
    )

    assert manifest["unique_records"] == 1
    assert session.calls.count(("A*", 2, 1)) == 2
