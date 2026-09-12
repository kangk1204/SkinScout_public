"""Unit tests for Stage 4 RCSB holo candidate handling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage4_prepare_structures import (  # noqa: E402
    DEPOSIT_DATE_ATTRIBUTE,
    MIN_HOLO_CHAIN_COVERAGE,
    RESOLUTION_ATTRIBUTE,
    HoloDownloadUnavailable,
    fetch_holo_candidates,
    pocket_box_from_ligand,
    prepare_holo_structure,
    select_target_chain,
)


class FakeResponse:
    """실제 코드가 만지는 것만 흉내 낸다.

    `status_code` 와 `content` 는 나중에 필요해진 것이다. 없이 두었더니 대역이
    실제 API보다 좁아서, 404 폴백과 204 처리를 테스트가 통과시켜 버렸다.
    """

    def __init__(
        self,
        payload: Any = None,
        text: str = "",
        status_error: Exception | None = None,
        status_code: int = 200,
        content: bytes | None = None,
    ):
        self._payload = payload
        self.text = text
        self._status_error = status_error
        self.status_code = status_code
        if content is not None:
            self.content = content
        elif payload is not None:
            # 실제 응답은 JSON 본문을 content 에도 담는다. 여기서 비워 두면 대역이
            # 실제보다 좁아져, content 를 보는 코드가 시험을 통과해 버린다.
            import json as _json

            self.content = _json.dumps(payload).encode()
        else:
            self.content = text.encode()

    def raise_for_status(self) -> None:
        if self._status_error:
            raise self._status_error

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.pdb_text: dict[str, str] = {}
        self.cif_gz: dict[str, bytes] = {}
        self.last_query: dict[str, Any] | None = None

    def post(self, url: str, json: dict[str, Any], timeout: int) -> FakeResponse:
        assert "rcsbsearch" in url
        assert json["return_type"] == "entry"
        self.last_query = json
        if not self.entries:
            # 맞는 항목이 없으면 RCSB는 204에 빈 본문을 준다.
            return FakeResponse(status_code=204, content=b"")
        return FakeResponse({
            "total_count": len(self.entries),
            "result_set": [{"identifier": pdb_id} for pdb_id in self.entries],
        })

    def get(self, url: str, timeout: int) -> FakeResponse:
        if "/core/entry/" in url:
            pdb_id = url.rsplit("/", 1)[-1].upper()
            return FakeResponse(self.entries[pdb_id])
        if url.endswith(".cif.gz"):
            pdb_id = Path(url).name[: -len(".cif.gz")].upper()
            if pdb_id not in self.cif_gz:
                return FakeResponse(status_code=404, content=b"")
            return FakeResponse(status_code=200, content=self.cif_gz[pdb_id])
        pdb_id = Path(url).stem.upper()
        if pdb_id not in self.pdb_text:
            return FakeResponse(status_code=404, content=b"")
        return FakeResponse(text=self.pdb_text[pdb_id])


def test_fetch_holo_candidates_filters_date_and_requires_ligand() -> None:
    session = FakeSession()
    session.entries = {
        "OLD1": {
            "rcsb_accession_info": {"deposit_date": "2020-01-01"},
            "rcsb_entry_info": {"nonpolymer_bound_components": ["ATP"], "resolution_combined": [2.5]},
        },
        "WAT1": {
            "rcsb_accession_info": {"deposit_date": "2024-01-01"},
            "rcsb_entry_info": {"nonpolymer_bound_components": ["HOH", "NA"], "resolution_combined": [1.5]},
        },
        "GOOD": {
            "rcsb_accession_info": {"deposit_date": "2024-02-01"},
            "rcsb_entry_info": {"nonpolymer_bound_components": ["NAD", "HOH"], "resolution_combined": [1.8]},
        },
    }

    candidates = fetch_holo_candidates("P12345", "2023-10-01", session=session)
    assert [c.pdb_id for c in candidates] == ["GOOD"]
    assert candidates[0].ligands == ("NAD",)
    assert candidates[0].resolution == 1.8


def test_fetch_holo_candidates_sorts_by_resolution() -> None:
    session = FakeSession()
    session.entries = {
        "LOWR": {
            "rcsb_accession_info": {"deposit_date": "2024-01-01"},
            "rcsb_entry_info": {"nonpolymer_bound_components": ["ATP"], "resolution_combined": [3.0]},
        },
        "HIGR": {
            "rcsb_accession_info": {"deposit_date": "2024-01-01"},
            "rcsb_entry_info": {"nonpolymer_bound_components": ["ATP"], "resolution_combined": [1.4]},
        },
    }

    candidates = fetch_holo_candidates("P12345", "2023-10-01", session=session)
    assert [c.pdb_id for c in candidates] == ["HIGR", "LOWR"]


def test_the_cutoff_date_is_pushed_into_the_query_not_applied_afterwards() -> None:
    """받은 뒤에 거르면 항목이 많은 표적에서 최근 구조가 아예 조회되지 않는다.

    P00918(탄산탈수효소 II)은 실험 구조가 1,249개이고 2023-10-01 이후 기탁만
    126개인데, 정렬 없이 앞 100개를 받아 오면 12CA·1A42 같은 옛 구조뿐이라 날짜로
    거르고 나면 0개가 된다. 이 스테이지의 목적인 "유출 안전한 실험 구조 사용"이
    그래서 한 번도 작동하지 않았다.
    """
    session = FakeSession()
    session.entries["8ZZZ"] = {
        "rcsb_accession_info": {"deposit_date": "2024-02-01"},
        "rcsb_entry_info": {"nonpolymer_bound_components": ["LIG"], "resolution_combined": [1.4]},
    }
    fetch_holo_candidates("P00918", "2023-10-01", session=session)

    query = session.last_query
    assert query is not None
    serialised = str(query)
    assert DEPOSIT_DATE_ATTRIBUTE in serialised, "날짜 조건이 질의에 없습니다"
    assert "2023-10-01" in serialised
    assert "greater_or_equal" in serialised
    # 창의 기준과 선택의 기준이 같아야 한다. 최신순으로 창을 잡고 해상도로 고르면,
    # 해상도로는 이겼을 구조가 창 밖에 있어 조회조차 되지 않는다 - 실측으로
    # P05067 의 8X52 는 42번째, P10636 의 9GG8 은 50번째였고 두 표적 모두 홀로
    # 후보 0건으로 AlphaFold 에 떨어졌다.
    sort = query["request_options"].get("sort")
    assert sort, "정렬이 없으면 앞쪽이 임의로 뽑힌다"
    assert sort[0]["sort_by"] == RESOLUTION_ATTRIBUTE, (
        "후보는 해상도로 고르므로 창도 해상도 순이어야 합니다"
    )
    assert sort[0]["direction"] == "asc"


def test_no_matching_entry_is_an_answer_not_a_crash() -> None:
    """RCSB는 결과 0건에 204와 빈 본문을 준다. raise_for_status 는 통과하고
    .json() 이 그 자리에서 터진다."""
    session = FakeSession()
    assert fetch_holo_candidates("P23219", "2023-10-01", session=session) == []




# ------------------------------------------ 실제 항목이 만드는 모양으로 짓는다
#
# 예전의 유일한 변환 테스트는 원자 2개, 체인 하나, 리간드 없음짜리 mmCIF 로
# `"ATOM" in body` 만 단언했다 - 프로덕션 코드가 하던 것과 똑같이 약한 검사다.
# 그 대역은 PDB 형식이 담지 못하는 것을 하나도 표현할 수 없어서, 실제 경로가
# Biopython 자신도 다시 읽지 못하는 파일을 쓰는 동안 계속 통과했다.


def _cif(atoms: list[tuple[str, str, str, str, str, float, float, float]]) -> str:
    """atom_site 루프만 있는 최소 mmCIF.

    각 항목은 (group_PDB, atom_id, comp_id, asym_id, seq_id, x, y, z).
    """
    header = """data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""
    lines = []
    for i, (group, atom, comp, asym, seq, x, y, z) in enumerate(atoms, start=1):
        element = atom[0]
        lines.append(
            f"{group} {i} {element} {atom} . {comp} {asym} 1 {seq} ? "
            f"{x:.3f} {y:.3f} {z:.3f} 1.00 10.00 {seq} {asym} 1"
        )
    return header + "\n".join(lines) + "\n"


def _protein_chain(asym: str, n: int, offset: float = 0.0):
    """CA 를 가진 글라이신 n 개. 서열은 'G' * n 이 된다."""
    atoms = []
    for i in range(1, n + 1):
        atoms.append(("ATOM", "N", "GLY", asym, str(i), offset + i, 0.0, 0.0))
        atoms.append(("ATOM", "CA", "GLY", asym, str(i), offset + i, 1.0, 0.0))
    return atoms


def _gz(text: str) -> bytes:
    import gzip

    return gzip.compress(text.encode())


MONOMER = "G" * 20


def test_a_five_character_ligand_code_does_not_reach_the_written_file(
    tmp_path: Path,
) -> None:
    """2023년부터 PDB 는 새 리간드에 5글자 CCD 코드를 발급한다.

    Biopython 의 서식 문자열은 `%3s` 인데 그것은 최소 너비지 자르기가 아니다.
    잔기 이름이 18-22 열로 넘치면 체인 id·잔기 번호·좌표가 두 칸씩 밀리고,
    Biopython 자신의 PDBParser 도 그 파일을 읽지 못한다. 그런데도 "ATOM 이라는
    글자가 있는가" 검사는 통과했다(실측: 30TA, 9HWM, 30XX).
    """
    from Bio.PDB import PDBParser

    session = FakeSession()
    session.cif_gz["8ZZZ"] = _gz(_cif(
        _protein_chain("A", 20)
        + [("HETATM", "N1", "A1J8Z", "A", "301", 5.0, 5.0, 5.0),
           ("HETATM", "C1", "A1J8Z", "A", "301", 6.0, 5.0, 5.0)]
    ))
    out = tmp_path / "8ZZZ.pdb"
    prepared = prepare_holo_structure("8ZZZ", ("A1J8Z",), MONOMER, out, session=session)

    assert prepared.fmt == "cif_converted"
    body = out.read_text()
    assert "A1J8Z" not in body, "PDB 형식이 담지 못하는 잔기 이름이 파일에 들어갔습니다"
    # 진짜 검사: 다시 읽힌다.
    PDBParser(QUIET=True).get_structure("8ZZZ", str(out))
    for line in body.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            assert len(line) <= 80, f"열이 밀렸습니다: {line!r}"


def test_the_written_structure_is_one_chain_not_the_whole_assembly(
    tmp_path: Path,
) -> None:
    """cutoff 이후 기탁물은 대개 복합체다.

    유일한 소비자인 stage5 의 `pdb_sequence` 는 모든 체인의 CA 를 파일 순서대로
    이어 붙인다. 실측으로 Q7L0Y3 은 6체인 1,304잔기가 되어 단량체(312)의 4.18배짜리,
    존재하지 않는 키메라 단백질이 Boltz-2 에 넘어갔다.
    """
    from Bio.PDB import PDBParser

    session = FakeSession()
    session.cif_gz["8ZZZ"] = _gz(_cif(
        _protein_chain("A", 20)
        + _protein_chain("B", 18, offset=100.0)
        + _protein_chain("C", 15, offset=200.0)
        + [("HETATM", "C1", "LIG", "A", "301", 5.0, 5.0, 5.0)]
    ))
    out = tmp_path / "8ZZZ.pdb"
    prepared = prepare_holo_structure("8ZZZ", ("LIG",), MONOMER, out, session=session)

    assert prepared.chain_id == "A"
    structure = PDBParser(QUIET=True).get_structure("8ZZZ", str(out))
    assert [c.id for m in structure for c in m] == ["A"]
    assert sum(1 for _ in structure.get_residues()) == 20


def test_an_entry_holding_only_a_fragment_of_the_target_is_refused(
    tmp_path: Path,
) -> None:
    """맞은 체인이 단량체의 절반도 안 되면 그것은 표적이 아니라 조각이다.

    조각의 친화도를 표적의 친화도로 읽으면 그 수치는 다른 분자의 것이다.
    실측으로 Q9Y251 의 9S8W 는 맞은 체인이 단량체의 15% 였다.
    """
    session = FakeSession()
    session.cif_gz["8ZZZ"] = _gz(_cif(
        _protein_chain("A", 4)
        + [("HETATM", "C1", "LIG", "A", "301", 5.0, 5.0, 5.0)]
    ))
    with pytest.raises(HoloDownloadUnavailable, match="단량체의"):
        prepare_holo_structure("8ZZZ", ("LIG",), MONOMER, tmp_path / "x.pdb",
                               session=session)


def test_the_docking_box_is_built_from_the_holo_ligand_not_the_alphafold_pocket(
    tmp_path: Path,
) -> None:
    """P2Rank 포켓은 AlphaFold 모델 좌표계에 있다. 실험 구조는 결정 좌표계다.

    예전에는 그 매니페스트를 홀로 구조 옆에 그대로 복사했다 - 실측으로 상자
    중심에서 결합 리간드까지 20.6 A 에서 337.3 A 까지 떨어져 있었고, 7개 중
    상자 안에 리간드가 들어간 것은 0개였다.
    """
    import math

    session = FakeSession()
    ligand_atoms = [
        ("HETATM", "C1", "LIG", "A", "301", 500.0, 500.0, 500.0),
        ("HETATM", "C2", "LIG", "A", "301", 502.0, 500.0, 500.0),
        ("HETATM", "C3", "LIG", "A", "301", 501.0, 502.0, 500.0),
    ]
    session.cif_gz["8ZZZ"] = _gz(_cif(_protein_chain("A", 20) + ligand_atoms))
    prepared = prepare_holo_structure("8ZZZ", ("LIG",), MONOMER,
                                      tmp_path / "8ZZZ.pdb", session=session)

    box = prepared.pocket_box
    assert box is not None
    pocket = box["pockets"][0]
    assert pocket["basis"] == "holo_bound_ligand"
    # 상자가 리간드를 실제로 담는다. 이것이 예전에 0/7 이던 성질이다.
    for _, _, _, _, _, x, y, z in ligand_atoms:
        assert math.dist(pocket["center"], (x, y, z)) <= pocket["radius"]
    # 그리고 AlphaFold 모델 좌표계(원점 근처)가 아니다.
    assert math.dist(pocket["center"], (0.0, 0.0, 0.0)) > 100.0


def test_an_entry_whose_declared_ligand_has_no_coordinates_is_refused(
    tmp_path: Path,
) -> None:
    """상자를 만들 수 없으면 다음 후보로 간다. AlphaFold 포켓을 갖다 쓰면 안 된다."""
    session = FakeSession()
    session.cif_gz["8ZZZ"] = _gz(_cif(_protein_chain("A", 20)))
    with pytest.raises(HoloDownloadUnavailable, match="도킹 상자"):
        prepare_holo_structure("8ZZZ", ("LIG",), MONOMER, tmp_path / "x.pdb",
                               session=session)


def test_a_legacy_pdb_download_goes_through_the_same_chain_selection(
    tmp_path: Path,
) -> None:
    """레거시 .pdb 를 텍스트로 그대로 복사하면 복합체인지도 보지 않고 나간다."""
    from Bio.PDB import PDBParser

    session = FakeSession()
    lines = []
    serial = 1
    for asym, n, off in (("A", 20, 0.0), ("B", 18, 100.0)):
        for i in range(1, n + 1):
            for atom, elem in (("N", "N"), ("CA", "C")):
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} GLY {asym}{i:4d}    "
                    f"{off + i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00           {elem}"
                )
                serial += 1
    lines.append(
        f"HETATM{serial:5d}  C1  LIG A 301    "
        f"{5.0:8.3f}{5.0:8.3f}{5.0:8.3f}  1.00 10.00           C"
    )
    session.pdb_text["8ZZZ"] = "\n".join(lines) + "\nEND\n"

    out = tmp_path / "8ZZZ.pdb"
    prepared = prepare_holo_structure("8ZZZ", ("LIG",), MONOMER, out, session=session)
    assert prepared.fmt == "pdb"
    assert prepared.chain_id == "A"
    structure = PDBParser(QUIET=True).get_structure("8ZZZ", str(out))
    assert [c.id for m in structure for c in m] == ["A"]


def test_an_entry_available_in_neither_format_says_so_instead_of_raising_http(
    tmp_path: Path,
) -> None:
    """다음 후보로 넘어가라는 신호여야 한다. HTTPError 면 실행 전체가 죽는다."""
    session = FakeSession()
    with pytest.raises(HoloDownloadUnavailable):
        prepare_holo_structure("NONE", ("LIG",), MONOMER, tmp_path / "none.pdb",
                               session=session)


def test_the_chain_is_chosen_by_matching_the_monomer_not_by_being_first() -> None:
    """검색이 맞춘 접근번호가 복합체의 부수 성분일 수 있다."""
    from Bio.PDB import MMCIFParser
    import io

    cif = _cif(_protein_chain("A", 30, offset=0.0) + _protein_chain("B", 20, offset=100.0))
    structure = MMCIFParser(QUIET=True).get_structure("T", io.StringIO(cif))
    # 단량체가 20잔기면 30잔기짜리 A 가 아니라 길이가 맞는 B 가 뽑혀야 한다.
    chain_id, coverage = select_target_chain(structure, "G" * 20)
    assert chain_id == "B"
    assert coverage == pytest.approx(1.0)


def test_the_coverage_floor_is_stated_as_a_fraction() -> None:
    assert 0.0 < MIN_HOLO_CHAIN_COVERAGE <= 1.0


def test_a_box_without_any_matching_ligand_atom_is_none() -> None:
    from Bio.PDB import MMCIFParser
    import io

    cif = _cif(_protein_chain("A", 10))
    structure = MMCIFParser(QUIET=True).get_structure("T", io.StringIO(cif))
    assert pocket_box_from_ligand(structure, "A", ("LIG",)) is None


def test_pocket_uses_the_ligand_instance_nearest_the_selected_chain() -> None:
    from Bio.PDB import MMCIFParser
    import io

    cif = _cif(
        _protein_chain("A", 10)
        + _protein_chain("B", 10, offset=100.0)
        + [("HETATM", "C1", "LIG", "A", "301", 5.0, 2.0, 0.0)]
        + [("HETATM", "C1", "LIG", "B", "301", 105.0, 2.0, 0.0)]
    )
    structure = MMCIFParser(QUIET=True).get_structure("T", io.StringIO(cif))
    box = pocket_box_from_ligand(structure, "A", ("LIG",))
    assert box is not None
    assert box["pockets"][0]["center"][0] == pytest.approx(5.0)
    assert box["n_ligand_atoms"] == 1


def test_unrelated_equal_length_chain_is_not_selected() -> None:
    from Bio.PDB import MMCIFParser
    import io

    cif = _cif(_protein_chain("A", 10))
    structure = MMCIFParser(QUIET=True).get_structure("T", io.StringIO(cif))
    chain_id, coverage = select_target_chain(structure, "A" * 10)
    assert chain_id == ""
    assert coverage == 0.0
