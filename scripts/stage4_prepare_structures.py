#!/usr/bin/env python3
"""stage4_prepare_structures.py — Prepare top-50 receptor structures.

For each candidate:
  1. Try RCSB experimental holo deposited >= --cutoff-date (data-leakage-safe).
  2. Otherwise reuse the Stage 0 cleaned AlphaFold structure.
  3. Copy the P2Rank pocket box for the chosen structure.

The docstring used to claim PROPKA protonation and cofactor labelling. Neither is
implemented here, and saying so made the stage look like it did more than it does.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

LOG = logging.getLogger("stage4.prep")
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
EXCLUDED_HOLO_COMPONENTS = {
    "HOH", "DOD", "WAT", "H2O",
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "CO", "NI",
    "SO4", "PO4", "ACT", "GOL", "EDO", "PEG",
}


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_top_targets(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"Stage 4 top target CSV is missing or empty: {path}")
    try:
        top = pd.read_csv(path, skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"Stage 4 top target CSV failed to parse: {path}: {exc}") from exc
    if "target_id" not in top.columns:
        raise SystemExit("Stage 4 top target CSV missing required column: target_id")
    invalid = [
        int(idx) for idx, value in top["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            "Stage 4 top target CSV column 'target_id' contains blank values at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    top = top.copy()
    top["target_id"] = top["target_id"].astype(str).str.strip()
    duplicate_ids = top["target_id"][top["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            "Stage 4 top target CSV contains duplicate target_id values: "
            f"{shown}{suffix}"
        )
    if top.empty:
        raise SystemExit("No Stage 3 targets available for structure preparation")
    return top


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _validate_pocket_manifest(path: Path, target_id: str) -> None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Invalid pocket manifest JSON for {target_id}: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"Pocket manifest JSON for {target_id} must be an object: {path}"
        )
    pockets = payload.get("pockets")
    if pockets is not None and not isinstance(pockets, list):
        raise SystemExit(
            f"Pocket manifest JSON for {target_id} field 'pockets' must be a list: {path}"
        )


@dataclass(frozen=True)
class HoloCandidate:
    pdb_id: str
    deposit_date: str
    ligands: tuple[str, ...]
    resolution: float | None


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


ACCESSION_ATTRIBUTE = (
    "rcsb_polymer_entity_container_identifiers."
    "reference_sequence_identifiers.database_accession"
)
DEPOSIT_DATE_ATTRIBUTE = "rcsb_accession_info.deposit_date"
RESOLUTION_ATTRIBUTE = "rcsb_entry_info.resolution_combined"
# 상세 조회는 항목당 요청 하나다. 한 표적에 수백 개를 돌면 표적 50개가 수천 요청이
# 된다. 날짜로 이미 걸렀으므로 앞쪽 일부만 상세 조회한다.
#
# 다만 **무엇의 앞쪽인가**가 중요하다. 예전에는 기탁일 최신순으로 창을 잡아 놓고
# 정작 후보는 해상도 순으로 골랐다. 창의 기준과 선택의 기준이 달라서, 해상도로는
# 이겼을 구조가 최신 40건 밖에 있으면 조회조차 되지 않았다 - 실측으로 P05067의
# 8X52(2.9 A)는 42번째, P10636의 9GG8(1.1 A)은 50번째였고 두 표적 모두 홀로
# 후보 0건으로 AlphaFold에 떨어졌다. 유출 안전성은 정렬이 아니라 질의의 날짜
# 조건이 지키므로, 창도 선택과 같은 해상도 순으로 잡는다.
MAX_ENTRY_DETAIL_FETCHES = 40


def _search_rcsb_entries(uniprot: str,
                         cutoff_date: str,
                         session: requests.Session | None = None,
                         timeout: int = 20) -> tuple[list[str], int]:
    """cutoff 이후에 기탁된 항목만, 해상도가 좋은 순으로.

    날짜 조건을 **질의에 넣는다.** 예전에는 정렬 없이 앞 100개를 가져와 받은 뒤에
    날짜로 걸렀는데, 그러면 항목이 100개를 넘는 표적에서 최근 구조가 아예 조회되지
    않는다. P00918(탄산탈수효소 II)은 실험 구조가 1,249개이고 2023-10-01 이후
    기탁만 126개인데, 받아 오던 앞 100개는 12CA·1A42 같은 옛 구조였다. 그래서 이
    스테이지의 목적인 "유출 안전한 실험 구조 사용"이 한 번도 작동하지 않고 늘
    AlphaFold로 폴백했다.

    총 몇 건이 걸렸는지도 함께 돌려준다. 0건인 것과 조회가 실패한 것은 다른 일이고,
    호출자가 그 둘을 구분해 기록해야 한다.
    """
    client = session or requests.Session()
    body = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": ACCESSION_ATTRIBUTE,
                        "operator": "exact_match",
                        "value": uniprot,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": DEPOSIT_DATE_ATTRIBUTE,
                        "operator": "greater_or_equal",
                        "value": cutoff_date,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": MAX_ENTRY_DETAIL_FETCHES},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": RESOLUTION_ATTRIBUTE, "direction": "asc"}],
        },
    }
    res = client.post(RCSB_SEARCH_URL, json=body, timeout=timeout)
    res.raise_for_status()
    # 맞는 항목이 없으면 RCSB는 204에 빈 본문을 준다. raise_for_status 는 통과하고
    # .json() 이 그 자리에서 터진다. 결과 0건은 실패가 아니라 답이므로 그렇게 읽는다.
    #
    # 204만 본다. 200인데 본문이 JSON이 아니라면 그것은 API가 달라졌다는 뜻이고,
    # 조용히 "결과 없음"으로 넘기면 안 되는 실패다.
    if res.status_code == 204:
        return [], 0
    payload = res.json()
    total = int(payload.get("total_count") or 0)
    return [r["identifier"].upper() for r in payload.get("result_set", [])], total


def _entry_holo_candidate(pdb_id: str,
                          cutoff_date: str,
                          session: requests.Session | None = None,
                          timeout: int = 20) -> HoloCandidate | None:
    client = session or requests.Session()
    res = client.get(RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=timeout)
    res.raise_for_status()
    payload = res.json()

    accession = payload.get("rcsb_accession_info", {})
    deposited = accession.get("deposit_date") or accession.get("initial_release_date")
    if not deposited or _parse_date(deposited) < _parse_date(cutoff_date):
        return None

    entry_info = payload.get("rcsb_entry_info", {})
    ligands = tuple(
        sorted(
            comp.upper()
            for comp in entry_info.get("nonpolymer_bound_components", []) or []
            if str(comp).upper() not in EXCLUDED_HOLO_COMPONENTS
        )
    )
    if not ligands:
        return None

    resolutions = entry_info.get("resolution_combined") or []
    resolution = None
    if resolutions:
        try:
            resolution = min(float(x) for x in resolutions if x is not None)
        except (TypeError, ValueError):
            resolution = None
    return HoloCandidate(
        pdb_id=pdb_id.upper(),
        deposit_date=deposited[:10],
        ligands=ligands,
        resolution=resolution,
    )


def fetch_holo_candidates(uniprot: str,
                          cutoff_date: str,
                          session: requests.Session | None = None) -> list[HoloCandidate]:
    pdb_ids, total = _search_rcsb_entries(uniprot, cutoff_date, session=session)
    if total > len(pdb_ids):
        LOG.info(
            "%s: cutoff 이후 기탁 %d건 중 해상도 상위 %d건만 상세 조회합니다.",
            uniprot, total, len(pdb_ids),
        )
    candidates = [
        cand for pdb_id in pdb_ids
        if (cand := _entry_holo_candidate(pdb_id, cutoff_date, session=session)) is not None
    ]
    return sorted(
        candidates,
        key=lambda c: (
            c.resolution if c.resolution is not None else 99.0,
            c.deposit_date,
            c.pdb_id,
        ),
    )


RCSB_CIF_GZ_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


class HoloDownloadUnavailable(RuntimeError):
    """이 항목은 받을 수 없다. 다음 후보로 넘어가라는 뜻이다."""


MIN_HOLO_CHAIN_COVERAGE = 0.5
MIN_HOLO_CHAIN_IDENTITY = 0.5


def _structure_from_bytes(pdb_id: str, payload: bytes, is_cif: bool):
    """mmCIF(gzip) 또는 PDB 텍스트를 Bio.PDB 구조로 읽는다."""
    import gzip
    import io

    from Bio.PDB import MMCIFParser, PDBParser

    text = (gzip.decompress(payload) if is_cif else payload).decode(
        "utf-8", errors="replace"
    )
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    try:
        return parser.get_structure(pdb_id, io.StringIO(text))
    except Exception as exc:  # noqa: BLE001 - 파서가 던지는 예외 종류가 넓다
        raise HoloDownloadUnavailable(
            f"{pdb_id}: 구조를 읽지 못했습니다 ({exc})"
        ) from exc


def _chain_sequence(chain) -> str:
    """CA 원자를 가진 잔기만의 1문자 서열. 리간드와 물은 빠진다."""
    from Bio.PDB.Polypeptide import protein_letters_3to1

    letters = []
    for residue in chain:
        if residue.id[0] != " " or "CA" not in residue:
            continue
        letters.append(protein_letters_3to1.get(residue.get_resname().upper(), "X"))
    return "".join(letters)


def _sequence_identity(a: str, b: str) -> float:
    """Global-alignment identity over aligned residue pairs."""
    if not a or not b:
        return 0.0
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(a, b)[0]
    matches = 0
    aligned = 0
    for (a_start, a_end), (b_start, b_end) in zip(
        alignment.aligned[0], alignment.aligned[1], strict=True
    ):
        span = min(a_end - a_start, b_end - b_start)
        aligned += span
        matches += sum(
            a[a_start + offset] == b[b_start + offset] for offset in range(span)
        )
    return matches / aligned if aligned else 0.0


def select_target_chain(structure, monomer_sequence: str) -> tuple[str, float]:
    """표적 자신을 담고 있는 auth 체인 하나를 고른다.

    cutoff 이후 기탁물은 대개 복합체다. 예전에는 그것을 통째로 써서, 유일한
    소비자인 stage5 의 `pdb_sequence`가 **모든 체인의 CA 를 파일 순서대로 이어
    붙인** 하나의 서열을 Boltz-2 에 넘겼다. 실측으로 Q7L0Y3 은 6체인 1,304잔기가
    되어 단량체(312)의 4.18배짜리, 존재하지 않는 키메라 단백질이 됐다.

    검색이 맞춘 UniProt 접근번호가 복합체의 부수 성분일 수도 있으므로, 체인은
    Stage 0 의 정제된 AlphaFold 단량체 서열과 맞대어 고른다. 그 단량체는 정의상
    표적 자신이다.
    """
    best = ("", 0.0, 0.0, 0.0)
    for model in structure:
        for chain in model:
            seq = _chain_sequence(chain)
            if not seq:
                continue
            coverage = len(seq) / max(1, len(monomer_sequence))
            identity = _sequence_identity(seq, monomer_sequence)
            # 길이비는 **대칭**으로 잰다. min(1.0, coverage) 로 두면 단량체보다
            # 훨씬 긴 체인이 벌점을 받지 않아, 복합체의 융합 파트너 쪽이 표적
            # 자신과 동점이 되고 파일 순서가 승자를 정한다.
            length_ratio = min(len(seq), len(monomer_sequence)) / max(
                1, len(seq), len(monomer_sequence)
            )
            score = identity * length_ratio
            if score > best[2]:
                best = (chain.id, coverage, score, identity)
        break  # 첫 모델만 본다(NMR 앙상블의 나머지는 같은 분자다)
    if best[3] < MIN_HOLO_CHAIN_IDENTITY:
        return "", 0.0
    return best[0], best[1]


def _write_protein_chain(structure, chain_id: str, out_pdb: Path,
                         pdb_id: str) -> None:
    """고른 체인의 **단백질 잔기만** PDB 로 쓴다.

    리간드를 함께 쓰지 않는 이유가 두 가지다.

    첫째, 하류가 쓰지 않는다. stage5 는 이 파일에서 서열만 꺼내고 리간드는
    stage1 이 만든 SDF 로 따로 받는다.

    둘째, PDB 형식이 담을 수 없다. 2023년부터 PDB 는 새 리간드에 5글자 CCD 코드를
    발급하는데 - 이 스테이지가 찾는 cutoff 이후 구조가 정확히 거기 해당한다 -
    Biopython 의 PDBIO 서식 문자열은 `%3s`이고 그것은 최소 너비지 자르기가
    아니다. 잔기 이름이 18-22 열로 넘치면서 체인 id, 잔기 번호, 좌표 세 개가
    두 칸씩 밀린다. 실측으로 30TA·9HWM·30XX 로 쓴 파일은 Biopython 자신의
    PDBParser 가 다시 읽지 못했다(`invalid literal for int(): 'A'`). 그런데도
    "ATOM 문자열이 있는가"만 보는 검사는 통과해서 그대로 하류로 나갔다.

    리간드의 위치 정보는 버리지 않는다. 도킹 상자를 만드는 데 쓰고
    (`pocket_box_from_ligand`), 어떤 리간드였는지는 매니페스트에 남는다.
    """
    from Bio.PDB import PDBIO, Select

    class _OneProteinChain(Select):
        def accept_model(self, model):
            return model.id == 0

        def accept_chain(self, chain):
            return chain.id == chain_id

        def accept_residue(self, residue):
            # 표준 잔기만. hetflag 가 빈칸이 아니면 리간드나 물이다.
            return residue.id[0] == " "

    writer = PDBIO()
    writer.set_structure(structure)
    tmp = out_pdb.with_suffix(out_pdb.suffix + ".tmp")
    try:
        writer.save(str(tmp), select=_OneProteinChain())
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HoloDownloadUnavailable(
            f"{pdb_id}: PDB 형식으로 담을 수 없는 구조입니다 ({exc})"
        ) from exc

    # 쓴 파일을 **다시 읽어** 확인한다. "ATOM 이라는 글자가 있는가"는 열이 밀린
    # 파일도 통과시킨다.
    from Bio.PDB import PDBParser

    try:
        reparsed = PDBParser(QUIET=True).get_structure(pdb_id, str(tmp))
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HoloDownloadUnavailable(
            f"{pdb_id}: 쓴 PDB 를 다시 읽지 못했습니다 ({exc})"
        ) from exc
    n_res = sum(1 for _ in reparsed.get_residues())
    if n_res == 0:
        tmp.unlink(missing_ok=True)
        raise HoloDownloadUnavailable(f"{pdb_id}: 변환 결과에 잔기가 없습니다")
    tmp.replace(out_pdb)


def pocket_box_from_ligand(structure, chain_id: str,
                           ligand_codes: tuple[str, ...],
                           radius_pad: float = 4.0) -> dict | None:
    """결합한 리간드의 좌표로 도킹 상자를 만든다. 실험 구조의 좌표계에서.

    P2Rank 포켓은 Stage 0 의 정제된 AlphaFold 모델 위에서 계산한 것이라 모델
    좌표계에 있다. 실험 구조는 결정/cryo-EM 좌표계에 있고, 두 좌표계는 겹치지
    않는다. 그런데 예전에는 그 포켓 매니페스트를 홀로 구조 옆에 그대로 복사했다 -
    실측으로 상자 중심에서 실제 결합 리간드까지 20.6 A 에서 337.3 A 까지 떨어져
    있었고, 7개 중 상자 안에 리간드가 들어간 것은 0개였다. 파일 모양이 정상적인
    AlphaFold 행과 똑같아서 화면과 하류 어디에서도 알 수 없었다.

    검색이 이미 어떤 리간드가 결합해 있는지 알고 있으므로, 그 리간드의 원자에서
    상자를 직접 만든다. 좌표계 문제가 원천적으로 사라진다.
    """
    import math

    wanted = {code.upper() for code in ligand_codes}
    target_atoms: list[tuple[float, float, float]] = []
    candidates: list[list[tuple[float, float, float]]] = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[0] == " ":
                        target_atoms.extend(
                            tuple(float(v) for v in atom.get_coord()) for atom in residue
                        )
            for residue in chain:
                if residue.id[0] == " ":
                    continue
                if residue.get_resname().upper() not in wanted:
                    continue
                candidates.append([
                    tuple(float(v) for v in atom.get_coord()) for atom in residue
                ])
        break
    if not candidates or not target_atoms:
        return None
    distances = [
        min(math.dist(ligand_atom, protein_atom)
            for ligand_atom in ligand for protein_atom in target_atoms)
        for ligand in candidates
    ]
    ranked = sorted(range(len(candidates)), key=lambda index: (distances[index], index))
    if len(ranked) > 1 and math.isclose(
        distances[ranked[0]], distances[ranked[1]], abs_tol=1e-6
    ):
        return None
    coords = candidates[ranked[0]]
    n = len(coords)
    center = [sum(c[i] for c in coords) / n for i in range(3)]
    radius = max(
        math.dist(center, c) for c in coords
    ) + radius_pad
    return {
        "pockets": [{
            "rank": 1,
            "score": None,
            "druggability": None,
            "center": [round(v, 4) for v in center],
            "radius": round(float(radius), 4),
            # 어느 구조의 어느 좌표계인지 파일 자신이 말하게 한다. 예전에는
            # 이 정보가 없어서 좌표계가 어긋나도 알아낼 방법이 없었다.
            "basis": "holo_bound_ligand",
            "ligands": sorted(wanted),
            "chain": chain_id,
        }],
        "source": "stage4_holo_ligand",
        "n_ligand_atoms": n,
    }


@dataclass(frozen=True)
class PreparedHolo:
    """받아서 체인까지 고른 홀로 구조."""
    fmt: str
    chain_id: str
    chain_coverage: float
    pocket_box: dict | None


def prepare_holo_structure(pdb_id: str,
                           ligand_codes: tuple[str, ...],
                           monomer_sequence: str,
                           out_pdb: Path,
                           session: requests.Session | None = None,
                           timeout: int = 60) -> PreparedHolo:
    """레거시 PDB를 먼저, 없으면 mmCIF를 받아 표적 체인 하나만 쓴다.

    2024년 이후 기탁된 항목에는 `.pdb` 파일이 없다. RCSB가 PDB 형식으로 담을 수
    없는 항목에 그것을 제공하지 않기 때문이다. 이 스테이지는 cutoff 이후 구조를
    찾도록 되어 있는데 정작 그 구조들이 여기 해당해서, 검색을 고치고 나니 곧바로
    404가 났다(30TA, 9HWM, 9O9N 모두 .pdb 404 / .cif 200).

    두 형식 모두 파싱해서 같은 길을 지나간다. 예전에는 레거시 `.pdb`를 텍스트로
    그대로 복사했는데, 그러면 복합체인지 아닌지도 보지 않고 하류로 나갔다.
    """
    client = session or requests.Session()
    res = client.get(RCSB_PDB_URL.format(pdb_id=pdb_id.upper()), timeout=timeout)
    if res.status_code == 200:
        payload, is_cif, fmt = res.content, False, "pdb"
    else:
        if res.status_code != 404:
            res.raise_for_status()
        cif = client.get(RCSB_CIF_GZ_URL.format(pdb_id=pdb_id.upper()), timeout=timeout)
        if cif.status_code == 404:
            raise HoloDownloadUnavailable(f"{pdb_id}: RCSB에 pdb도 cif도 없습니다")
        cif.raise_for_status()
        payload, is_cif, fmt = cif.content, True, "cif_converted"

    structure = _structure_from_bytes(pdb_id.upper(), payload, is_cif)
    chain_id, coverage = select_target_chain(structure, monomer_sequence)
    if not chain_id:
        raise HoloDownloadUnavailable(f"{pdb_id}: 단백질 체인을 찾지 못했습니다")
    if coverage < MIN_HOLO_CHAIN_COVERAGE:
        # 맞은 체인이 단량체의 절반도 안 되면 이 항목은 표적의 조각만 담고 있다.
        # 조각을 표적으로 삼아 친화도를 예측하면 그 수치는 다른 분자의 것이다.
        raise HoloDownloadUnavailable(
            f"{pdb_id}: 표적 체인 {chain_id} 이 단량체의 "
            f"{coverage:.0%}뿐입니다(최소 {MIN_HOLO_CHAIN_COVERAGE:.0%})"
        )
    box = pocket_box_from_ligand(structure, chain_id, ligand_codes)
    if box is None:
        # 검색은 결합 리간드가 있다고 했는데 좌표에는 없다. 그러면 상자를 만들 수
        # 없고, AlphaFold 포켓을 갖다 쓰면 좌표계가 어긋난다.
        raise HoloDownloadUnavailable(
            f"{pdb_id}: 결합 리간드 {','.join(ligand_codes)} 의 좌표가 없어 "
            "도킹 상자를 만들 수 없습니다"
        )
    _write_protein_chain(structure, chain_id, out_pdb, pdb_id.upper())
    return PreparedHolo(fmt=fmt, chain_id=chain_id, chain_coverage=coverage,
                        pocket_box=box)


def _monomer_sequence(cleaned_pdb: Path) -> str:
    """Stage 0 정제 AlphaFold 단량체의 서열. 체인 선택의 기준이다."""
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("monomer", str(cleaned_pdb))
    return "".join(_chain_sequence(chain)
                   for model in structure for chain in model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-csv", required=True, type=Path)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--pocket-dir", required=True, type=Path)
    parser.add_argument("--cutoff-date", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument(
        "--allow-rcsb-lookup-failure",
        action="store_true",
        help="Fall back to AlphaFold when RCSB lookup fails only for explicit degraded diagnostics.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    tmp_manifest = args.out_manifest.with_suffix(args.out_manifest.suffix + ".tmp")
    _remove_outputs(args.out_manifest, tmp_manifest)

    top = _read_top_targets(args.top_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_ids = [str(uid) for uid in top["target_id"]]
    for uid in target_ids:
        _remove_outputs(
            args.out_dir / uid / f"{uid}_input.pdb",
            args.out_dir / uid / "pocket_box.json",
        )

    n_written = 0
    try:
        with tempfile.TemporaryDirectory(prefix=".stage4_prepare_", dir=args.out_dir) as tmp:
            stage_root = Path(tmp)
            with tmp_manifest.open("w", newline="") as fh:
                w = csv.writer(fh, delimiter="\t")
                w.writerow([
                    "target_id", "source", "input_pdb", "pocket_box_json",
                    "holo_pdb_id", "holo_deposit_date", "holo_ligands",
                    "holo_resolution", "holo_format",
                    # 어느 체인을 썼는지, 상자를 무엇에서 만들었는지. 이 두 열이
                    # 없으면 복합체를 통째로 넘긴 것과 단량체를 넘긴 것이, 그리고
                    # 좌표계가 맞는 상자와 어긋난 상자가 파일에서 구분되지 않는다.
                    "holo_chain", "holo_chain_coverage", "pocket_box_basis",
                ])
                for uid in target_ids:
                    final_target_dir = args.out_dir / uid
                    stage_target_dir = stage_root / uid
                    stage_target_dir.mkdir(parents=True, exist_ok=True)
                    final_input_pdb = final_target_dir / f"{uid}_input.pdb"
                    stage_input_pdb = stage_target_dir / f"{uid}_input.pdb"
                    rcsb_lookup_failed = False
                    rcsb_download_failed = False
                    holo_format = ""
                    try:
                        holo = fetch_holo_candidates(uid, args.cutoff_date)
                    except requests.RequestException as exc:
                        if not args.allow_rcsb_lookup_failure:
                            raise SystemExit(
                                f"RCSB lookup failed for {uid}; refusing AlphaFold fallback without "
                                "--allow-rcsb-lookup-failure"
                            ) from exc
                        LOG.warning("RCSB lookup failed for %s; falling back to AlphaFold", uid)
                        holo = []
                        rcsb_lookup_failed = True
                    # 후보를 순서대로 시도한다. 받을 수 없는 항목이 하나 있다고
                    # AlphaFold로 떨어지면, 그 아래의 멀쩡한 실험 구조를 버리는 셈이다.
                    cleaned = args.clean_dir / f"{uid}_clean.pdb"
                    if not _nonempty(cleaned):
                        raise SystemExit(
                            f"Missing or empty cleaned AlphaFold structure for {uid}: {cleaned}"
                        )
                    monomer_seq = _monomer_sequence(cleaned)
                    best = None
                    prepared = None
                    for candidate in holo:
                        try:
                            prepared = prepare_holo_structure(
                                candidate.pdb_id, candidate.ligands,
                                monomer_seq, stage_input_pdb,
                            )
                        except HoloDownloadUnavailable as exc:
                            LOG.warning("%s: %s. 다음 후보로 넘어갑니다.", uid, exc)
                            continue
                        best = candidate
                        holo_format = prepared.fmt
                        break

                    if best is not None and prepared is not None:
                        src = "rcsb_holo_post_cutoff"
                        holo_pdb_id = best.pdb_id
                        holo_deposit_date = best.deposit_date
                        holo_ligands = ";".join(best.ligands)
                        holo_resolution = (
                            "" if best.resolution is None else f"{best.resolution:.2f}"
                        )
                        holo_chain = prepared.chain_id
                        holo_chain_coverage = f"{prepared.chain_coverage:.3f}"
                    else:
                        if holo:
                            # 후보가 있었는데 하나도 못 받았다. "없었다"와 다른 일이다.
                            LOG.warning(
                                "%s: 홀로 후보 %d개를 모두 받지 못해 AlphaFold로 갑니다.",
                                uid, len(holo),
                            )
                            rcsb_download_failed = True
                        src = (
                            "alphafold_cleaned_rcsb_lookup_failed"
                            if rcsb_lookup_failed
                            else "alphafold_cleaned_holo_download_failed"
                            if rcsb_download_failed
                            else "alphafold_cleaned"
                        )
                        _copy_atomic(cleaned, stage_input_pdb)
                        holo_pdb_id = ""
                        holo_deposit_date = ""
                        holo_ligands = ""
                        holo_resolution = ""
                        holo_chain = ""
                        holo_chain_coverage = ""

                    final_pocket_box = final_target_dir / "pocket_box.json"
                    stage_pocket_box = stage_target_dir / "pocket_box.json"
                    if prepared is not None and prepared.pocket_box is not None:
                        # 실험 구조에는 실험 구조의 좌표계로 만든 상자를 쓴다.
                        # P2Rank 포켓은 AlphaFold 모델 좌표계에 있어 여기 쓸 수 없다.
                        _write_text_atomic(
                            stage_pocket_box,
                            json.dumps(dict(prepared.pocket_box, uniprot=uid),
                                       ensure_ascii=False, indent=2) + "\n",
                        )
                        pocket_box_basis = "holo_bound_ligand"
                    else:
                        src_pocket = args.pocket_dir / f"{uid}.pockets.json"
                        if _nonempty(src_pocket):
                            _validate_pocket_manifest(src_pocket, uid)
                            _copy_atomic(src_pocket, stage_pocket_box)
                            pocket_box_basis = "p2rank_alphafold"
                        else:
                            raise SystemExit(
                                f"Missing or empty pocket manifest for {uid}: {src_pocket}"
                            )

                    w.writerow([
                        uid, src, str(final_input_pdb), str(final_pocket_box),
                        holo_pdb_id, holo_deposit_date, holo_ligands, holo_resolution,
                        holo_format, holo_chain, holo_chain_coverage, pocket_box_basis,
                    ])
                    n_written += 1
            if n_written == 0:
                raise SystemExit("Stage 4 wrote no prepared structures")
            for uid in target_ids:
                final_target_dir = args.out_dir / uid
                final_target_dir.mkdir(parents=True, exist_ok=True)
                for staged in sorted((stage_root / uid).iterdir()):
                    staged.replace(final_target_dir / staged.name)
            tmp_manifest.replace(args.out_manifest)
    except BaseException:
        _remove_outputs(tmp_manifest)
        raise
    LOG.info("Wrote manifest → %s", args.out_manifest)


if __name__ == "__main__":
    main()
