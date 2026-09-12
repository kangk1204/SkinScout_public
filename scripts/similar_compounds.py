#!/usr/bin/env python3
"""Find measured molecules that look like the query, and say what they measured.

Answers "is there already something like my compound, and what is known about
it" - a different question from target ranking, and one the pipeline could not
answer before: similarity was computed only to aggregate it into target scores
and was then discarded.

Sorted by Tanimoto, but never *only* Tanimoto. Every row carries what the
molecule's own measurements said, because the two come apart in practice: EGCG's
nearest MMP2 analogues sit at similarity 0.727 with pActivity 4.06-4.66. A list
that showed the 0.727 and not the 4.06 would read as a strong lead.

Uses the compact index from build_similarity_index.py: 272 MB and about half a
second, against the retrieval index's 3.5 GB and 27 s.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data" / "similarity_index_202609"
SCHEMA_VERSION = "skinscout.similarity-index.v2"

# One byte -> how many bits are set. Faster than unpackbits by ~2x and the
# difference is what makes this usable from a web request.
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)


@dataclass
class SimilarityIndex:
    fingerprints: np.ndarray      # (n, 256) uint8, packed bits
    popcounts: np.ndarray         # (n,) int32
    ligands: pd.DataFrame
    manifest: dict[str, Any]
    # InChIKey -> 행 위치. 1.06M행을 매 요청마다 훑지 않기 위해 처음 필요할 때
    # 한 번만 만든다. 전체 키와 앞 14자(연결성) 두 벌을 함께 만든다.
    exact_positions: dict[str, int] | None = None
    skeleton_positions: dict[str, int] | None = None


def _build_key_maps(index: SimilarityIndex) -> None:
    if index.exact_positions is not None and index.skeleton_positions is not None:
        return
    keys = index.ligands["standard_inchikey"].astype(str)
    skeletons = keys.str.slice(0, 14)
    # 같은 키가 여러 행이면 표적 수가 가장 많은 행을 남긴다.
    order = np.argsort(-index.ligands["target_count"].to_numpy(), kind="stable")
    exact: dict[str, int] = {}
    skeleton: dict[str, int] = {}
    for position in order:
        position = int(position)
        exact.setdefault(keys.iat[position], position)
        skeleton.setdefault(skeletons.iat[position], position)
    # 다 채운 뒤에 한 번에 붙인다. 빈 dict를 먼저 붙이면, 동시에 들어온 다른
    # 요청이 절반만 채워진 표를 보고 "근거 없음"을 돌려준다.
    index.exact_positions = exact
    index.skeleton_positions = skeleton


def evidence_for_inchikeys(index: SimilarityIndex, inchikeys: list[str]) -> dict[str, dict[str, Any]]:
    """다른 라이브러리에서 고른 분자에 측정 근거를 붙인다.

    전체 InChIKey가 정확히 있으면 그 행을 쓴다. 없을 때만 앞 14자(연결성)로
    내려간다. 순서가 이 방향이어야 하는 이유가 있다.

    처음에는 연결성으로만 맞췄다. 등재 구조와 활성 측정 기록은 입체화학과 양성자화
    상태가 서로 다르게 적혀 있는 경우가 흔해서, 전체 키로만 맞추면 붙어야 할 근거가
    붙지 않기 때문이다. 그런데 연결성이 같은 행이 여럿일 때 "표적 수가 가장 많은
    행"을 고르게 되어 있어서, **자기 자신의 측정값이 인덱스에 있는데도 더 잘 측정된
    이성질체의 값을 대신 보여 주는** 일이 생겼다. 3-Methylfentanyl은 자기 행이
    pAct 8.16인데 화면에는 다른 입체이성질체의 10.7이 찍혔다. 인덱스 전체로는
    연결성 하나에 여러 전체 키가 걸리는 경우가 52,845건이고, 그중 4,226건은
    pActivity가 2 로그 이상 벌어진다(최대 7.40).

    내려간 경우에는 `match`가 `connectivity`가 되고 실제로 맞은 키가
    `matched_inchikey`로 따라 나간다 - 화면이 "이 분자의 값"과 "연결성만 같은
    분자의 값"을 구분해 적을 수 있어야 한다.
    """
    _build_key_maps(index)
    found: dict[str, dict[str, Any]] = {}
    for key in inchikeys:
        key = str(key or "")
        if not key:
            continue
        position = index.exact_positions.get(key)
        match = "exact"
        if position is None:
            position = index.skeleton_positions.get(key[:14])
            match = "connectivity"
        if position is None:
            continue
        row = index.ligands.iloc[position]
        activity = row.get("best_pactivity")
        found[key] = {
            "match": match,
            "matched_inchikey": str(row.get("standard_inchikey") or ""),
            "target_count": int(row.get("target_count") or 0),
            "best_pactivity": None if activity is None or activity != activity else round(float(activity), 2),
            "evidence": str(row.get("evidence") or "none"),
            "top_targets": unique_targets(row.get("top_targets")),
            "targets": unique_targets(row.get("all_targets")),
        }
    return found


def unique_targets(value: Any) -> list[str]:
    """`top_targets` 문자열을 순서를 지키며 중복 없이 편다.

    v1 인덱스는 같은 UniProt를 세 번 담을 수 있었다(측정이 여러 건이면 그렇게
    된다). 빌더는 고쳤지만, 읽는 쪽도 막아 둔다.
    """
    seen: list[str] = []
    for target in str(value or "").split(";"):
        target = target.strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def load_index(directory: Path = DEFAULT_INDEX) -> SimilarityIndex:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"similarity index is not built: {directory}. Run "
            "scripts/build_similarity_index.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"similarity index schema must be {SCHEMA_VERSION}: {manifest_path}")

    fingerprints = np.load(directory / "fingerprints.npy")
    ligands = pd.read_parquet(directory / "ligands.parquet")
    if len(fingerprints) != len(ligands):
        raise SystemExit(
            f"similarity index is inconsistent: {len(fingerprints)} fingerprints, "
            f"{len(ligands)} ligand rows"
        )
    return SimilarityIndex(
        fingerprints=fingerprints,
        popcounts=_POPCOUNT[fingerprints].sum(1, dtype=np.int32),
        ligands=ligands,
        manifest=manifest,
    )


def _query_fingerprint(smiles: str) -> np.ndarray:
    """Standardised the same way the index was, so the numbers are comparable."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from rdkit import DataStructs

    from activity_retrieval_scoring import query_features

    fingerprint, _, _, _ = query_features(smiles)
    array = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return np.packbits(array)


def find_similar(
    smiles: str,
    index: SimilarityIndex,
    *,
    limit: int = 25,
    min_similarity: float = 0.35,
    exclude_self: bool = False,
) -> pd.DataFrame:
    """Rank the reference set by Tanimoto to `smiles`.

    `exclude_self` drops exact-fingerprint matches. Leave it off for a
    researcher checking a known ingredient - being told the compound is already
    in the reference set is the answer, not noise - and turn it on when asking
    what *else* looks like it.
    """
    query = _query_fingerprint(smiles)
    query_bits = int(_POPCOUNT[query].sum())

    intersection = _POPCOUNT[np.bitwise_and(index.fingerprints, query)].sum(1, dtype=np.int32)
    union = index.popcounts + query_bits - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)

    keep = similarity >= min_similarity
    if exclude_self:
        keep &= similarity < 1.0
    positions = np.flatnonzero(keep)
    if positions.size == 0:
        return index.ligands.head(0).assign(similarity=pd.Series(dtype=float))

    order = positions[np.argsort(-similarity[positions], kind="stable")][:limit]
    frame = index.ligands.iloc[order].copy()
    frame.insert(0, "similarity", np.round(similarity[order], 4))
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    index = load_index(args.index_dir)
    frame = find_similar(
        args.smiles,
        index,
        limit=args.limit,
        min_similarity=args.min_similarity,
        exclude_self=args.exclude_self,
    )
    if frame.empty:
        print(f"유사도 {args.min_similarity} 이상인 분자가 없습니다.")
        return 0

    if args.out_csv:
        frame.to_csv(args.out_csv, index=False)
        print(f"wrote {args.out_csv}")

    label = {
        "at_or_above_threshold": "활성 문턱 이상",
        "between_thresholds": "문턱 사이",
        "below_threshold": "문턱 아래",
        "none": "측정값 없음",
    }
    print(f"{'순위':>4} {'유사도':>6} {'표적수':>5} {'최대 pAct':>9}  {'근거':10} InChIKey")
    for row in frame.itertuples(index=False):
        activity = "-" if pd.isna(row.best_pactivity) else f"{row.best_pactivity:.2f}"
        print(
            f"{row.rank:>4} {row.similarity:>6.3f} {row.target_count:>5} {activity:>9}  "
            f"{label.get(row.evidence, row.evidence):10} {row.standard_inchikey}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
