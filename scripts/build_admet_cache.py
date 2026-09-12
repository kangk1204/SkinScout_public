#!/usr/bin/env python3
"""build_admet_cache.py — CosIng 원료 전체의 ADMET 예측을 미리 계산해 둔다.

유사체 검색 결과에 ADMET 열을 붙이려면 후보마다 예측이 필요한데, 요청마다
계산하면 화면이 기다린다. 라이브러리는 자주 바뀌지 않으므로 한 번 계산해 두고
InChIKey 로 조회한다.

실측(ADMET-AI 1.4.0, CPU): 200종 배치에 2.6초, 분자당 13.1 ms. 7,484종 전체가
약 100초다. 한 개씩 부르면 매번 모델을 다시 적재해 훨씬 느리므로 반드시
배치로 넘긴다.

ADMET-AI 는 MIT 라이선스다(설치본 메타데이터 `License-Expression: MIT` 확인).
이 파이프라인은 상업 이용이 가능한 도구만 쓴다 - LICENSE_POLICY.md 참고.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("admet.cache")

# 화면에 띄울 항목. ADMET-AI 는 98개를 내주지만, 화장품 원료 판단에 쓰이는 것만
# 고른다. 나머지는 원본 예측에 그대로 있으므로 필요하면 늘리면 된다.
KEPT_COLUMNS = (
    "molecular_weight", "logP", "tpsa", "hydrogen_bond_donors",
    "hydrogen_bond_acceptors", "QED", "Lipinski",
    "Skin_Reaction", "AMES", "hERG", "DILI", "Carcinogens_Lagunin", "ClinTox",
    "Caco2_Wang", "Solubility_AqSolDB", "Lipophilicity_AstraZeneca",
    "BBB_Martins", "Bioavailability_Ma", "PPBR_AZ", "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ", "CYP3A4_Veith", "CYP2D6_Veith",
)


def load_model():
    try:
        from admet_ai import ADMETModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "admet_ai 를 불러오지 못했습니다. cosmax-dti 환경에서 실행하세요: "
            f"{exc}"
        )
    return ADMETModel()


def predict_in_batches(model, smiles: list[str], batch_size: int) -> pd.DataFrame:
    """배치로 나눠 예측한다. 한 번에 다 넣으면 메모리가 튀고, 하나씩 넣으면 느리다.

    ADMETModel.predict 는 **입력 SMILES 를 인덱스로** 하는 표를 돌려준다(실측:
    4종을 넣으면 index 가 그 4개 SMILES 이고 순서도 같다). 그 인덱스를 버리면
    남는 검사는 행 수뿐인데, 행 수는 순서가 밀려도 그대로다 - 레티놀의 발암성
    예측이 다른 원료에 붙어도 통과한다. 그래서 인덱스를 살려 배치마다 입력과
    같은지 확인한다. 배치 안에서 순서가 바뀌면 여기서 멈춘다.
    """
    frames: list[pd.DataFrame] = []
    started = time.monotonic()
    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start:start + batch_size]
        out = model.predict(smiles=chunk)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame([out], index=list(chunk[:1]))
        if list(out.index.astype(str)) != list(chunk):
            raise SystemExit(
                f"ADMET 예측이 입력과 다른 순서로 돌아왔습니다(배치 {start}). "
                "그대로 붙이면 다른 분자의 값이 붙으므로 저장하지 않습니다."
            )
        frames.append(out)
        done = min(start + batch_size, len(smiles))
        elapsed = time.monotonic() - started
        LOG.info("ADMET %d/%d · 경과 %.0f초 · 남은 예상 %.0f초",
                 done, len(smiles), elapsed,
                 elapsed / done * (len(smiles) - done) if done else 0.0)
    if not frames:
        return pd.DataFrame()
    joined = pd.concat(frames)
    joined.index = joined.index.astype(str)
    return joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("data/cosing/cosing.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/cosing/admet_cache.parquet"))
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="시험용. 0이면 전체.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.library.exists():
        raise SystemExit(f"원료 라이브러리가 없습니다: {args.library}")

    # 화면이 쓰는 것과 **같은 키**로 만든다. `cosing.parquet` 의 `inchikey` 열을
    # 그대로 쓰면 안 된다 - `load_ingredient_library()` 는 SMILES 를
    # `_standardize_mol` 로 정규화한 뒤 InChIKey 를 **다시 계산**하고, 그 값이
    # 원본과 다른 원료가 실측 925종(7,484 중 12.4%)이다. 원본 키로 캐시를 만들면
    # 그 925종은 화면에서 영원히 조인되지 않고, ADMET 열이 비어 안전 축이 통째로
    # 중앙값으로 채워진다. 예측이 실패한 것이 아니라 키가 어긋난 것인데 화면에서는
    # 구분되지 않는다.
    #
    # 정규화된 구조로 예측하는 것이 뜻으로도 맞다 - 화면이 보여 주는 구조가 그것이다.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alternative_ingredients import load_ingredient_library

    library = load_ingredient_library(args.library)
    frame = library.frame
    for column in ("canonical_smiles", "inchikey"):
        if column not in frame.columns:
            raise SystemExit(f"라이브러리에 '{column}' 열이 없습니다: {args.library}")
    rows = frame[["inchikey", "canonical_smiles"]].rename(
        columns={"canonical_smiles": "smiles"}
    ).dropna()
    rows = rows[rows["smiles"].astype(str).str.strip().ne("")]
    rows = rows.drop_duplicates("inchikey").reset_index(drop=True)
    if args.limit > 0:
        rows = rows.head(args.limit)
    if rows.empty:
        raise SystemExit("예측할 분자가 없습니다")
    LOG.info("대상 %d종 · 배치 %d", len(rows), args.batch_size)

    model = load_model()
    wanted = rows["smiles"].astype(str).tolist()
    predictions = predict_in_batches(model, wanted, args.batch_size)
    if len(predictions) != len(rows):
        raise SystemExit(
            f"예측 행 수가 입력과 다릅니다: {len(predictions)} != {len(rows)}. "
            "행이 밀리면 다른 분자의 값이 붙으므로 저장하지 않습니다."
        )
    # 행 수가 같아도 순서가 밀렸을 수 있다. 아래 concat 은 자리로 붙이므로,
    # 붙이기 전에 i번째 예측이 정말 i번째 입력의 것인지 확인한다.
    if list(predictions.index) != wanted:
        first = next((i for i, (a, b) in enumerate(zip(predictions.index, wanted)) if a != b), 0)
        raise SystemExit(
            "예측 순서가 입력과 다릅니다. 자리로 붙이면 다른 분자의 값이 붙으므로 "
            f"저장하지 않습니다(첫 어긋남 {first}번째: 예측 {predictions.index[first]!r} "
            f"≠ 입력 {wanted[first]!r})."
        )

    keep = [c for c in KEPT_COLUMNS if c in predictions.columns]
    missing = [c for c in KEPT_COLUMNS if c not in predictions.columns]
    if missing:
        LOG.warning("예측에 없는 항목 %d개는 건너뜁니다: %s", len(missing), missing[:6])
    out = pd.concat(
        [rows[["inchikey"]].reset_index(drop=True), predictions[keep].reset_index(drop=True)],
        axis=1,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(args.out)
    LOG.info("Wrote %s (%d행 · %d항목)", args.out, len(out), len(keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
