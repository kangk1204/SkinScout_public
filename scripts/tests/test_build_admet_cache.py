"""ADMET 캐시가 **그 분자의** 값을 담는지에 대한 계약.

이 캐시는 화면의 안전 축(Skin_Reaction·AMES·hERG·DILI·발암)을 그대로 먹인다.
행이 한 칸 밀리면 레티놀의 발암성 예측이 다른 원료에 붙고, 그 상태로도 표는
멀쩡해 보인다 - 값이 비어 있지도, 범위를 벗어나지도 않기 때문이다. 화면을
봐서는 알 수 없는 종류의 오류라서, 붙이기 전에 막아야 한다.

`ADMETModel.predict` 는 입력 SMILES 를 인덱스로 하는 표를 돌려준다(실측 확인).
여기서 검증하는 것은 그 인덱스를 실제로 대조하는가이지, 예측값 자체가 아니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_admet_cache import predict_in_batches  # noqa: E402


class FakeModel:
    """입력 SMILES 를 인덱스로 돌려주는 모델. 실제 ADMETModel 과 같은 모양이다."""

    def __init__(self, transform=None) -> None:
        self.transform = transform

    def predict(self, smiles):
        order = list(smiles) if self.transform is None else self.transform(list(smiles))
        return pd.DataFrame({"logP": [float(len(s)) for s in order]}, index=order)


def test_predictions_keep_the_smiles_they_were_asked_about():
    smiles = ["CCO", "c1ccccc1", "NC(=O)c1cccnc1", "CC(=O)O"]
    out = predict_in_batches(FakeModel(), smiles, batch_size=2)
    assert list(out.index) == smiles
    # 자리로 붙일 것이므로, i번째 값이 i번째 입력의 것이어야 한다.
    assert list(out["logP"]) == [float(len(s)) for s in smiles]


def test_a_batch_that_comes_back_reordered_stops_the_build():
    """행 수는 같고 순서만 밀린 경우. 길이 검사만으로는 통과한다."""
    smiles = ["CCO", "c1ccccc1", "NC(=O)c1cccnc1", "CC(=O)O"]
    reversed_batch = FakeModel(transform=lambda chunk: list(reversed(chunk)))
    with pytest.raises(SystemExit) as exc:
        predict_in_batches(reversed_batch, smiles, batch_size=2)
    assert "순서" in str(exc.value)


def test_a_substituted_molecule_stops_the_build():
    """한 분자가 다른 것으로 바뀌어 돌아온 경우. 행 수는 그대로다."""
    smiles = ["CCO", "c1ccccc1"]
    swapped = FakeModel(transform=lambda chunk: ["CCC"] + list(chunk[1:]))
    with pytest.raises(SystemExit):
        predict_in_batches(swapped, smiles, batch_size=8)


def test_an_empty_input_is_an_empty_table_not_a_crash():
    assert predict_in_batches(FakeModel(), [], batch_size=4).empty
