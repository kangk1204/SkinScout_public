"""3D 비교 예산: conformer 초과 뒤에는 MCS를 시작하지 않고, timeout은 부적합과 다르다.

deadline은 컨포머 생성 전에만 확인돼서, 예산을 넘긴 뒤에도 MCS와 정렬이 돌았고
그 결과가 "공통 구조 없음"과 같은 unavailable로 나왔다. MCS에도 남은 예산이
timeout으로 전달돼야 한다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import discover_substitutes as discovery  # noqa: E402


def _freeze_clock(monkeypatch) -> dict[str, float]:
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    return clock


def test_conformer_overrun_exhausts_budget_and_never_calls_mcs(monkeypatch) -> None:
    """fake conformer가 예산을 넘기면 MCS는 호출되지 않고 score는 null이다."""
    parent = Chem.MolFromSmiles("CCO")
    candidate = Chem.MolFromSmiles("CCN")
    clock = _freeze_clock(monkeypatch)

    def overrunning_ensemble(*_args, **_kwargs):
        clock["now"] = 2.0
        return candidate, [0], "fixture_candidate"

    def forbidden_mcs(*_args, **_kwargs):
        raise AssertionError("budget exhausted before MCS; FindMCS must not run")

    monkeypatch.setattr(discovery, "_conformer_ensemble", overrunning_ensemble)
    monkeypatch.setattr(discovery.rdFMCS, "FindMCS", forbidden_mcs)

    result = discovery._pharmacophore_3d_comparison(
        parent,
        candidate,
        factory=None,
        parent_ensemble=(parent, [0], "fixture_parent"),
        seed=7,
        max_conformers=1,
        deadline=1.0,
    )

    assert result["status"] == "budget_exhausted"
    assert result["feature_family_recall"] is None
    assert result["feature_distance_rmsd"] is None
    assert result["evaluated_pairs"] == 0


def test_mcs_receives_remaining_budget_and_cancel_is_not_unavailable(
    monkeypatch,
) -> None:
    """MCS timeout은 남은 예산으로 줄고, 취소는 부적합이 아니라 예산 소진이다."""
    parent = Chem.MolFromSmiles("CCO")
    candidate = Chem.MolFromSmiles("CCN")
    clock = _freeze_clock(monkeypatch)
    monkeypatch.setattr(
        discovery,
        "_conformer_ensemble",
        lambda *_args, **_kwargs: (candidate, [0], "fixture_candidate"),
    )
    seen: dict[str, object] = {}

    class _Canceled:
        canceled = True
        smartsString = ""

    def canceling_mcs(_mols, **kwargs):
        seen.update(kwargs)
        clock["now"] = 10.0
        return _Canceled()

    monkeypatch.setattr(discovery.rdFMCS, "FindMCS", canceling_mcs)

    result = discovery._pharmacophore_3d_comparison(
        parent,
        candidate,
        factory=None,
        parent_ensemble=(parent, [0], "fixture_parent"),
        seed=7,
        max_conformers=1,
        deadline=3.0,
    )

    assert seen["timeout"] == 3
    assert result["status"] == "budget_exhausted"
    assert result["feature_family_recall"] is None
    assert result["feature_distance_rmsd"] is None


def test_completed_mcs_without_common_core_is_unavailable(monkeypatch) -> None:
    """취소가 아닌 빈 MCS는 여전히 구조 부적합(unavailable)이다."""
    parent = Chem.MolFromSmiles("CCO")
    candidate = Chem.MolFromSmiles("CCO")
    monkeypatch.setattr(
        discovery,
        "_conformer_ensemble",
        lambda *_args, **_kwargs: (candidate, [0], "fixture_candidate"),
    )

    class _Empty:
        canceled = False
        smartsString = ""

    monkeypatch.setattr(discovery.rdFMCS, "FindMCS", lambda *_args, **_kwargs: _Empty())

    result = discovery._pharmacophore_3d_comparison(
        parent,
        candidate,
        factory=None,
        parent_ensemble=(parent, [0], "fixture_parent"),
        seed=7,
        max_conformers=1,
        deadline=None,
    )

    assert result["status"] == "unavailable"
    assert result["feature_family_recall"] is None
