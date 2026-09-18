"""An operator may choose among dev-qualified recipes; never around the gate.

The promoted recipe is picked by the highest dev-gate score, which is a
benchmark-wide criterion. This tool is for cosmetics, and the two disagree:
`union_nonneg_consensus` scores lower on dev (0.4376 vs 0.4607) and is better on
the whitening compounds - hydroquinone -> tyrosinase 33 -> 26, EGCG 92 -> 81 -
because it ignores analogues whose only measurements sit under the activity
threshold.

Both pass the same dev gate, so choosing between them is selection, not a
bypass. `union_nonneg_max` did not pass, and must stay unpromotable by this
route.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _setup():
    from eval.activity_retrieval_model import _choose_recipe
    from activity_retrieval_scoring import CANDIDATES

    by_id = {recipe.recipe_id: recipe for recipe in CANDIDATES}
    passing = [
        (0.4607, by_id["union_any_consensus"]),
        (0.4376, by_id["union_nonneg_consensus"]),
    ]
    rows = [
        {"recipe_id": "union_any_consensus", "dev_gate_score": 0.4607, "dev_gate_failures": ""},
        {"recipe_id": "union_nonneg_consensus", "dev_gate_score": 0.4376, "dev_gate_failures": ""},
        {
            "recipe_id": "union_nonneg_max",
            "dev_gate_score": 0.3611,
            "dev_gate_failures": "log_loss_improvement=-0.0018 must be > 1e-12",
        },
    ]
    return _choose_recipe, passing, rows


def test_without_a_preference_the_highest_dev_score_still_wins() -> None:
    choose, passing, rows = _setup()

    assert choose(passing, rows, None).recipe_id == "union_any_consensus"


def test_a_qualified_recipe_can_be_chosen_over_the_dev_winner() -> None:
    choose, passing, rows = _setup()

    assert choose(passing, rows, "union_nonneg_consensus").recipe_id == "union_nonneg_consensus"


def test_a_recipe_that_failed_the_gate_cannot_be_chosen() -> None:
    """The whole point. A preference selects among qualifiers; it does not
    qualify anything."""
    choose, passing, rows = _setup()

    with pytest.raises(SystemExit) as failure:
        choose(passing, rows, "union_nonneg_max")

    message = str(failure.value)
    assert "failed the dev gate" in message
    assert "log_loss_improvement" in message, "say which gate and by how much"
    assert "not a ranking to override" in message


def test_an_unknown_recipe_id_is_refused_too() -> None:
    choose, passing, rows = _setup()

    with pytest.raises(SystemExit, match="not a known candidate"):
        choose(passing, rows, "union_wishful_thinking")


def test_the_shipped_artifact_records_that_a_human_chose_it() -> None:
    """A promotion that departs from the dev winner must say so in the artifact,
    or the next reader assumes the benchmark picked it."""
    import json

    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    recipe_path = ROOT / config["docking"]["daina_recipe_path"]
    if not recipe_path.exists():
        pytest.skip("the configured recipe artifact is not built here")
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))

    assert payload["passes_dev_gate"] is True
    if payload["selected_recipe"]["recipe_id"] != "union_any_consensus":
        assert payload.get("selection_basis") == "operator_choice_among_dev_qualified"
        assert payload.get("selection_rationale"), "an override needs its reason recorded"


def test_the_workflow_records_the_operator_choice_from_config() -> None:
    """선택을 워크플로가 스스로 기록해야 게이트 재생성이 사람 손을 타지 않는다.

    이전에는 운영자 선택이 수동 명령에만 있었고, dev_selection을 다시 돌리면
    dev 점수 1위가 나와 출하 레시피와 갈라졌다(create-operational sha256 불일치).
    """
    import json

    import yaml

    config = yaml.safe_load((ROOT / "workflow" / "config.yaml").read_text(encoding="utf-8"))
    ar = config["evaluation"]["activity_retrieval"]

    assert ar.get("prefer_recipe"), "config must state the operator choice"
    assert ar.get("prefer_recipe_rationale", "").strip(), "an override needs its reason"

    source = (ROOT / "workflow" / "rules" / "activity_retrieval.smk").read_text(encoding="utf-8")
    assert "AR_PREFER_RECIPE_ARG" in source
    assert "AR_SELECTION_RATIONALE_ARG" in source
    assert "--prefer-recipe" in source

    # 출하 레시피와 config의 선택이 같은 id여야 한다.
    recipe_path = ROOT / config["docking"]["daina_recipe_path"]
    if recipe_path.exists():
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        assert payload["selected_recipe"]["recipe_id"] == ar["prefer_recipe"]
