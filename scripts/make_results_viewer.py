#!/usr/bin/env python3
"""Build an offline results viewer for a docking run.

A wet-lab reader needs three things from a run and should not have to open a
terminal for any of them: which targets came out on top, what each score
actually rests on, and what the docked pose looks like. This writes a directory
of static HTML that answers those from any browser, with no server and no
network - Mol* is bundled alongside the structures it renders.

    python scripts/make_results_viewer.py --run-dir results/runs/<id> --out-dir <id>_viewer

Open <out-dir>/index.html.

Structures are embedded in the page rather than fetched. A browser opened on a
file:// URL refuses cross-origin fetches, so the earlier loadStructureFromUrl
wiring produced a blank canvas on exactly the double-click path the README
documents.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from target_failure_modes import failure_modes_for, load_curated  # noqa: E402

MOLSTAR_DIR = ROOT / "scripts" / "report_assets" / "molstar"


def _script_safe_json(value: Any) -> str:
    """Serialize JSON without allowing data to terminate its script element."""

    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("&", r"\u0026")
    )


@dataclass(frozen=True)
class Scorer:
    """One scoring method and everything needed to render it honestly.

    The direction hint used to live at tuple index 3 while the renderer read
    index 2, so every target page printed a raw column name or the literal
    "None" where the units belonged. Naming the fields removes that class of
    bug.
    """

    key: str
    label: str
    files: tuple[str, ...]
    column: str
    inline: str | None
    direction: str
    help_html: str


SCORERS: tuple[Scorer, ...] = (
    Scorer(
        key="autodock",
        label="AutoDock ΔG",
        files=("autodock_all_targets.tsv", "autodock_top5k.tsv"),
        column="vina_score",
        inline="autodock_energy_kcal_mol",
        direction="낮을수록 좋음 (kcal/mol)",
        help_html=(
            "<dt>AutoDock ΔG</dt><dd>도킹 결합 에너지(kcal/mol), <strong>낮을수록</strong> "
            "좋습니다. 크기가 큰 분자가 자동으로 낮게 나오므로 화합물끼리 비교하면 안 됩니다.</dd>"
        ),
    ),
    Scorer(
        key="gnina",
        label="GNINA CNN",
        files=("gnina_rescores.tsv", "gnina_pose_rescores.tsv"),
        column="cnn_affinity",
        inline="gnina_cnn_affinity",
        direction="높을수록 좋음",
        help_html=(
            "<dt>GNINA CNN</dt><dd>AutoDock이 만든 <strong>같은 pose</strong>를 다시 채점한 "
            "값으로, 높을수록 좋습니다.</dd>"
        ),
    ),
    Scorer(
        key="rtm",
        label="RTMScore",
        files=("rtmscore_rescores.tsv",),
        column="rtm_score",
        inline=None,
        direction="높을수록 좋음",
        help_html=(
            "<dt>RTMScore</dt><dd>같은 pose를 다른 방법으로 채점한 값으로, 높을수록 좋습니다.</dd>"
        ),
    ),
    Scorer(
        key="boltz",
        label="Boltz-2",
        files=("boltz2_affinity_top.tsv",),
        column="boltz2_neg_log_uM",
        inline=None,
        direction="높을수록 좋음",
        help_html="<dt>Boltz-2</dt><dd>pose를 보지 않는 독립 예측기입니다.</dd>",
    ),
    Scorer(
        key="diffdock",
        label="DiffDock 신뢰도",
        files=("diffdock_blind_full.tsv",),
        column="diffdock_confidence",
        inline=None,
        direction="높을수록 좋음",
        help_html=(
            "<dt>DiffDock 신뢰도</dt><dd>포켓을 지정하지 않고 결합 위치를 직접 예측한 값입니다. "
            "이 경로로만 후보에 오른 표적이 있으므로 진입 경로 열을 함께 보세요.</dd>"
        ),
    ),
)
SCORER_BY_KEY = {scorer.key: scorer for scorer in SCORERS}

# Columns carrying the provenance of a re-scored pose. They exist to be shown:
# a false `scored_actual_docked_pose` silently invalidates the score beside it.
POSE_PROVENANCE_FILES = (
    ("gnina", "gnina_rescores.tsv"),
    ("gnina", "gnina_pose_rescores.tsv"),
    ("rtm", "rtmscore_rescores.tsv"),
)

# A UniProt accession is [A-Z0-9] with '-' for isoforms. A dot never carries
# meaning here but does carry traversal risk, so it is not in the allowed set.
_SLUG_ALLOWED = re.compile(r"[^A-Za-z0-9_-]")


def _slug(target_id: str) -> str:
    """A filesystem-safe stem for a target ID.

    Target IDs arrive from a CSV, so a malformed or hostile row could otherwise
    write outside the output directory - `../../escaped` produced a file one
    level up from the requested folder.
    """
    cleaned = _SLUG_ALLOWED.sub("_", target_id.strip())
    return cleaned or "unknown_target"


def _contained(parent: Path, child: Path) -> Path:
    """Return `child` only if it really resolves inside `parent`."""
    root = parent.resolve()
    resolved = (root / child).resolve() if not child.is_absolute() else child.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"경로가 출력 디렉터리를 벗어납니다: {child}")
    return resolved


def _read_table(path: Path, *, sep: str = "\t") -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, sep=sep)
    except Exception as error:  # pragma: no cover - surfaced, never swallowed
        print(f"경고: {path.name}을 읽지 못했습니다 ({error})", file=sys.stderr)
        return None


def _target_labels(metadata: Path) -> dict[str, dict[str, str]]:
    """Gene symbol, protein name and molecular function per UniProt accession."""
    if not metadata.exists():
        return {}
    labels: dict[str, dict[str, str]] = {}
    with metadata.open(encoding="utf-8", newline="", errors="replace") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession = (row.get("Uniprot") or "").strip()
            if not accession:
                continue
            for part in (item.strip() for item in accession.split(",")):
                if part and part not in labels:
                    labels[part] = {
                        "gene": (row.get("Gene") or "").strip(),
                        "protein": (row.get("Gene description") or "").strip(),
                        "function": (row.get("Molecular function") or "").strip(),
                    }
    return labels


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def _stage_dir(run_dir: Path) -> Path:
    """Pick the mode whose consensus table actually exists.

    A bare `mode_comprehensive/` holding a stray ligand file used to win over a
    populated `mode_fast/` purely because the directory existed, which left 27
    committed runs with no viewer at all.
    """
    candidates = [
        run_dir / "03_targets" / "mode_comprehensive",
        run_dir / "03_targets" / "mode_fast",
    ]
    for stage in candidates:
        if _consensus_path(stage) is not None:
            return stage
    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"채점 결과 CSV를 찾지 못했습니다. 확인한 경로: {searched}")


def _consensus_path(stage: Path) -> Path | None:
    if not stage.is_dir():
        return None
    # The band-reranked table first: it is the order the measurement supports
    # (docs/RERANK_EXPERIMENT_20260828.md) and it carries daina_rank, so the
    # similarity position stays visible for every row.
    for name in ("top50_band_reranked.csv", "top50_4way_consensus.csv", "top50.csv"):
        candidate = stage / name
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _map_coverage(stage: Path) -> dict[str, Any] | None:
    """Grid coverage behind every AutoDock number in this run."""
    frame = _read_table(stage / "autodock_all_targets.tsv")
    if frame is None or "map_coverage_complete" not in frame.columns:
        return None
    complete = frame["map_coverage_complete"].astype(str).str.lower().isin({"true", "1"})
    numerator = _number(frame["map_coverage_numerator"].iloc[0]) if "map_coverage_numerator" in frame else None
    denominator = _number(frame["map_coverage_denominator"].iloc[0]) if "map_coverage_denominator" in frame else None
    return {
        "complete": bool(complete.all()),
        "numerator": numerator,
        "denominator": denominator,
        "rows": int(len(frame)),
    }


def _entry_routes(stage: Path) -> dict[str, str]:
    """Which pipeline branch put each target into the re-scoring shortlist."""
    frame = _read_table(stage / "top_pct_pre_rescore.csv", sep=",")
    if frame is None or "source" not in frame.columns:
        return {}
    return {
        str(row["target_id"]): str(row["source"])
        for row in frame.to_dict("records")
        if row.get("target_id") is not None
    }


def _pose_provenance(stage: Path) -> dict[str, dict[str, Any]]:
    """`scored_actual_docked_pose` / `gpu_enabled` per target, per scorer."""
    provenance: dict[str, dict[str, Any]] = {}
    for key, name in POSE_PROVENANCE_FILES:
        frame = _read_table(stage / name)
        if frame is None or "target_id" not in frame.columns:
            continue
        for record in frame.to_dict("records"):
            uid = str(record["target_id"])
            entry = provenance.setdefault(uid, {})
            if "scored_actual_docked_pose" in record:
                flag = str(record["scored_actual_docked_pose"]).strip().lower()
                entry[f"{key}_scored_docked_pose"] = flag in {"true", "1"}
            if "gpu_enabled" in record:
                flag = str(record["gpu_enabled"]).strip().lower()
                entry[f"{key}_gpu"] = flag in {"true", "1"}
    return provenance


def _collect(run_dir: Path, metadata: Path, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = _stage_dir(run_dir)
    consensus_path = _consensus_path(stage)
    assert consensus_path is not None  # _stage_dir guarantees this
    consensus = pd.read_csv(consensus_path).head(top_n)

    scores: dict[str, tuple[pd.DataFrame, str]] = {}
    for scorer in SCORERS:
        for name in scorer.files:
            frame = _read_table(stage / name)
            if frame is not None and scorer.column in frame.columns:
                scores[scorer.key] = (frame.set_index("target_id"), scorer.column)
                break

    labels = _target_labels(metadata)
    curated = load_curated()
    routes = _entry_routes(stage)
    provenance = _pose_provenance(stage)
    pose_dir = next(
        (stage / name for name in ("autodock_all_target_poses", "poses")
         if (stage / name).is_dir()),
        None,
    )

    rows: list[dict[str, Any]] = []
    for rank, record in enumerate(consensus.to_dict("records"), start=1):
        uid = str(record["target_id"])
        label = labels.get(uid, {})
        row: dict[str, Any] = {
            "rank": rank,
            "target_id": uid,
            "slug": _slug(uid),
            "gene": label.get("gene") or uid,
            "protein": label.get("protein") or "",
            "consensus": record.get("rrf_score"),
            "source_count": record.get("source_count"),
            "sources": str(record.get("sources", "")),
            "route": routes.get(uid),
            "warnings": failure_modes_for(
                uid, molecular_function=label.get("function", ""), curated=curated
            ),
        }
        row.update(provenance.get(uid, {}))
        method_ranks: dict[str, float] = {}
        for scorer in SCORERS:
            value = None
            if scorer.key in scores:
                frame, column = scores[scorer.key]
                if uid in frame.index:
                    value = _number(frame.loc[uid, column])
            if value is None and scorer.inline and scorer.inline in record:
                value = _number(record.get(scorer.inline))
            row[scorer.key] = value
            method_rank = _number(record.get(f"{scorer.key}_rank"))
            row[f"{scorer.key}_rank"] = method_rank
            if method_rank is not None:
                method_ranks[scorer.key] = method_rank
        # Four methods producing a number is not four methods agreeing. The
        # spread is the honest summary and it is large: across the committed
        # proteome run the median top-50 spread is 209 positions.
        row["rank_spread"] = (
            max(method_ranks.values()) - min(method_ranks.values())
            if len(method_ranks) > 1 else None
        )
        # The fast path publishes its retrieval evidence inline, and telling a
        # looked-up interaction from a predicted one matters more to a reader
        # than any score on the row.
        self_match = record.get("daina_is_self_match")
        if self_match is not None and not pd.isna(self_match):
            row["evidence"] = (
                "조회"
                if str(self_match).strip().lower() in {"true", "1"}
                else "예측"
            )
            row["nearest"] = record.get("daina_max_tanimoto")
            row["reference"] = record.get("daina_supporting_molecule_id")
        for extra in ("ranking_basis", "structural_status", "skin_effect_direction",
                      "daina_known_ligand_count", "cell_type_preferred", "skin_tier",
                      "skin_score", "daina_rank", "rerank_band"):
            value = record.get(extra)
            if value is not None and not pd.isna(value):
                row[extra] = value
        pose = pose_dir / f"{uid}.sdf" if pose_dir is not None else None
        row["pose"] = pose if pose is not None and pose.exists() else None
        rows.append(row)

    context = {
        "stage": stage.name,
        "stage_path": stage,
        "map_coverage": _map_coverage(stage),
        "routes_present": bool(routes),
        "route_counts": {},
    }
    if routes:
        counts: dict[str, int] = {}
        for value in routes.values():
            counts[value] = counts.get(value, 0) + 1
        context["route_counts"] = counts
    return rows, context


WARNING_LABELS = {
    "missing_metal_cofactor": "금속 보조인자 누락",
    "gpcr_inactive_state": "GPCR 비활성 상태",
}

ROUTE_LABELS = {
    "autodock": "AutoDock 격자",
    "diffdock_blind": "DiffDock 블라인드",
}

PAGE_CSS = """
:root{--ink:#16202b;--muted:#5d6b7a;--line:#dde3ea;--bg:#f7f9fb;--accent:#1f6f6b;
      --warn-bg:#fff6e5;--warn-line:#e0b062;--flag-bg:#fdecec;--flag-line:#d98b8b;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 system-ui,-apple-system,"Noto Sans KR",sans-serif;
     color:var(--ink);background:var(--bg)}
main{max-width:1240px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:1.6rem;margin:0 0 4px}
h2{font-size:1.1rem;margin:32px 0 10px}
.sub{color:var(--muted);margin:0 0 22px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:0 0 20px}
.howto{background:#fff;border-left:4px solid var(--accent)}
.howto dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:10px 0 0}
.howto dt{font-weight:600}
.howto dd{margin:0;color:var(--muted)}
.banner{background:var(--warn-bg);border:1px solid var(--warn-line);border-radius:8px;
        padding:12px 16px;margin:0 0 20px}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:#fff;font-size:14px}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:#eef2f6;font-weight:600;position:sticky;top:0;cursor:pointer;user-select:none}
th[data-sort]:after{content:" ⇅";color:#9aa7b4;font-size:11px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#f4f8f8}
a{color:var(--accent)}
.warn{display:inline-block;background:var(--warn-bg);border:1px solid var(--warn-line);
      border-radius:999px;padding:1px 9px;font-size:12px;margin:1px 3px 1px 0;white-space:nowrap}
.flag{display:inline-block;background:var(--flag-bg);border:1px solid var(--flag-line);
      border-radius:999px;padding:1px 9px;font-size:12px;margin:1px 3px 1px 0;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.metric{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.metric span{display:block;color:var(--muted);font-size:12px}
.metric strong{font-size:1.25rem;font-variant-numeric:tabular-nums}
#viewer{height:560px;border:1px solid var(--line);border-radius:8px;background:#101418}
.controls{margin:12px 0}
.controls button{font:inherit;padding:7px 14px;margin-right:8px;border:1px solid var(--line);
                 background:#fff;border-radius:6px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
.controls button[disabled]{opacity:.45;cursor:not-allowed}
.note{color:var(--muted);font-size:13px}
.fallback{background:var(--warn-bg);border:1px solid var(--warn-line);border-radius:8px;
          padding:12px 14px;margin-top:12px}
.chart strong{display:block;margin-bottom:8px}
.chart-art{overflow-x:auto}
.chart-art svg{max-width:100%;height:auto;display:block}
"""

TARGET_JS = """
(function () {
  const node = document.getElementById("structure-data");
  const data = JSON.parse(node.textContent);
  const fallback = document.getElementById("fallback");
  let viewer = null;
  function fail(error) {
    fallback.hidden = false;
    fallback.textContent = "3D 뷰어를 열 수 없습니다: " + error +
      " \u2014 구조 파일은 위 링크에서 내려받을 수 있습니다.";
  }
  async function clearViewer() {
    // Mol*'s Viewer exposes clear() on its plugin, not on itself; calling
    // viewer.clear() threw on every page and left an empty canvas.
    if (!viewer || !viewer.plugin || typeof viewer.plugin.clear !== "function") {
      throw new Error("Mol* clear API\uB97C \uCC3E\uC744 \uC218 \uC5C6\uC2B5\uB2C8\uB2E4");
    }
    await viewer.plugin.clear();
  }
  async function show(mode) {
    await clearViewer();
    // Structures are embedded, not fetched: a file:// page cannot fetch its
    // own siblings, which is exactly how the README tells a reader to open this.
    if (mode === "complex" && data.receptor) {
      await viewer.loadStructureFromData(data.receptor, "pdb", false);
    }
    if (data.pose) {
      await viewer.loadStructureFromData(data.pose, "sdf", false);
    }
  }
  async function boot() {
    if (!window.molstar || !window.molstar.Viewer) throw new Error("Mol* \uBC88\uB4E4 \uC5C6\uC74C");
    viewer = await window.molstar.Viewer.create("viewer", {
      layoutIsExpanded: false, layoutShowControls: false, layoutShowSequence: false,
      layoutShowLog: false, layoutShowLeftPanel: false, viewportShowExpand: true,
    });
    // Exposed so an end-to-end browser check can assert what actually loaded
    // rather than trusting a screenshot's pixels.
    window.__viewerPlugin = viewer.plugin;
    await show(data.receptor ? "complex" : "ligand");
    const complexButton = document.getElementById("btn-complex");
    const ligandButton = document.getElementById("btn-ligand");
    complexButton.addEventListener("click", () => {
      complexButton.setAttribute("aria-pressed", "true");
      ligandButton.setAttribute("aria-pressed", "false");
      show("complex").catch(fail);
    });
    ligandButton.addEventListener("click", () => {
      ligandButton.setAttribute("aria-pressed", "true");
      complexButton.setAttribute("aria-pressed", "false");
      show("ligand").catch(fail);
    });
  }
  boot().catch((error) => fail(error && error.message ? error.message : String(error)));
})();
"""

SORT_JS = """
document.querySelectorAll('th[data-sort]').forEach((header, column) => {
  header.addEventListener('click', () => {
    const table = header.closest('table');
    const body = table.tBodies[0];
    const numeric = header.classList.contains('num');
    const descending = header.dataset.dir !== 'desc';
    header.dataset.dir = descending ? 'desc' : 'asc';
    const rows = Array.from(body.rows);
    rows.sort((a, b) => {
      const x = a.cells[column].dataset.value ?? a.cells[column].textContent.trim();
      const y = b.cells[column].dataset.value ?? b.cells[column].textContent.trim();
      if (numeric) {
        const nx = parseFloat(x), ny = parseFloat(y);
        const missingX = Number.isNaN(nx), missingY = Number.isNaN(ny);
        // Missing values sort last in both directions; treating them as
        // Infinity used to float empty rows to the top of a descending sort.
        if (missingX && missingY) return 0;
        if (missingX) return 1;
        if (missingY) return -1;
        return descending ? ny - nx : nx - ny;
      }
      return descending ? y.localeCompare(x) : x.localeCompare(y);
    });
    rows.forEach((row) => body.appendChild(row));
  });
});
"""


# Charts are drawn server-side as inline SVG. A chart library would add several
# megabytes of JavaScript to a folder meant to be zipped and emailed, and the
# two things a reader needs to see here - how far the methods disagree, and
# where the shown targets sit in the whole distribution - are static pictures.
CHART_INK = "#16202b"
CHART_MUTED = "#8a97a5"
CHART_ACCENT = "#1f6f6b"
CHART_WARN = "#c2643c"


# Korean labels have to survive into the SVG. matplotlib draws text as glyph
# paths by default, and its bundled DejaVu font has no Hangul, so every Korean
# label came out empty. Writing real <text> elements hands rendering to the
# browser, which already has the page's Korean font stack.
CHART_FONT_STACK = 'system-ui, -apple-system, "Noto Sans KR", "Noto Sans CJK KR", sans-serif'
_CHART_STYLE_APPLIED = False


def _prepare_chart_style() -> None:
    global _CHART_STYLE_APPLIED
    if _CHART_STYLE_APPLIED:
        return
    import matplotlib

    matplotlib.rcParams["svg.fonttype"] = "none"
    # Only affects layout metrics; the browser picks the face it draws with.
    for candidate in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "DejaVu Sans"):
        try:
            from matplotlib import font_manager

            font_manager.findfont(candidate, fallback_to_default=False)
        except Exception:  # noqa: BLE001
            continue
        matplotlib.rcParams["font.family"] = [candidate]
        break
    _CHART_STYLE_APPLIED = True


def _svg_figure(figure) -> str:
    """Render a matplotlib figure to inline SVG with no XML preamble."""
    import io

    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", transparent=True)
    markup = buffer.getvalue()
    start = markup.find("<svg")
    markup = markup[start:] if start >= 0 else ""
    # Give the browser the same font stack the page uses.
    return markup.replace(
        "<svg ", f'<svg style="font-family:{CHART_FONT_STACK}" ', 1
    )


def _rank_spread_chart(rows: list[dict[str, Any]], present: list[str]) -> str:
    """One line per target from its best to its worst rank across methods.

    "Four methods scored this" and "four methods agreed" are different claims,
    and the table's numbers alone did not make the difference visible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _prepare_chart_style()
    except ImportError:  # pragma: no cover - matplotlib ships with the runtime
        return ""
    entries = []
    for row in rows:
        ranks = [row.get(f"{key}_rank") for key in present]
        ranks = [value for value in ranks if value is not None]
        if len(ranks) > 1:
            entries.append((row["rank"], row["gene"], min(ranks), max(ranks)))
    if len(entries) < 3:
        return ""

    height = max(2.6, 0.19 * len(entries))
    figure, axes = plt.subplots(figsize=(8.4, height))
    for position, (_, _, low, high) in enumerate(entries):
        axes.plot([low, high], [position, position], color=CHART_MUTED, linewidth=1.4,
                  solid_capstyle="round", zorder=1)
        axes.scatter([low], [position], s=16, color=CHART_ACCENT, zorder=2)
        axes.scatter([high], [position], s=16, color=CHART_WARN, zorder=2)
    axes.set_yticks(range(len(entries)))
    axes.set_yticklabels(
        [f"{rank}. {gene}" for rank, gene, _, _ in entries], fontsize=7, color=CHART_INK
    )
    axes.invert_yaxis()
    axes.set_xlabel("각 방법이 매긴 순위 (왼쪽이 좋음)", fontsize=9, color=CHART_INK)
    axes.tick_params(axis="x", labelsize=8, colors=CHART_INK)
    axes.grid(axis="x", color="#e6ebf0", linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color("#dde3ea")
    markup = _svg_figure(figure)
    plt.close(figure)
    return markup


def _score_distribution_chart(
    stage: Path, rows: list[dict[str, Any]], present: list[str]
) -> str:
    """Where the shown targets sit inside the full screen.

    A top-50 table cannot say whether those 50 stand apart from the other
    thousands or sit in the middle of the pile.
    """
    if "autodock" not in present:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _prepare_chart_style()
    except ImportError:  # pragma: no cover
        return ""
    frame = _read_table(stage / "autodock_all_targets.tsv")
    if frame is None:
        frame = _read_table(stage / "autodock_top5k.tsv")
    if frame is None or "vina_score" not in frame.columns or len(frame) < 50:
        return ""
    values = pd.to_numeric(frame["vina_score"], errors="coerce").dropna()
    shown = [row["autodock"] for row in rows if row.get("autodock") is not None]
    if values.empty:
        return ""

    figure, axes = plt.subplots(figsize=(8.4, 2.9))
    axes.hist(values, bins=60, color="#cfd9e2", edgecolor="none")
    for value in shown:
        axes.axvline(value, color=CHART_ACCENT, linewidth=0.9, alpha=0.75)
    axes.set_xlabel("AutoDock ΔG (kcal/mol, 낮을수록 좋음)", fontsize=9, color=CHART_INK)
    axes.set_ylabel(f"표적 수 (전체 {len(values):,})", fontsize=9, color=CHART_INK)
    axes.tick_params(labelsize=8, colors=CHART_INK)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#dde3ea")
    markup = _svg_figure(figure)
    plt.close(figure)
    return markup


def _similarity_chart(rows: list[dict[str, Any]]) -> str:
    """Nearest measured analog per target - the number that predicts recovery."""
    values = [row.get("nearest") for row in rows]
    values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if len(values) < 5:
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _prepare_chart_style()
    except ImportError:  # pragma: no cover
        return ""
    figure, axes = plt.subplots(figsize=(8.4, 2.4))
    axes.hist(values, bins=20, range=(0, 1), color="#cfd9e2", edgecolor="none")
    axes.axvline(0.6, color=CHART_ACCENT, linewidth=1.2, linestyle="--")
    axes.text(0.605, axes.get_ylim()[1] * 0.92, "0.6", fontsize=8, color=CHART_ACCENT)
    axes.set_xlabel(
        "최근접 측정 유사체와의 Tanimoto (0.6 이상에서 회수가 잘 됨)",
        fontsize=9, color=CHART_INK,
    )
    axes.set_ylabel("표적 수", fontsize=9, color=CHART_INK)
    axes.tick_params(labelsize=8, colors=CHART_INK)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#dde3ea")
    markup = _svg_figure(figure)
    plt.close(figure)
    return markup


def _charts_section(
    rows: list[dict[str, Any]], present: list[str], context: dict[str, Any]
) -> str:
    stage = context.get("stage_path")
    blocks: list[tuple[str, str, str]] = []
    spread = _rank_spread_chart(rows, present)
    if spread:
        blocks.append((
            "방법마다 순위가 얼마나 갈리나",
            spread,
            "왼쪽 점은 그 표적을 가장 높게 본 방법의 순위, 오른쪽 점은 가장 낮게 본 방법의 "
            "순위입니다. 선이 길수록 방법들이 다르게 봤다는 뜻입니다.",
        ))
    if isinstance(stage, Path):
        distribution = _score_distribution_chart(stage, rows, present)
        if distribution:
            blocks.append((
                "이 표적들은 전체에서 어디쯤인가",
                distribution,
                "회색은 채점된 표적 전체의 분포이고, 초록 선은 이 표에 실린 표적입니다. "
                "선이 왼쪽 끝에 몰려 있지 않다면 상위권과 나머지의 차이가 크지 않다는 뜻입니다.",
            ))
    similarity = _similarity_chart(rows)
    if similarity:
        blocks.append((
            "근거가 얼마나 가까운가",
            similarity,
            "검증 실측에서 이 값이 0.6 이상인 화합물은 알려진 표적 대부분을 상위 30위 안에 "
            "회수했고, 그 아래에서는 크게 떨어졌습니다.",
        ))
    if not blocks:
        return ""
    cards = "".join(
        f'<section class="card chart"><strong>{html.escape(title)}</strong>'
        f'<div class="chart-art">{markup}</div>'
        f'<p class="note">{html.escape(caption)}</p></section>'
        for title, markup, caption in blocks
    )
    return f"<h2>그림으로 보기</h2>{cards}"


def _warning_html(warnings: list[dict[str, str]]) -> str:
    if not warnings:
        return '<span class="note">—</span>'
    return "".join(
        f'<span class="warn" title="{html.escape(item.get("detail", ""))}">'
        f'{html.escape(WARNING_LABELS.get(item["failure_mode"], item["failure_mode"]))}</span>'
        for item in warnings
    )


def _page(title: str, body: str, depth: int = 0) -> str:
    up = "../" * depth
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{up}assets/viewer.css">
</head><body><main>{body}</main></body></html>"""


def _coverage_banner(context: dict[str, Any]) -> str:
    coverage = context.get("map_coverage")
    if not coverage or coverage.get("complete"):
        return ""
    numerator = coverage.get("numerator")
    denominator = coverage.get("denominator")
    if numerator and denominator:
        percent = 100.0 * numerator / denominator
        detail = (
            f"수용체 {int(denominator):,}개 중 <strong>{int(numerator):,}개"
            f"({percent:.1f}%)</strong>에만 도킹 격자가 만들어졌습니다."
        )
    else:
        detail = "일부 수용체에만 도킹 격자가 만들어졌습니다."
    return (
        f'<div class="banner"><strong>AutoDock 격자가 불완전합니다.</strong> {detail} '
        "격자가 없는 표적은 낮은 점수가 아니라 <em>점수 없음</em>이며, 이 표에 오르지 못한 "
        "표적 중에 실제 결합자가 있을 수 있습니다.</div>"
    )


def _route_banner(rows: list[dict[str, Any]], context: dict[str, Any]) -> str:
    counts = context.get("route_counts") or {}
    if len(counts) < 2:
        return ""
    total = sum(counts.values())
    parts = ", ".join(
        f"{html.escape(ROUTE_LABELS.get(name, name))} {count:,}개"
        for name, count in sorted(counts.items(), key=lambda item: -item[1])
    )
    shown: dict[str, int] = {}
    for row in rows:
        route = row.get("route")
        if route:
            shown[route] = shown.get(route, 0) + 1
    missing = [
        ROUTE_LABELS.get(name, name) for name in counts if name not in shown
    ]
    tail = ""
    if missing:
        tail = (
            f" 이 표의 상위 {len(rows)}개에는 "
            f"{html.escape(', '.join(missing))} 경로로 들어온 표적이 <strong>하나도 없습니다</strong>."
        )
    return (
        f'<div class="banner"><strong>재채점 후보 {total:,}개가 두 경로로 들어왔습니다.</strong> '
        f"{parts}.{tail} 경로가 다르면 점수의 의미도 다르므로 <em>진입 경로</em> 열을 함께 보세요.</div>"
    )


def _index_html(
    rows: list[dict[str, Any]], run_id: str, present: list[str], context: dict[str, Any]
) -> str:
    columns = "".join(
        f'<th class="num" data-sort>{html.escape(SCORER_BY_KEY[key].label)}</th>'
        for key in present
    )
    has_ranks = any(row.get("rank_spread") is not None for row in rows)
    rank_head = '<th class="num" data-sort>순위 편차</th>' if has_ranks else ""
    moved = [
        row for row in rows
        if row.get("daina_rank") is not None and int(row["daina_rank"]) != row["rank"]
    ]
    similarity_head = (
        '<th class="num" data-sort>유사도 순위</th>' if moved else ""
    )
    has_routes = any(row.get("route") for row in rows)
    route_head = '<th data-sort>진입 경로</th>' if has_routes else ""
    has_evidence = any(row.get("evidence") for row in rows)
    evidence_head = (
        '<th data-sort>근거</th><th class="num" data-sort>최근접 유사도</th>'
        if has_evidence else ""
    )
    # Explain only the scorers this run actually produced; a definition for a
    # column that is not on screen reads as a missing result.
    score_help = "\n    ".join(SCORER_BY_KEY[key].help_html for key in present)
    reranked = any(
        str(row.get("ranking_basis", "")) == "daina_head_then_band_rrf" for row in rows
    )
    rerank_help = (
        "    <dt>유사도 순위</dt><dd>도킹을 보기 전, <strong>유사도만으로</strong> 매긴 순위입니다. "
        "이 표의 순위와 다르면 도킹이 그 표적을 움직였다는 뜻입니다. "
        "실측에 따라 <strong>상위 10위는 손대지 않고 11–50위만</strong> 재정렬합니다 — "
        "유사도가 이미 확신하는 구간에서는 도킹이 8건 중 8건을 나쁘게 만들었고, "
        "애매한 구간에서는 13건 중 11건을 개선했습니다.</dd>"
        if reranked and moved else ""
    )
    spreads = sorted(
        row["rank_spread"] for row in rows if row.get("rank_spread") is not None
    )
    median_spread = spreads[len(spreads) // 2] if spreads else None
    if has_ranks:
        score_help += (
            "\n    <dt>순위 편차</dt><dd>이 표적에 대해 각 방법이 매긴 순위의 "
            "<strong>최댓값 − 최솟값</strong>입니다. 작을수록 방법들이 비슷하게 봤다는 뜻이고, "
            "크면 한 방법만 이 표적을 밀어올렸다는 뜻입니다."
            + (
                f" 이 표의 중앙값은 <strong>{median_spread:,.0f}위</strong>입니다."
                if median_spread is not None else ""
            )
            + "</dd>"
        )
    if has_evidence:
        score_help += (
            "\n    <dt>근거 (조회 / 예측)</dt><dd><strong>조회</strong>는 입력 화합물과 지문이 "
            "같은 참조 리간드가 이 표적에서 이미 측정돼 있다는 뜻입니다. 새로 예측한 것이 "
            "아니라 알려진 상호작용을 찾아온 것이므로 새 정보가 아닙니다. "
            "<strong>예측</strong>은 유사 화합물에서 외삽한 결과입니다.</dd>"
            "\n    <dt>최근접 유사도</dt><dd>이 표적의 측정된 리간드 중 입력과 가장 가까운 것의 "
            "Tanimoto 값입니다. 1.000이면 조회입니다.</dd>"
        )
    body_rows = []
    for row in rows:
        extra_cells = ""
        if moved:
            original = row.get("daina_rank")
            shift = None if original is None else int(original) - row["rank"]
            note = "" if not shift else f" title=\"{'▲' if shift > 0 else '▼'} {abs(shift)}칸\""
            extra_cells += (
                f'<td class="num" data-value="{"" if original is None else int(original)}"{note}>'
                f'{"—" if original is None else int(original)}</td>'
            )
        if has_ranks:
            spread = row.get("rank_spread")
            extra_cells += (
                f'<td class="num" data-value="{"" if spread is None else spread}">'
                f'{_fmt(spread, 0)}</td>'
            )
        if has_routes:
            route = row.get("route")
            extra_cells += (
                f'<td>{html.escape(ROUTE_LABELS.get(route, route)) if route else "—"}</td>'
            )
        if has_evidence:
            kind = row.get("evidence") or "—"
            title = f"참조 분자 {row.get('reference')}" if row.get("reference") else ""
            extra_cells += (
                f'<td title="{html.escape(title)}">{html.escape(kind)}</td>'
                f'<td class="num" data-value="{html.escape(str(row.get("nearest") or ""))}">'
                f'{html.escape(_fmt(row.get("nearest")))}</td>'
            )
        cells = "".join(
            f'<td class="num" data-value="{"" if row[key] is None else row[key]}">'
            f"{html.escape(_fmt(row[key]))}</td>"
            for key in present
        )
        body_rows.append(
            f'<tr><td class="num" data-value="{row["rank"]}">{row["rank"]}</td>'
            f'<td><a href="targets/{html.escape(row["slug"])}.html">'
            f'<strong>{html.escape(row["gene"])}</strong></a><br>'
            f'<span class="note">{html.escape(row["target_id"])}</span></td>'
            f'<td>{html.escape(row["protein"][:70])}</td>'
            f'<td class="num" data-value="{html.escape(str(row["consensus"] or ""))}">{html.escape(_fmt(row["consensus"], 5))}</td>'
            f'<td class="num" data-value="{html.escape(str(row["source_count"] or ""))}">{html.escape(str(row["source_count"] or "—"))}</td>'
            f"{cells}{extra_cells}<td>{_warning_html(row['warnings'])}</td></tr>"
        )
    return _page(
        f"SkinScout 결과 — {run_id}",
        f"""
<h1>표적 예측 결과</h1>
<p class="sub">실행 <code>{html.escape(run_id)}</code> · 상위 {len(rows)}개 ·
   유전자명을 눌러 3D 구조를 봅니다.</p>
{_coverage_banner(context)}{_route_banner(rows, context)}
<section class="card howto">
  <strong>읽는 법</strong>
  <dl>
    <dt>합의 점수</dt><dd>여러 채점 방법의 <em>순위</em>를 합친 값입니다. 결합 세기가 아니라
      이 실행 안에서의 상대 순위이며, 다른 실행과 절대값을 비교하지 마세요.</dd>
{rerank_help}
    <dt>점수를 낸 방법 수</dt><dd>이 표적에 <em>숫자를 만들어낸</em> 방법의 개수입니다.
      <strong>방법들이 동의했다는 뜻이 아닙니다.</strong> 커밋된 전체 프로테옴 실행(220 표적)에서
      서로 상관이 있는 쌍은 GNINA–RTMScore(ρ=+0.45) 하나뿐이었고 나머지는 0 근처였습니다.
      실제로 얼마나 갈리는지는 <em>순위 편차</em> 열과 각 표적 페이지에서 확인하세요.</dd>
{score_help}
    <dt>수용체 한계</dt><dd>이 표시가 있으면 도킹 점수가 낮아도 결합이 약하다는 근거가 되지 않습니다.
      구조 준비가 금속 보조인자를 넣지 못했거나 GPCR을 비활성 상태로 모델링한 경우입니다.</dd>
  </dl>
  <p class="note">이 결과는 <strong>실험 우선순위를 정하기 위한 계산 가설</strong>입니다.
     직접 결합이나 효능을 뜻하지 않습니다. 해석 지침은
     <code>docs/RESEARCHER_GUIDE.md</code>를 보세요.</p>
</section>

<div class="tablewrap"><table><thead><tr>
<th class="num" data-sort>순위</th><th data-sort>유전자 / UniProt</th><th data-sort>단백질</th>
<th class="num" data-sort>합의 점수</th><th class="num" data-sort>점수를 낸 방법 수</th>
{columns}{similarity_head}{rank_head}{route_head}{evidence_head}<th>수용체 한계</th>
</tr></thead><tbody>
{"".join(body_rows)}
</tbody></table></div>
<p class="note">표 머리글을 눌러 정렬할 수 있습니다. 엑셀용 파일: <a href="summary.csv">summary.csv</a></p>
{_charts_section(rows, present, context)}
<script src="assets/sort.js"></script>
""",
    )


def _method_rank_block(row: dict[str, Any], present: list[str]) -> str:
    ranks = [(SCORER_BY_KEY[key].label, row.get(f"{key}_rank")) for key in present]
    ranks = [(label, value) for label, value in ranks if value is not None]
    if len(ranks) < 2:
        return ""
    cells = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span>'
        f"<strong>{_fmt(value, 0)}위</strong></div>"
        for label, value in ranks
    )
    spread = row.get("rank_spread")
    verdict = (
        "방법들이 비슷하게 봤습니다."
        if spread is not None and spread <= 20
        else "방법에 따라 크게 갈립니다. 한 방법이 끌어올린 결과일 수 있습니다."
    )
    return (
        "<h2>방법별 순위</h2>"
        f'<div class="grid">{cells}</div>'
        f'<p class="note">최대 − 최소 = <strong>{_fmt(spread, 0)}위</strong>. {verdict} '
        "합의 점수는 이 순위들을 합친 값이며, 값이 높다고 방법들이 동의했다는 뜻은 아닙니다.</p>"
    )


def _provenance_block(row: dict[str, Any]) -> str:
    items = []
    for key, label in (("gnina", "GNINA"), ("rtm", "RTMScore")):
        flag = row.get(f"{key}_scored_docked_pose")
        if flag is None:
            continue
        if flag:
            items.append(f"<li>{label}는 실제로 도킹된 pose를 채점했습니다.</li>")
        else:
            items.append(
                f'<li><span class="flag">주의</span> {label}는 도킹된 pose가 아닌 '
                "구조를 채점했습니다. 이 값은 도킹 결과의 재채점으로 읽으면 안 됩니다.</li>"
            )
    if row.get("gnina_gpu") is False:
        items.append("<li>GNINA는 CPU에서 실행됐습니다.</li>")
    route = row.get("route")
    if route:
        items.append(
            f"<li>이 표적은 <strong>{html.escape(ROUTE_LABELS.get(route, route))}</strong> "
            "경로로 후보에 올랐습니다.</li>"
        )
    if not items:
        return ""
    return f'<h2>점수의 출처</h2><section class="card"><ul>{"".join(items)}</ul></section>'


def _target_html(row: dict[str, Any], present: list[str], run_id: str) -> str:
    metrics = "".join(
        f'<div class="metric"><span>{html.escape(SCORER_BY_KEY[key].label)}</span>'
        f"<strong>{html.escape(_fmt(row[key]))}</strong>"
        f'<span>{html.escape(SCORER_BY_KEY[key].direction)}</span></div>'
        for key in present
    )
    warnings = row["warnings"]
    warn_block = ""
    if warnings:
        items = "".join(
            f"<li><strong>{html.escape(WARNING_LABELS.get(w['failure_mode'], w['failure_mode']))}</strong>"
            f" — {html.escape(w.get('detail', ''))}</li>"
            for w in warnings
        )
        warn_block = (
            '<section class="card fallback"><strong>이 수용체의 알려진 한계</strong>'
            f"<ul>{items}</ul>"
            "<p class='note'>도킹 점수가 낮아도 결합이 약하다는 근거가 되지 않습니다.</p></section>"
        )
    evidence_block = ""
    if row.get("evidence"):
        detail = (
            "입력 화합물과 지문이 같은 참조 리간드가 이 표적에서 이미 측정돼 있습니다. "
            "새로 예측한 결과가 아닙니다."
            if row["evidence"] == "조회"
            else "가장 가까운 측정 리간드로부터 외삽한 결과입니다."
        )
        extras = [f'최근접 유사도 {html.escape(_fmt(row.get("nearest")))}']
        if row.get("reference"):
            extras.append(f'참조 분자 {html.escape(str(row["reference"]))}')
        if row.get("daina_known_ligand_count") is not None:
            extras.append(
                f'이 표적의 측정된 리간드 {html.escape(_fmt(row["daina_known_ligand_count"], 0))}개'
            )
        if row.get("ranking_basis"):
            extras.append(f'순위 기준 {html.escape(str(row["ranking_basis"]))}')
        evidence_block = (
            '<h2>이 표적이 나온 근거</h2><section class="card">'
            f'<p><strong>{html.escape(row["evidence"])}</strong> — {detail}</p>'
            f'<p class="note">{" · ".join(extras)}</p></section>'
        )
    context_rows = []
    for field, label in (
        ("skin_tier", "피부 발현 등급"),
        ("skin_score", "피부 발현 점수"),
        ("cell_type_preferred", "주요 세포 유형"),
        ("skin_effect_direction", "작용 방향"),
        ("structural_status", "구조 상태"),
    ):
        if row.get(field) is not None:
            context_rows.append(
                f"<dt>{label}</dt><dd>{html.escape(str(row[field]))}</dd>"
            )
    context_block = (
        f'<h2>피부 맥락</h2><section class="card howto"><dl>{"".join(context_rows)}</dl></section>'
        if context_rows else ""
    )

    missing_receptor = bool(row.get("missing_receptor"))
    if missing_receptor:
        pose_note = (
            "이 표적은 준비된 수용체 구조가 없어 3D로 표시할 수 없습니다. "
            "구조가 없다는 것은 결합이 약하다는 뜻이 아니라 계산에 쓸 구조가 없었다는 뜻입니다."
        )
    elif row["pose"] is not None:
        pose_note = "수용체 구조와 도킹된 리간드 pose를 함께 표시합니다."
    else:
        pose_note = "이 표적은 도킹된 pose가 없어 수용체 구조만 표시합니다."

    receptor_text = row.get("receptor_text")
    pose_text = row.get("pose_text")
    download_links = []
    if receptor_text is not None:
        download_links.append(
            f'<a href="../structures/{html.escape(row["slug"])}_receptor.pdb">수용체 PDB</a>'
        )
    if pose_text is not None:
        download_links.append(
            f'<a href="../structures/{html.escape(row["slug"])}_pose.sdf">리간드 pose SDF</a>'
        )
    downloads = (
        f'<p class="note">내려받기: {" · ".join(download_links)}</p>'
        if download_links else ""
    )

    viewer_block = ""
    if receptor_text is not None or pose_text is not None:
        viewer_block = f"""
<div class="controls">
  <button id="btn-complex" aria-pressed="true"{" disabled" if receptor_text is None else ""}>수용체 + 리간드</button>
  <button id="btn-ligand" aria-pressed="false"{" disabled" if pose_text is None else ""}>리간드만</button>
</div>
<div id="viewer"></div>
<div class="fallback" id="fallback" hidden></div>
{downloads}
<link rel="stylesheet" href="../assets/molstar.css">
<script src="../assets/molstar.js"></script>
<script id="structure-data" type="application/json">{_script_safe_json({
    "receptor": receptor_text, "pose": pose_text
})}</script>
<script src="../assets/target.js"></script>
"""
    else:
        viewer_block = (
            '<div class="fallback">표시할 구조 파일이 없습니다. '
            "수용체 구조와 도킹 pose가 모두 준비되지 않았습니다.</div>"
        )

    return _page(
        f"{row['gene']} — {run_id}",
        f"""
<p class="sub"><a href="../index.html">← 전체 목록</a></p>
<h1>{html.escape(row['gene'])} <span class="note">{html.escape(row['target_id'])}</span></h1>
<p class="sub">{html.escape(row['protein'])} · 합의 순위 {row['rank']}위 ·
   점수를 낸 방법 {html.escape(str(row['source_count']))}개 ({html.escape(row['sources'])})</p>
{warn_block}
{evidence_block}
<h2>점수</h2>
<div class="grid">{metrics}</div>
{_method_rank_block(row, present)}
{_provenance_block(row)}
{context_block}
<h2>3D 구조</h2>
<p class="note">{pose_note} 마우스 끌기로 회전, 휠로 확대·축소합니다.</p>
{viewer_block}
""",
        depth=1,
    )


CSV_HEADERS = {
    "rank": "순위",
    "target_id": "UniProt",
    "gene": "유전자",
    "protein": "단백질",
    "consensus_score": "합의 점수",
    "source_count": "점수를 낸 방법 수",
    "sources": "채점 방법",
    "rank_spread": "순위 편차",
    "route": "진입 경로",
    "evidence": "근거",
    "nearest_similarity": "최근접 유사도",
    "reference_molecule": "참조 분자",
    "receptor_limitations": "수용체 한계",
}


def _write_outputs(
    rows: list[dict[str, Any]], out_dir: Path, run_id: str, present: list[str],
    context: dict[str, Any],
) -> None:
    (out_dir / "index.html").write_text(
        _index_html(rows, run_id, present, context), encoding="utf-8"
    )
    targets_dir = out_dir / "targets"
    for row in rows:
        page = _contained(targets_dir, Path(f"{row['slug']}.html"))
        page.write_text(_target_html(row, present, run_id), encoding="utf-8")

    has_ranks = any(row.get("rank_spread") is not None for row in rows)
    has_routes = any(row.get("route") for row in rows)
    has_evidence = any(row.get("evidence") for row in rows)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        columns = [
            "rank", "target_id", "gene", "protein", "consensus_score",
            "source_count", "sources",
        ]
        columns += [SCORER_BY_KEY[key].label for key in present]
        if has_ranks:
            columns += [f"{SCORER_BY_KEY[key].label} 순위" for key in present]
            columns.append("rank_spread")
        if has_routes:
            columns.append("route")
        if has_evidence:
            columns += ["evidence", "nearest_similarity", "reference_molecule"]
        columns.append("receptor_limitations")
        writer.writerow([CSV_HEADERS.get(name, name) for name in columns])
        for row in rows:
            values: list[Any] = [
                row["rank"], row["target_id"], row["gene"], row["protein"],
                row["consensus"], row["source_count"], row["sources"],
            ]
            values += [row.get(key) for key in present]
            if has_ranks:
                values += [row.get(f"{key}_rank") for key in present]
                values.append(row.get("rank_spread"))
            if has_routes:
                values.append(row.get("route"))
            if has_evidence:
                values += [row.get("evidence"), row.get("nearest"), row.get("reference")]
            values.append(";".join(w["failure_mode"] for w in row["warnings"]))
            writer.writerow(values)


def build(run_dir: Path, out_dir: Path, metadata: Path, clean_dir: Path, top_n: int) -> None:
    rows, context = _collect(run_dir, metadata, top_n)
    present = [
        scorer.key for scorer in SCORERS
        if any(row.get(scorer.key) is not None for row in rows)
    ]

    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish atomically from a sibling directory. A half-written viewer that
    # replaced a good one used to read as a current result, and staging under
    # the system temp root cannot be renamed across filesystems.
    staging = out_dir.parent / f".{out_dir.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if staging.exists():  # pragma: no cover - defensive
        shutil.rmtree(staging)
    (staging / "assets").mkdir(parents=True)
    (staging / "targets").mkdir()
    (staging / "structures").mkdir()

    try:
        # Bundled so the viewer keeps working with no network and no server.
        for name in ("molstar.js", "molstar.css", "LICENSE"):
            source = MOLSTAR_DIR / name
            if source.exists():
                shutil.copy2(source, staging / "assets" / name)
        (staging / "assets" / "viewer.css").write_text(PAGE_CSS, encoding="utf-8")
        (staging / "assets" / "sort.js").write_text(SORT_JS, encoding="utf-8")
        (staging / "assets" / "target.js").write_text(TARGET_JS, encoding="utf-8")

        for row in rows:
            slug = row["slug"]
            receptor = clean_dir / f"{row['target_id']}_clean.pdb"
            if receptor.exists():
                destination = _contained(staging / "structures", Path(f"{slug}_receptor.pdb"))
                shutil.copy2(receptor, destination)
                row["receptor_text"] = receptor.read_text(encoding="utf-8", errors="replace")
            else:
                row["missing_receptor"] = True
                row["receptor_text"] = None
            if row["pose"] is not None:
                destination = _contained(staging / "structures", Path(f"{slug}_pose.sdf"))
                shutil.copy2(row["pose"], destination)
                row["pose_text"] = row["pose"].read_text(encoding="utf-8", errors="replace")
            else:
                row["pose_text"] = None

        _write_outputs(rows, staging, run_dir.name, present, context)

        previous = None
        if out_dir.exists():
            previous = out_dir.parent / f".{out_dir.name}.old-{uuid.uuid4().hex[:8]}"
            os.replace(out_dir, previous)
        os.replace(staging, out_dir)
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    missing = sum(1 for row in rows if row.get("missing_receptor"))
    note = f", 수용체 구조 없음 {missing}개" if missing else ""
    print(
        f"뷰어 생성 완료: {out_dir}/index.html  "
        f"({len(rows)}개 표적, 채점 {', '.join(present)}{note})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-metadata", type=Path,
                        default=ROOT / "data" / "hpa" / "proteinatlas.tsv")
    parser.add_argument("--clean-dir", type=Path, default=ROOT / "data" / "human_clean")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()
    build(args.run_dir, args.out_dir, args.target_metadata, args.clean_dir, args.top_n)


if __name__ == "__main__":
    main()
