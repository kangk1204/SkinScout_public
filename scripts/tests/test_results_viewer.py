"""Regression tests for the offline results viewer.

The viewer is what a wet-lab collaborator actually opens, so it has to work
from a plain file:// double-click with no server and no network.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_results_viewer.py"


# Minimal but genuinely parseable structures. Mol* rejects a placeholder such
# as a bare "ATOM" line, and the browser gate below is the only test that can
# tell the difference.
_MINIMAL_PDB = """\
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
ATOM      3  C   ALA A   1      13.146   6.243  -5.155  1.00  0.00           C
ATOM      4  O   ALA A   1      13.708   6.923  -6.013  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.006   7.144  -4.276  1.00  0.00           C
TER       6      ALA A   1
END
"""

_MINIMAL_SDF = """\
{name}
  SkinScout test fixture

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$
"""


def _write_run(tmp_path: Path, *, with_boltz: bool = True) -> Path:
    run = tmp_path / "run_case"
    stage = run / "03_targets" / "mode_comprehensive"
    stage.mkdir(parents=True)
    poses = stage / "autodock_all_target_poses"
    poses.mkdir()
    rows = [("P14679", 1, 0.05, 4), ("P29274", 2, 0.04, 3)]
    with (stage / "top50_4way_consensus.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["recipe_id", "target_id", "rrf_score", "source_count", "sources",
             "autodock_rank", "gnina_rank", "rtm_rank", "boltz_rank"]
        )
        for target, rank, score, count in rows:
            writer.writerow(["r1", target, score, count, "autodock;gnina", rank, rank, rank, rank])
    (stage / "autodock_all_targets.tsv").write_text(
        "target_id\tvina_score\nP14679\t-6.10\nP29274\t-5.20\n"
    )
    (stage / "gnina_rescores.tsv").write_text(
        "target_id\tcnn_affinity\nP14679\t4.10\nP29274\t3.90\n"
    )
    (stage / "rtmscore_rescores.tsv").write_text(
        "target_id\trtm_score\nP14679\t12.0\nP29274\t9.5\n"
    )
    if with_boltz:
        (stage / "boltz2_affinity_top.tsv").write_text(
            "target_id\tboltz2_neg_log_uM\nP14679\t0.44\nP29274\t0.31\n"
        )
    for target, *_ in rows:
        (poses / f"{target}.sdf").write_text(_MINIMAL_SDF.format(name=target))
    return run


def _clean_dir(tmp_path: Path, targets: list[str]) -> Path:
    clean = tmp_path / "clean"
    clean.mkdir(exist_ok=True)
    for target in targets:
        (clean / f"{target}_clean.pdb").write_text(_MINIMAL_PDB)
    return clean


def _metadata(tmp_path: Path) -> Path:
    path = tmp_path / "proteinatlas.tsv"
    path.write_text(
        "Gene\tUniprot\tGene description\tMolecular function\n"
        "TYR\tP14679\tTyrosinase\tEnzyme\n"
        "ADORA2A\tP29274\tAdenosine receptor A2a\tG-protein coupled receptor\n"
    )
    return path


def _build(tmp_path: Path, **kwargs) -> Path:
    run = _write_run(tmp_path, **kwargs)
    out = tmp_path / "viewer"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)),
         "--clean-dir", str(_clean_dir(tmp_path, ["P14679", "P29274"]))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


def test_the_viewer_is_self_contained(tmp_path: Path) -> None:
    """It has to open from file:// on a machine that never ran the pipeline."""
    out = _build(tmp_path)
    assert (out / "index.html").exists()
    assert (out / "assets" / "molstar.js").exists()
    assert (out / "assets" / "molstar.css").exists()
    for target in ("P14679", "P29274"):
        assert (out / "targets" / f"{target}.html").exists()
        assert (out / "structures" / f"{target}_receptor.pdb").exists()
        assert (out / "structures" / f"{target}_pose.sdf").exists()

    # No absolute paths and no remote origins anywhere in the pages.
    for page in out.rglob("*.html"):
        text = page.read_text()
        assert "http://" not in text
        assert "https://" not in text
        assert str(ROOT) not in text


def test_gene_names_replace_accessions_for_a_reader(tmp_path: Path) -> None:
    index = (_build(tmp_path) / "index.html").read_text()
    assert "TYR" in index
    assert "Tyrosinase" in index
    assert "P14679" in index


def test_a_receptor_limitation_is_shown_where_it_applies(tmp_path: Path) -> None:
    """A GPCR docking score must not be read as weak binding without warning."""
    out = _build(tmp_path)
    index = (out / "index.html").read_text()
    assert "GPCR" in index
    page = (out / "targets" / "P29274.html").read_text()
    assert "GPCR" in page
    # Tyrosinase is curated for its copper centre.
    assert "금속" in (out / "targets" / "P14679.html").read_text()


def test_the_summary_opens_in_excel(tmp_path: Path) -> None:
    """A BOM is what makes Excel read the Korean headers correctly.

    The headers are Korean too. This file exists for a Korean researcher to
    double-click into Excel, and nothing in the repo parses it, so English
    column names only made the one human-facing export harder to read.
    """
    path = _build(tmp_path) / "summary.csv"
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    assert [row["UniProt"] for row in rows] == ["P14679", "P29274"]
    assert rows[0]["유전자"] == "TYR"
    assert "합의 점수" in rows[0]


def test_a_missing_scorer_is_simply_absent(tmp_path: Path) -> None:
    """Runs without Boltz-2 still produce a viewer rather than failing."""
    out = _build(tmp_path, with_boltz=False)
    index = (out / "index.html").read_text()
    assert "Boltz-2" not in index
    assert "GNINA" in index


def test_every_page_declares_utf8(tmp_path: Path) -> None:
    for page in _build(tmp_path).rglob("*.html"):
        assert 'charset="utf-8"' in page.read_text()


def test_a_finished_run_gets_its_viewer_without_being_asked(tmp_path: Path) -> None:
    """A wet-lab reader should find the viewer already there.

    Requiring a command after every run is exactly the step that does not
    happen, so run_skinscout builds it as part of finishing.
    """
    source = (ROOT / "scripts" / "run_skinscout.py").read_text()
    assert "_build_results_viewer(run_dir)" in source
    # Built in-process so it cannot disturb the pipeline's subprocess sequence.
    assert "from make_results_viewer import build as build_viewer" in source


def test_a_viewer_failure_never_fails_a_finished_run(tmp_path: Path) -> None:
    """The results already exist; the viewer is a convenience over them."""
    source = (ROOT / "scripts" / "run_skinscout.py").read_text()
    body = source.split("def _build_results_viewer", 1)[1].split("\ndef ", 1)[0]
    assert "_die(" not in body, "a viewer problem must not abort the run"
    assert "분석 결과 자체는 정상입니다" in body


def _write_fast_run(tmp_path: Path) -> Path:
    """A Daina fast-path run: different filenames, evidence carried inline."""
    run = tmp_path / "fast_case"
    stage = run / "03_targets" / "mode_fast"
    stage.mkdir(parents=True)
    (stage / "poses").mkdir()
    with (stage / "top50.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "target_id", "rrf_score", "source_count", "sources",
            "autodock_energy_kcal_mol", "gnina_cnn_affinity",
            "daina_max_tanimoto", "daina_is_self_match", "daina_supporting_molecule_id",
        ])
        writer.writerow(["P14679", 1.0, 2, "daina;gnina", -5.1, 4.2, 1.0, "True", "CHEMBL113"])
        writer.writerow(["P29274", 0.6, 2, "daina;gnina", -4.4, 3.1, 0.61, "False", "CHEMBL9999"])
    (stage / "autodock_top5k.tsv").write_text(
        "target_id\tvina_score\nP14679\t-5.10\nP29274\t-4.40\n"
    )
    (stage / "gnina_pose_rescores.tsv").write_text(
        "target_id\tcnn_affinity\nP14679\t4.20\nP29274\t3.10\n"
    )
    for target in ("P14679", "P29274"):
        (stage / "poses" / f"{target}.sdf").write_text(f"{target}\n  x\n\nM  END\n$$$$\n")
    return run


def _build_fast(tmp_path: Path) -> Path:
    run = _write_fast_run(tmp_path)
    out = tmp_path / "fast_viewer"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)),
         "--clean-dir", str(_clean_dir(tmp_path, ["P14679", "P29274"]))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


def test_the_fast_path_scores_are_found_under_their_own_filenames(tmp_path: Path) -> None:
    """The two paths name their score tables differently.

    Reading only the comprehensive names left a Daina run - which is what the
    Demo install produces - showing a ranking with no scores at all.
    """
    index = (_build_fast(tmp_path) / "index.html").read_text()
    assert "AutoDock ΔG" in index
    assert "GNINA CNN" in index
    assert "-5.100" in index or "-5.1" in index


def test_retrieval_is_distinguished_from_prediction(tmp_path: Path) -> None:
    """The most important thing on a fast-path row, and it is a fact not a score.

    A well characterised cosmetic ingredient is usually its own reference
    ligand, so its top hits are looked-up interactions rather than predictions.
    """
    out = _build_fast(tmp_path)
    index = (out / "index.html").read_text()
    assert "조회" in index and "예측" in index
    assert "CHEMBL113" in index  # the reference molecule, on hover

    looked_up = (out / "targets" / "P14679.html").read_text()
    assert "조회" in looked_up
    assert "새로 예측한 결과가 아닙니다" in looked_up
    predicted = (out / "targets" / "P29274.html").read_text()
    assert "예측" in predicted
    assert "외삽한 결과" in predicted


def test_the_summary_carries_the_evidence_columns(tmp_path: Path) -> None:
    path = _build_fast(tmp_path) / "summary.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    assert rows[0]["근거"] == "조회"
    assert rows[0]["참조 분자"] == "CHEMBL113"
    assert rows[1]["근거"] == "예측"


def test_a_run_without_evidence_omits_the_column(tmp_path: Path) -> None:
    """Comprehensive runs have no Daina retrieval, so the column would be empty."""
    index = (_build(tmp_path) / "index.html").read_text()
    assert "최근접 유사도" not in index


# --- Defects an independent audit found in the first version of this viewer ---


def test_structures_are_embedded_so_a_file_url_can_render_them(tmp_path: Path) -> None:
    """A file:// page cannot fetch its own siblings.

    The first version called loadStructureFromUrl, so double-clicking the page
    - the exact path the README documents - produced a blank canvas with a CORS
    error. The structure text has to be in the page.
    """
    out = _build(tmp_path)
    page = (out / "targets" / "P14679.html").read_text(encoding="utf-8")
    # The Mol* code is a bundled asset so the served copy needs no
    # 'unsafe-inline'; the structures stay inline as data.
    script = (out / "assets" / "target.js").read_text(encoding="utf-8")
    assert "loadStructureFromUrl" not in script
    assert "loadStructureFromData" in script
    payload = re.search(
        r'<script id="structure-data" type="application/json">(.*?)</script>',
        page, re.DOTALL,
    )
    assert payload is not None
    data = json.loads(payload.group(1))
    assert data["receptor"].startswith("ATOM")
    assert "P14679" in data["pose"]


def test_embedded_structure_json_cannot_terminate_its_script_element() -> None:
    spec = importlib.util.spec_from_file_location("results_viewer_xss_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    encoded = module._script_safe_json({"pose": "</script><script>alert(1)</script>"})

    assert "</script" not in encoded.lower()
    assert json.loads(encoded)["pose"] == "</script><script>alert(1)</script>"


def test_the_viewer_clears_through_the_plugin(tmp_path: Path) -> None:
    """Mol*'s Viewer has no clear(); it lives on viewer.plugin."""
    script = (_build(tmp_path) / "assets" / "target.js").read_text(encoding="utf-8")
    assert "viewer.plugin.clear()" in script
    assert "await viewer.clear()" not in script


def test_units_are_shown_instead_of_a_column_name(tmp_path: Path) -> None:
    """The metric caption read SCORE_FILES[key][2] - the column name, not the units."""
    page = (_build(tmp_path) / "targets" / "P14679.html").read_text(encoding="utf-8")
    assert "낮을수록 좋음 (kcal/mol)" in page
    assert "높을수록 좋음" in page
    assert "autodock_energy_kcal_mol" not in page
    assert "<span>None</span>" not in page


def test_a_hostile_target_id_cannot_escape_the_output_directory(tmp_path: Path) -> None:
    """Target IDs come from a CSV, so they are untrusted input."""
    run = _write_run(tmp_path)
    stage = run / "03_targets" / "mode_comprehensive"
    consensus = stage / "top50_4way_consensus.csv"
    text = consensus.read_text().replace("P29274", "../../escaped")
    consensus.write_text(text)
    out = tmp_path / "nested" / "viewer"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)),
         "--clean-dir", str(_clean_dir(tmp_path, ["P14679"]))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "escaped.html").exists()
    assert not (tmp_path / "nested" / "escaped.html").exists()
    written = {path.name for path in (out / "targets").iterdir()}
    assert all(".." not in name for name in written)


def test_an_empty_comprehensive_directory_falls_back_to_the_fast_path(tmp_path: Path) -> None:
    """A stray file in mode_comprehensive used to win over a populated mode_fast."""
    run = _write_run(tmp_path)
    stage = run / "03_targets" / "mode_comprehensive"
    fast = run / "03_targets" / "mode_fast"
    fast.mkdir(parents=True)
    for item in stage.iterdir():
        item.rename(fast / item.name)
    (stage / "ligand.pdbqt").write_text("stray\n")

    out = tmp_path / "viewer"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)),
         "--clean-dir", str(_clean_dir(tmp_path, ["P14679", "P29274"]))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "index.html").exists()


def test_a_missing_receptor_is_explained_rather_than_shown_blank(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    out = tmp_path / "viewer"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)),
         # Only one of the two targets has a prepared receptor.
         "--clean-dir", str(_clean_dir(tmp_path, ["P14679"]))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    page = (out / "targets" / "P29274.html").read_text(encoding="utf-8")
    assert "준비된 수용체 구조가 없어" in page
    assert "결합이 약하다는 뜻이 아니라" in page
    assert "수용체 구조 없음 1개" in result.stdout


def test_source_count_is_not_described_as_agreement(tmp_path: Path) -> None:
    """Measured: only GNINA-RTMScore correlate; the median top-50 spread is 209."""
    index = (_build(tmp_path) / "index.html").read_text(encoding="utf-8")
    assert "여러 방법이 동의했다는 뜻입니다" not in index
    assert "방법들이 동의했다는 뜻이 아닙니다" in index


def test_per_method_ranks_reach_the_reader(tmp_path: Path) -> None:
    """They were collected and then dropped, hiding how far the methods disagree."""
    out = _build(tmp_path)
    assert "순위 편차" in (out / "index.html").read_text(encoding="utf-8")
    page = (out / "targets" / "P14679.html").read_text(encoding="utf-8")
    assert "방법별 순위" in page
    rows = list(csv.DictReader(
        (out / "summary.csv").read_text(encoding="utf-8-sig").splitlines()
    ))
    assert "AutoDock ΔG 순위" in rows[0]


def test_a_failed_build_leaves_the_previous_viewer_alone(tmp_path: Path) -> None:
    """A half-written viewer that replaced a good one reads as a current result."""
    out = _build(tmp_path)
    marker = (out / "index.html").read_text(encoding="utf-8")

    broken = tmp_path / "broken_run"
    (broken / "03_targets" / "mode_fast").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(broken), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)),
         "--clean-dir", str(_clean_dir(tmp_path, []))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert (out / "index.html").read_text(encoding="utf-8") == marker
    leftovers = [p.name for p in out.parent.iterdir() if ".staging-" in p.name]
    assert leftovers == []


@pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None
    or shutil.which("google-chrome") is None,
    reason="Playwright and local Google Chrome are required for the browser gate",
)
def test_the_viewer_renders_in_a_real_browser_from_a_file_url(tmp_path: Path) -> None:
    """The completion gate for this viewer.

    Static assertions cannot tell a page that renders from one that throws:
    the previous version passed every unit test in this file while a real
    browser reported `viewer.clear is not a function` and drew nothing. This
    opens the page the way the README tells a reader to - a file:// URL, no
    server - and asks Mol* itself what it loaded.
    """
    from playwright.sync_api import sync_playwright

    page_path = (_build(tmp_path) / "targets" / "P14679.html").resolve()
    with sync_playwright() as engine:
        browser = engine.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        console_errors: list[str] = []
        page_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.goto(page_path.as_uri())
        page.wait_for_timeout(6000)
        state = page.evaluate(
            """() => {
              const fallback = document.getElementById('fallback');
              const cells = Array.from(window.__viewerPlugin.state.data.cells.values());
              let trajectories = 0, atoms = 0;
              for (const cell of cells) {
                const obj = cell.obj;
                if (!obj || !obj.type) continue;
                if (obj.type.name === 'Trajectory') trajectories += 1;
                if (obj.type.name === 'Model') {
                  atoms = Math.max(atoms, obj.data.atomicHierarchy.atoms._rowCount || 0);
                }
              }
              return {
                fallbackVisible: !fallback.hidden,
                fallbackText: fallback.textContent,
                canvases: document.querySelectorAll('#viewer canvas').length,
                trajectories,
                atoms,
              };
            }"""
        )
        browser.close()

    assert page_errors == []
    assert console_errors == []
    assert state["fallbackVisible"] is False, state["fallbackText"]
    assert state["canvases"] == 1
    # Receptor and pose, each loaded exactly once.
    assert state["trajectories"] == 2
    assert state["atoms"] > 0



def test_the_pages_carry_no_executable_inline_script(tmp_path: Path) -> None:
    """The Workbench serves this viewer under a CSP without 'unsafe-inline'.

    Keeping the code in bundled assets means the embedded copy renders without
    relaxing script-src, and the file:// copy is unaffected because a
    same-directory <script src> is always allowed there.
    """
    out = _build(tmp_path)
    for page in out.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        for match in re.finditer(r"<script([^>]*)>", text):
            attributes = match.group(1)
            assert "src=" in attributes or 'type="application/json"' in attributes, (
                f"{page.name} has an inline executable script: {match.group(0)}"
            )
    assert (out / "assets" / "sort.js").is_file()
    assert (out / "assets" / "target.js").is_file()


def _build_many(tmp_path: Path, count: int = 6) -> Path:
    """A run with enough targets for the charts to be worth drawing."""
    run = tmp_path / "many"
    stage = run / "03_targets" / "mode_comprehensive"
    stage.mkdir(parents=True)
    targets = [f"P{index:05d}" for index in range(1, count + 1)]
    with (stage / "top50_4way_consensus.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_id", "rrf_score", "source_count", "sources",
                         "autodock_rank", "gnina_rank", "rtm_rank", "boltz_rank"])
        for position, target in enumerate(targets, start=1):
            # Deliberately divergent: this is what the chart exists to show.
            writer.writerow([target, 0.05 - position * 0.001, 4, "autodock;gnina",
                             position, position * 7, position * 3, position * 20])
    (stage / "autodock_all_targets.tsv").write_text(
        "target_id\tvina_score\n"
        + "".join(f"{target}\t{-7.0 + index * 0.1:.2f}\n" for index, target in enumerate(targets))
        + "".join(f"Q{index:05d}\t{-4.0 + index * 0.01:.2f}\n" for index in range(1, 200))
    )
    (stage / "gnina_rescores.tsv").write_text(
        "target_id\tcnn_affinity\n"
        + "".join(f"{target}\t{4.0 + index * 0.05:.2f}\n" for index, target in enumerate(targets))
    )
    clean = tmp_path / "clean_many"
    clean.mkdir()
    for target in targets:
        (clean / f"{target}_clean.pdb").write_text(_MINIMAL_PDB)
    out = tmp_path / "viewer_many"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)), "--clean-dir", str(clean)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


def test_the_index_draws_the_disagreement_it_describes(tmp_path: Path) -> None:
    """The help text says the methods do not agree; the chart has to show it."""
    index = (_build_many(tmp_path) / "index.html").read_text(encoding="utf-8")

    assert "방법마다 순위가 얼마나 갈리나" in index
    assert "이 표적들은 전체에서 어디쯤인가" in index
    assert "<svg" in index
    # Text, not glyph paths: matplotlib's bundled font has no Hangul, so drawing
    # the labels as paths silently dropped every Korean word.
    assert "<text" in index
    assert "각 방법이 매긴 순위" in index


def test_charts_add_no_javascript_dependency(tmp_path: Path) -> None:
    """The folder is meant to be zipped and emailed; a chart library is megabytes."""
    out = _build(tmp_path)
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "plotly" not in index.lower()
    assert "d3.js" not in index.lower()
    assets = {path.name for path in (out / "assets").iterdir()}
    assert assets <= {"molstar.js", "molstar.css", "LICENSE", "viewer.css",
                      "sort.js", "target.js"}


def _write_reranked_run(tmp_path: Path) -> Path:
    """A run whose reader-facing table came from the band re-ranking."""
    run = tmp_path / "reranked_run"
    stage = run / "03_targets" / "mode_fast"
    stage.mkdir(parents=True)
    rows = []
    for position in range(1, 13):
        # Two rows moved: similarity had them lower than the table shows.
        daina_rank = {11: 25, 12: 27}.get(position, position)
        rows.append({
            "target_id": f"P{daina_rank:05d}",
            "final_rank": position,
            "daina_rank": daina_rank,
            "rerank_band": position >= 11,
            "ranking_basis": "daina_head_then_band_rrf",
            "rrf_score": 0.9 - 0.01 * daina_rank,
            "source_count": 2,
            "sources": "daina;autodock",
            "autodock_energy_kcal_mol": -6.0 - 0.01 * position,
        })
    pd = pytest.importorskip("pandas")
    pd.DataFrame(rows).to_csv(stage / "top50_band_reranked.csv", index=False)
    # The similarity-ordered artifact stays beside it, untouched.
    pd.DataFrame(rows).sort_values("daina_rank").to_csv(stage / "top50.csv", index=False)
    return run


def test_the_viewer_prefers_the_reranked_table(tmp_path: Path) -> None:
    run = _write_reranked_run(tmp_path)
    out = tmp_path / "viewer"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run), "--out-dir", str(out),
         "--target-metadata", str(_metadata(tmp_path)),
         "--clean-dir", str(_clean_dir(tmp_path, []))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    index = (out / "index.html").read_text(encoding="utf-8")
    # The pre-rerank position must stay on screen; otherwise a reader cannot
    # tell that docking moved anything.
    assert "유사도 순위" in index
    assert "상위 10위는 손대지 않고" in index


def test_a_similarity_ordered_run_shows_no_rerank_column(tmp_path: Path) -> None:
    """Nothing moved, so there is nothing to explain."""
    index = (_build(tmp_path) / "index.html").read_text(encoding="utf-8")

    assert "유사도 순위" not in index
