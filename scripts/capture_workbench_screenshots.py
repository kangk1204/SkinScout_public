#!/usr/bin/env python3
"""Capture the Workbench screens in the order a researcher meets them.

The README's three screenshots predate the input, batch, 3D-embed and download
work, so they no longer show what the tool does. These are taken from the running
application against real committed runs - not mockups - so a collaborator reading
only the top of the README can see what exists today.

Run the Workbench first, then:

    python scripts/capture_workbench_screenshots.py --port 8391
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

# (filename, nav target, what to do before the shot, wait note)
VIEWPORT = {"width": 1440, "height": 900}


def _dismiss_overlays(page) -> None:
    """Close anything transient that would sit over the screen being captured."""
    for selector in ("#toast", ".toast"):
        try:
            page.eval_on_selector(selector, "el => el.style.display = 'none'")
        except Exception:  # noqa: BLE001 - absent is fine
            pass


def _focus_view(page, view_id: str) -> None:
    """Scroll the named view to the top of the window.

    The views are stacked in one document rather than swapped, so clicking a nav
    item leaves the target section below the fold - a screenshot taken straight
    after shows an empty band, not the screen.
    """
    page.eval_on_selector(view_id, "el => el.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(900)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8391)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--example",
        default="Niacinamide",
        help="compound name typed into the search box for the input screenshot",
    )
    parser.add_argument(
        "--run-id", help="which committed run to open for the results screenshots"
    )
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    base = f"http://127.0.0.1:{args.port}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shots: list[tuple[str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, executable_path="/usr/bin/google-chrome"
        )
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(base, wait_until="networkidle")
        page.wait_for_timeout(2500)
        _dismiss_overlays(page)

        def shot(name: str, caption: str) -> None:
            path = args.out_dir / name
            page.screenshot(path=str(path))
            shots.append((name, caption))
            print(f"  {name}  ({path.stat().st_size // 1024} KB)  {caption}")

        # 1. what the tool can do on this machine right now
        shot("wb-1-readiness.png", "readiness: what this machine can run")

        # 2. input - name search with the instant structure check
        page.click('.nav-item[data-view="analyze"]')
        page.wait_for_timeout(1200)
        _focus_view(page, "#view-analyze")
        page.fill("#name-search", args.example)
        page.wait_for_timeout(3000)
        try:
            page.click("#name-results button", timeout=4000)
            page.wait_for_timeout(3500)
        except Exception as error:  # noqa: BLE001
            print(f"  (no name suggestion clicked: {error})", file=sys.stderr)
        _focus_view(page, "#view-analyze")
        _dismiss_overlays(page)
        shot("wb-2-input.png", "input: name search, and the structure checked on the spot")

        # 3. the analysis choices, each with its cost and its disclosure
        try:
            page.eval_on_selector(
                "#analysis-question-title", "el => el.scrollIntoView({block: 'start'})"
            )
            page.wait_for_timeout(1500)
        except Exception:  # noqa: BLE001
            pass
        shot("wb-3-choices.png", "choose an analysis; each says what it costs")

        # 4. results - a run has to be opened from the run list first
        page.click('.nav-item[data-view="runs"]')
        page.wait_for_timeout(3500)
        _focus_view(page, "#view-runs")
        _dismiss_overlays(page)
        shot("wb-4-runs.png", "every run, with its state")

        opened = False
        if args.run_id:
            try:
                page.click(f'.open-run[data-run-id="{args.run_id}"]', timeout=6000)
                opened = True
            except Exception as error:  # noqa: BLE001
                print(f"  (run {args.run_id} not in the list: {error})", file=sys.stderr)
        if not opened:
            try:
                page.click(".open-run", timeout=6000)
                opened = True
            except Exception as error:  # noqa: BLE001
                print(f"  (no run to open: {error})", file=sys.stderr)
        page.wait_for_timeout(6000)
        _focus_view(page, "#view-results")
        _dismiss_overlays(page)
        shot("wb-5-results.png", "results: the verdict, then the ranked targets")

        # 5. the 3D viewer embedded in the results screen
        try:
            page.eval_on_selector(
                "#result-viewer", "el => el.scrollIntoView({block: 'center'})"
            )
            page.wait_for_timeout(6000)
            _dismiss_overlays(page)
            shot("wb-6-viewer.png", "3D structure, in the page, no install")
        except Exception as error:  # noqa: BLE001
            print(f"  (viewer not shown: {error})", file=sys.stderr)

        browser.close()

    print(f"\nwrote {len(shots)} screenshot(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
