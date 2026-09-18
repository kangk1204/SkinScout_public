#!/usr/bin/env python3
"""Verify a local SkinScout Mol* report in desktop and mobile headless Chrome."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Iterator

from PIL import Image


VIEWPORTS = {
    "desktop": (1280, 900),
    "mobile": (390, 844),
}


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@contextmanager
def _report_server(root: Path) -> Iterator[str]:
    def handler(*args: object, **kwargs: object) -> _QuietHandler:
        return _QuietHandler(*args, directory=str(root), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/index.html?selftest=1"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _browser_binary(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["SKINSCOUT_CHROME_BIN"])
        if os.environ.get("SKINSCOUT_CHROME_BIN")
        else None,
        *(Path(path) if path else None for path in (
            shutil.which("google-chrome"),
            shutil.which("google-chrome-stable"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
        )),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "Headless Chrome/Chromium is required to seal the Mol* report; "
        "set SKINSCOUT_CHROME_BIN to an executable browser"
    )


def _screenshot_has_rendered_pixels(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.width < 2 or rgb.height < 2:
            return False
        rgb.thumbnail((256, 256))
        colors = rgb.getcolors(maxcolors=256 * 256)
    return colors is None or len(colors) >= 3


def _playwright_api() -> tuple[Any, type[Exception], type[Exception]]:
    try:
        from playwright.sync_api import (  # type: ignore[import-not-found]
            Error as PlaywrightError,
            TimeoutError as PlaywrightTimeoutError,
            sync_playwright,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Playwright 1.62.0 is required to seal the Mol* report"
        ) from exc
    return sync_playwright, PlaywrightError, PlaywrightTimeoutError


def _download_probe(page: Any) -> dict[str, object]:
    result = page.evaluate(
        """async () => {
          const result = {};
          for (const id of ['download-receptor', 'download-pose']) {
            const anchor = document.getElementById(id);
            const href = anchor && anchor.getAttribute('href');
            if (!href) {
              result[id] = { ok: false, reason: 'missing_href' };
              continue;
            }
            try {
              const response = await fetch(href, { cache: 'no-store' });
              const bytes = response.ok ? (await response.arrayBuffer()).byteLength : 0;
              result[id] = { ok: response.ok && bytes > 0, status: response.status, bytes };
            } catch (error) {
              result[id] = { ok: false, reason: String(error) };
            }
          }
          return result;
        }"""
    )
    if not isinstance(result, dict) or not result:
        raise RuntimeError("Mol* report download probe returned no results")
    failed = [name for name, record in result.items() if not record.get("ok")]
    if failed:
        raise RuntimeError(
            "Mol* report download probe failed for: " + ", ".join(sorted(failed))
        )
    return result


def _verify_viewport(
    *,
    browser: Any,
    url: str,
    viewport_name: str,
    width: int,
    height: int,
    output_dir: Path,
    require_target_switch: bool,
    playwright_timeout_error: type[Exception],
) -> dict[str, str]:
    screenshot = output_dir / f"molstar-{viewport_name}.png"
    context = browser.new_context(
        viewport={"width": width, "height": height},
        device_scale_factor=1,
        is_mobile=viewport_name == "mobile",
        has_touch=viewport_name == "mobile",
    )
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    target_switch_value = "passed" if require_target_switch else "not-applicable"
    try:
        page.wait_for_function(
            """expectedTargetSwitch => {
              const viewer = document.getElementById('molstar-viewer');
              if (document.body.dataset.viewerError) return true;
              return viewer
                && viewer.dataset.viewerLoaded === 'true'
                && viewer.dataset.canvasNonblank === 'true'
                && viewer.dataset.poseSwitchTest === 'passed'
                && viewer.dataset.targetSwitchTest === expectedTargetSwitch;
            }""",
            arg=target_switch_value,
            timeout=60_000,
        )
    except playwright_timeout_error as exc:
        error = page.locator("body").get_attribute("data-viewer-error")
        status = page.locator("#viewer-status").inner_text()
        raise RuntimeError(
            f"Mol* {viewport_name} browser smoke timed out; "
            f"viewer_error={error!r}; status={status!r}; page_errors={page_errors!r}"
        ) from exc

    error = page.locator("body").get_attribute("data-viewer-error")
    if error:
        raise RuntimeError(f"Mol* {viewport_name} browser smoke failed: {error}")

    viewer = page.locator("#molstar-viewer")
    required = {
        "data-viewer-loaded": "true",
        "data-canvas-nonblank": "true",
        "data-pose-switch-test": "passed",
        "data-target-switch-test": target_switch_value,
    }
    for attribute, expected in required.items():
        actual = viewer.get_attribute(attribute)
        if actual != expected:
            raise RuntimeError(
                f"Mol* {viewport_name} verification expected {attribute}={expected!r}, "
                f"got {actual!r}; viewer_error={error!r}"
            )
    interaction_text = page.locator("#interaction-status").inner_text().strip()
    if not interaction_text:
        raise RuntimeError(f"Mol* {viewport_name} interaction status is blank")
    _download_probe(page)
    canvas = viewer.locator("canvas").first
    if canvas.count() != 1:
        raise RuntimeError(f"Mol* {viewport_name} rendered no canvas")
    canvas.screenshot(path=str(screenshot), animations="disabled")
    if not _screenshot_has_rendered_pixels(screenshot):
        raise RuntimeError(f"Mol* {viewport_name} canvas screenshot is blank")
    context.close()
    return {
        "browser_dom": "passed",
        "canvas_pixels": "passed",
        "downloads": "passed",
        "interactions": "passed",
        "pose_switch": "passed",
        "screenshot": "passed",
        "target_switch": "passed" if require_target_switch else "not_applicable",
    }


def _verify_webgl_fallback(
    *,
    browser: Any,
    url: str,
    playwright_timeout_error: type[Exception],
) -> dict[str, str]:
    context = browser.new_context(viewport={"width": 900, "height": 700})
    page = context.new_page()
    page.add_init_script(
        """(() => {
          const original = HTMLCanvasElement.prototype.getContext;
          HTMLCanvasElement.prototype.getContext = function(type, ...args) {
            if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') {
              return null;
            }
            return original.call(this, type, ...args);
          };
        })();"""
    )
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    try:
        page.locator("#viewer-fallback").wait_for(state="visible", timeout=30_000)
    except playwright_timeout_error as exc:
        raise RuntimeError("Mol* WebGL fallback did not become visible") from exc
    error = page.locator("body").get_attribute("data-viewer-error")
    fallback_text = page.locator("#viewer-fallback").inner_text().strip()
    if not error or "3D viewer unavailable" not in fallback_text:
        raise RuntimeError(
            "Mol* WebGL fallback lacks an explicit user-visible failure reason"
        )
    _download_probe(page)
    context.close()
    return {
        "downloads": "passed",
        "fallback_message": "passed",
        "webgl_unavailable": "passed",
    }


def verify_report_package(
    package_root: Path,
    *,
    browser_path: Path | None = None,
    screenshot_dir: Path | None = None,
) -> dict[str, object]:
    package_root = package_root.resolve()
    if not (package_root / "index.html").is_file():
        raise RuntimeError(f"Mol* report index is missing: {package_root / 'index.html'}")
    targets_path = package_root / "targets.json"
    try:
        targets = json.loads(targets_path.read_text(encoding="utf-8"))["targets"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Mol* target metadata is invalid: {targets_path}") from exc
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("Mol* target metadata contains no targets")
    require_target_switch = len(targets) > 1
    browser_executable = _browser_binary(browser_path)
    sync_playwright, playwright_error, playwright_timeout_error = _playwright_api()
    with tempfile.TemporaryDirectory(prefix="skinscout-molstar-") as temporary:
        output_root = screenshot_dir or Path(temporary)
        output_root.mkdir(parents=True, exist_ok=True)
        launch_args = [
            "--disable-dev-shm-usage",
            "--enable-unsafe-swiftshader",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
        ]
        if os.geteuid() == 0:
            launch_args.append("--no-sandbox")
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=str(browser_executable),
                    headless=True,
                    args=launch_args,
                )
                try:
                    with _report_server(package_root) as url:
                        viewports = {
                            name: _verify_viewport(
                                browser=browser,
                                url=url,
                                viewport_name=name,
                                width=size[0],
                                height=size[1],
                                output_dir=output_root,
                                require_target_switch=require_target_switch,
                                playwright_timeout_error=playwright_timeout_error,
                            )
                            for name, size in VIEWPORTS.items()
                        }
                        fallback = _verify_webgl_fallback(
                            browser=browser,
                            url=url,
                            playwright_timeout_error=playwright_timeout_error,
                        )
                finally:
                    browser.close()
        except playwright_error as exc:
            raise RuntimeError(f"Playwright Mol* verification failed: {exc}") from exc
    return {
        "schema_version": "skinscout.report_browser_verification.v1",
        "status": "passed",
        "viewports": viewports,
        "webgl_fallback": fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--screenshot-dir", type=Path)
    args = parser.parse_args()
    result = verify_report_package(
        args.package_root,
        browser_path=args.browser,
        screenshot_dir=args.screenshot_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
