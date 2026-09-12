#!/usr/bin/env python3
"""Validate Stage 2 ADMET / skin-sens runtime wiring before a compound run."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONDA_PACKAGES = {
    "rdkit": "RDKit chemistry parsing and PAINS/Brenk filters",
    "requests": "HuSSPred/STopTox/Pred-Skin web adapters",
}
REQUIRED_PIP_PACKAGES = {
    "admet-ai": "ADMET-AI local model",
}
REQUIRED_ACTIVE_MODULES = {
    "rdkit": "RDKit chemistry parsing and PAINS/Brenk filters",
    "requests": "HuSSPred/STopTox/Pred-Skin web adapters",
    "admet_ai": "ADMET-AI local model",
}
DEGRADED_OPTIONAL_CONDA = {"requests", "admet-ai"}
DEGRADED_OPTIONAL_MODULES = {"requests", "admet_ai"}


def _package_name(spec: str) -> str:
    return re.split(r"[=<>!~\s]", spec.strip(), maxsplit=1)[0].lower()


def _env_packages(env_yml: Path) -> tuple[set[str], set[str]]:
    import yaml

    payload = yaml.safe_load(env_yml.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"Conda env file must be a YAML object: {env_yml}")
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        raise SystemExit(f"Conda env file missing dependency list: {env_yml}")

    conda: set[str] = set()
    pip: set[str] = set()
    for item in dependencies:
        if isinstance(item, str):
            conda.add(_package_name(item))
        elif isinstance(item, dict):
            pip_items = item.get("pip")
            if isinstance(pip_items, list):
                pip.update(_package_name(str(dep)) for dep in pip_items)
    return conda, pip


def _check_env_manifest(
    env_yml: Path,
    *,
    allow_degraded: bool = False,
) -> list[dict[str, Any]]:
    conda, pip = _env_packages(env_yml)
    checks: list[dict[str, Any]] = []
    for package, reason in sorted(REQUIRED_CONDA_PACKAGES.items()):
        present = package in conda or package in pip
        optional = allow_degraded and package in DEGRADED_OPTIONAL_CONDA
        checks.append(
            {
                "name": f"env:{package}",
                "ok": present or optional,
                "reason": reason,
                "warning": not present and optional,
            }
        )
    for package, reason in sorted(REQUIRED_PIP_PACKAGES.items()):
        present = package in pip or package in conda
        optional = allow_degraded and package in DEGRADED_OPTIONAL_CONDA
        checks.append(
            {
                "name": f"env:{package}",
                "ok": present or optional,
                "reason": reason,
                "warning": not present and optional,
            }
        )
    return checks


def _check_active_modules(*, allow_degraded: bool = False) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for module, reason in sorted(REQUIRED_ACTIVE_MODULES.items()):
        present = importlib.util.find_spec(module) is not None
        optional = allow_degraded and module in DEGRADED_OPTIONAL_MODULES
        checks.append(
            {
                "name": f"module:{module}",
                "ok": present or optional,
                "reason": reason,
                "warning": not present and optional,
            }
        )
    return checks


def _constant_from_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    raise SystemExit(f"Could not find string constant {name} in {path}")


def _check_stage2_endpoints() -> list[dict[str, Any]]:
    husspred_url = _constant_from_source(ROOT / "scripts/stage2_husspred.py", "API_URL")
    stoptox_url = _constant_from_source(ROOT / "scripts/stage2_stoptox.py", "WEB_URL")
    pred_skin_url = _constant_from_source(
        ROOT / "scripts/stage2_pred_skin.py",
        "PRED_SKIN_URL",
    )
    return [
        {
            "name": "endpoint:husspred",
            "ok": husspred_url.rstrip("/") == "https://husspred.mml.unc.edu/smiles",
            "reason": "HuSSPred public web-client endpoint",
        },
        {
            "name": "endpoint:stoptox_web",
            "ok": stoptox_url.rstrip("/") == "https://stoptox.mml.unc.edu/predict",
            "reason": "STopTox public web result endpoint",
        },
        {
            "name": "endpoint:pred_skin",
            "ok": pred_skin_url.rstrip("/")
            == "https://predskin.labmol.com.br/api/predskin/predict",
            "reason": "Pred-Skin public task submission endpoint",
        },
    ]


def _online_probe(name: str, url: str) -> dict[str, Any]:
    try:
        import requests

        response = requests.get(url, timeout=15, allow_redirects=True)
        ok = 200 <= response.status_code < 300
        return {
            "name": f"online:{name}",
            "ok": ok,
            "status_code": response.status_code,
            "reason": url,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": f"online:{name}",
            "ok": False,
            "error": str(exc),
            "reason": url,
        }


def _check_online() -> list[dict[str, Any]]:
    return [
        _online_probe("husspred_models", "https://husspred.mml.unc.edu/models"),
        _online_probe("stoptox_home", "https://stoptox.mml.unc.edu/"),
        _online_probe("predskin_page", "https://predskin.labmol.com.br/predskin"),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-yml", type=Path, default=ROOT / "envs/dti.yml")
    parser.add_argument(
        "--runtime-mode",
        choices=["conda", "active"],
        default="conda",
        help="Check conda env manifest or active Python imports.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Probe current public web-service reachability.",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Treat optional ADMET-AI/web-adapter availability as degraded warnings.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = []
    if args.runtime_mode == "conda":
        checks.extend(
            _check_env_manifest(args.env_yml, allow_degraded=args.allow_degraded)
        )
    else:
        checks.extend(_check_active_modules(allow_degraded=args.allow_degraded))
    checks.extend(_check_stage2_endpoints())
    if args.online:
        online_checks = _check_online()
        if args.allow_degraded:
            for check in online_checks:
                if not check["ok"]:
                    check["ok"] = True
                    check["warning"] = True
        checks.extend(online_checks)

    failed = [check for check in checks if not check["ok"]]
    payload = {"status": "ok" if not failed else "failed", "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for check in checks:
            status = (
                "WARN" if check.get("warning") else ("OK" if check["ok"] else "FAIL")
            )
            print(f"[{status}] {check['name']} - {check['reason']}")
    if failed:
        if not args.json:
            print("Safety readiness preflight failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
