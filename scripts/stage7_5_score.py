#!/usr/bin/env python3
"""stage7_5_score.py — Score AiZynthFinder routes and apply the §13.2 rules.

Decision (per analog):
    n_routes == 0          → DROP
    min_steps > 8          → DOWNWEIGHT
    stock_fraction < 0.5   → REVIEW
    otherwise              → PROCEED
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage7_5.score")
DECISION_ORDER = {"PROCEED": 0, "REVIEW": 1, "DOWNWEIGHT": 2, "DROP": 3}


def decide(n_routes: int, min_steps: int, stock_fraction: float) -> str:
    if n_routes == 0:
        return "DROP"
    if min_steps > 8:
        return "DOWNWEIGHT"
    if stock_fraction < 0.5:
        return "REVIEW"
    return "PROCEED"


def _is_bool_like(value: object) -> bool:
    return (
        isinstance(value, bool)
        or type(value).__name__ == "bool_"
        or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    )


def _finite_float(value: object, label: str) -> float:
    if _is_bool_like(value):
        raise SystemExit(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"{label} must be finite")
    return parsed


def _nonnegative_int(value: object, label: str) -> int:
    parsed = _finite_float(value, label)
    if parsed < 0 or not parsed.is_integer():
        raise SystemExit(f"{label} must be a non-negative integer")
    return int(parsed)


def _required_route_value(route: dict, keys: tuple[str, ...], label: str) -> object:
    for key in keys:
        if key in route:
            return route[key]
    raise SystemExit(f"{label} is missing")


def score_routes(payload: dict, analog_id: str = "route") -> dict:
    if not payload.get("routes"):
        return {"n_routes": 0, "min_steps": -1, "best_route_score": 0.0,
                "stock_fraction": 0.0,
                "ra_score": None, "sa_score": None, "sc_score": None}
    routes = payload["routes"]
    if not all(isinstance(r, dict) for r in routes):
        raise SystemExit(f"AiZynth route entries must be objects for {analog_id}")
    steps = [
        _nonnegative_int(
            _required_route_value(
                r,
                ("n_steps", "steps"),
                f"AiZynth route n_steps for {analog_id}",
            ),
            f"AiZynth route n_steps for {analog_id}",
        )
        for r in routes
    ]
    route_scores = [
        _finite_float(
            _required_route_value(
                r,
                ("score",),
                f"AiZynth route score for {analog_id}",
            ),
            f"AiZynth route score for {analog_id}",
        )
        for r in routes
    ]
    stock_fractions = []
    for route in routes:
        stock_fraction = _finite_float(
            _required_route_value(
                route,
                ("stock_fraction",),
                f"AiZynth route stock_fraction for {analog_id}",
            ),
            f"AiZynth route stock_fraction for {analog_id}",
        )
        if stock_fraction < 0.0 or stock_fraction > 1.0:
            raise SystemExit(
                f"AiZynth route stock_fraction for {analog_id} must be in [0, 1]"
            )
        stock_fractions.append(stock_fraction)
    best_idx = max(range(len(route_scores)), key=lambda idx: route_scores[idx])
    best_route = routes[best_idx]
    return {
        "n_routes": len(routes),
        "min_steps": min(steps) if steps else 0,
        "best_route_score": route_scores[best_idx],
        "stock_fraction": stock_fractions[best_idx],
        "ra_score": best_route.get("ra_score"),
        "sa_score": best_route.get("sa_score"),
        "sc_score": best_route.get("sc_score"),
    }


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"AiZynth manifest is missing: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"AiZynth manifest is empty: {path}")
    manifest = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    required = {"analog_id", "smiles", "routes_json", "n_routes"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise SystemExit(
            f"AiZynth manifest missing required columns {missing}: {path}"
        )
    if manifest.empty:
        raise SystemExit(f"AiZynth manifest contains no route rows: {path}")
    for col in sorted(required):
        normalized = manifest[col].fillna("").astype(str).str.strip()
        blank_indexes = normalized[normalized == ""].index.tolist()
        if blank_indexes:
            shown = ",".join(str(idx) for idx in blank_indexes[:10])
            suffix = "..." if len(blank_indexes) > 10 else ""
            raise SystemExit(
                f"AiZynth manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        manifest[col] = normalized
    manifest["n_routes"] = [
        _nonnegative_int(value, f"AiZynth manifest n_routes for {analog_id}")
        for analog_id, value in zip(manifest["analog_id"], manifest["n_routes"], strict=True)
    ]
    duplicate_ids = manifest["analog_id"][manifest["analog_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ",".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"AiZynth manifest contains duplicate analog_id values: {shown}{suffix}"
        )
    resolved_routes = manifest["routes_json"].map(
        lambda value: str(Path(value).expanduser().resolve(strict=False))
    )
    duplicate_routes = resolved_routes[resolved_routes.duplicated()].tolist()
    if duplicate_routes:
        shown = ",".join(duplicate_routes[:10])
        suffix = "..." if len(duplicate_routes) > 10 else ""
        raise SystemExit(
            f"AiZynth manifest contains duplicate routes_json values: {shown}{suffix}"
        )
    return manifest


def read_routes_payload(path: Path, analog_id: str) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing AiZynth route JSON for {analog_id}: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty AiZynth route JSON for {analog_id}: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Invalid AiZynth route JSON for {analog_id}: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"AiZynth route JSON is not an object for {analog_id}: {path}"
        )
    routes = payload.get("routes")
    if not isinstance(routes, list):
        raise SystemExit(
            f"AiZynth route JSON missing list field 'routes' for {analog_id}: {path}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-ranking", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_ranking.exists():
        args.out_ranking.unlink()
    manifest = read_manifest(args.manifest)
    rows: list[dict] = []
    for _, m in manifest.iterrows():
        analog_id = str(m["analog_id"])
        path = Path(str(m["routes_json"]))
        payload = read_routes_payload(path, analog_id)
        s = score_routes(payload, analog_id)
        if int(m["n_routes"]) != s["n_routes"]:
            raise SystemExit(
                "AiZynth manifest n_routes does not match route JSON for "
                f"{analog_id}: manifest={int(m['n_routes'])} json={s['n_routes']}"
            )
        decision = decide(s["n_routes"], s["min_steps"], s["stock_fraction"])
        rows.append({
            "analog_id": analog_id,
            "smiles": m["smiles"],
            **s,
            "decision": decision,
        })
    if not rows:
        raise SystemExit(f"No AiZynth routes were scored from manifest: {args.manifest}")
    df = pd.DataFrame(rows)
    df["decision_rank"] = df["decision"].map(DECISION_ORDER)
    df = df.sort_values(["decision_rank", "best_route_score"], ascending=[True, False])
    df = df.drop(columns=["decision_rank"])
    args.out_ranking.parent.mkdir(parents=True, exist_ok=True)
    tmp_ranking = args.out_ranking.with_suffix(args.out_ranking.suffix + ".tmp")
    df.to_csv(tmp_ranking, index=False)
    tmp_ranking.replace(args.out_ranking)
    LOG.info("Wrote %s", args.out_ranking)


if __name__ == "__main__":
    main()
