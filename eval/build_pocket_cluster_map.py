#!/usr/bin/env python3
"""Bind Foldseek pocket-fragment clusters to the full target universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "skinscout.pocket-cluster-map.v1"
FRAGMENT_SCHEMA_VERSION = "skinscout.pocket-fragment-universe.v1"
TARGET_CLUSTER_SCHEMA_VERSION = "skinscout.target-cluster-map.v2"
# The screenable proteome map carries the same artifact contract under its own
# schema name; accepting only the evidence map kept pocket clustering inside the
# population that already had ligand evidence.
SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION = "skinscout.screenable-target-cluster-map.v2"
SUPPORTED_TARGET_CLUSTER_SCHEMA_VERSIONS = (
    TARGET_CLUSTER_SCHEMA_VERSION,
    SCREENABLE_TARGET_CLUSTER_SCHEMA_VERSION,
)
EXPECTED_FIELDS = [
    "uniprot",
    "pocket_available",
    "exclusion_reason",
    "pocket_cluster_tm40",
    "pocket_cluster_tm50",
    "pocket_cluster_tm60",
]
FRAGMENT_INDEX_FIELDS = {
    "uniprot",
    "fragment_path",
    "fragment_sha256",
}
FRAGMENT_NAME_RE = re.compile(r"^(?P<uniprot>.+)_p2rank_top1_context(?P<context>\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256(path).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{label} is required and must be non-empty: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    _require_file(path, label)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{label} must contain a CSV header: {path}")
        return [
            {str(key).strip(): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _unique_uniprots(rows: list[dict[str, str]], label: str) -> list[str]:
    values = [row.get("uniprot", "").strip() for row in rows]
    if any(not value for value in values):
        raise SystemExit(f"{label} contains blank uniprot values")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise SystemExit(f"{label} contains duplicate uniprot values: {', '.join(duplicates[:10])}")
    return values


def _artifact(payload: dict[str, Any], key: str | None = None) -> dict[str, Any]:
    if key is None:
        artifact = payload.get("artifact")
    else:
        artifacts = payload.get("artifacts")
        artifact = artifacts.get(key) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        name = "artifact" if key is None else f"artifacts.{key}"
        raise SystemExit(f"manifest missing {name} object")
    return artifact


def _validate_artifact(path: Path, rows: list[dict[str, str]], artifact: dict[str, Any], label: str) -> None:
    expected_sha = str(artifact.get("sha256") or "").strip()
    if expected_sha != _sha256(path):
        raise SystemExit(f"{label} sha256 does not match manifest")
    if artifact.get("rows") != len(rows):
        raise SystemExit(f"{label} row count does not match manifest")


def _read_targets(path: Path, manifest_path: Path, expected_count: int) -> tuple[list[str], dict[str, Any]]:
    rows = _read_csv(path, "target cluster CSV")
    targets = sorted(_unique_uniprots(rows, "target cluster CSV"))
    if len(targets) != expected_count:
        raise SystemExit(
            f"target cluster CSV must contain exactly {expected_count} targets; observed {len(targets)}"
        )
    manifest = _read_json(manifest_path, "target cluster manifest")
    if manifest.get("schema_version") not in SUPPORTED_TARGET_CLUSTER_SCHEMA_VERSIONS:
        allowed = " or ".join(SUPPORTED_TARGET_CLUSTER_SCHEMA_VERSIONS)
        raise SystemExit(f"target cluster manifest schema must be {allowed}")
    _validate_artifact(path, rows, _artifact(manifest), "target cluster CSV")
    return targets, manifest


def _manifest_path_matches(manifest_path: str, supplied: Path) -> bool:
    return bool(manifest_path) and Path(manifest_path).resolve() == supplied.resolve()


def _read_fragment_universe(
    index_path: Path,
    exclusions_path: Path,
    manifest_path: Path,
    targets: list[str],
    target_csv: Path,
    target_manifest_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, Any], Path]:
    index_rows = _read_csv(index_path, "pocket fragment index CSV")
    exclusions = _read_csv(exclusions_path, "pocket fragment exclusions CSV")
    fields = set(index_rows[0].keys()) if index_rows else set(FRAGMENT_INDEX_FIELDS)
    if not FRAGMENT_INDEX_FIELDS <= fields:
        raise SystemExit("pocket fragment index CSV missing required columns")
    accepted_ids = _unique_uniprots(index_rows, "pocket fragment index CSV")
    excluded_ids = _unique_uniprots(exclusions, "pocket fragment exclusions CSV")
    exclusion_reasons = {row["uniprot"]: row.get("reason", "") for row in exclusions}
    if any(not reason for reason in exclusion_reasons.values()):
        raise SystemExit("pocket fragment exclusions CSV contains blank reason values")

    manifest = _read_json(manifest_path, "pocket fragment manifest")
    if manifest.get("schema_version") != FRAGMENT_SCHEMA_VERSION:
        raise SystemExit(
            f"pocket fragment manifest schema must be {FRAGMENT_SCHEMA_VERSION}"
        )
    _validate_artifact(index_path, index_rows, _artifact(manifest, "index"), "pocket fragment index CSV")
    _validate_artifact(exclusions_path, exclusions, _artifact(manifest, "exclusions"), "pocket fragment exclusions CSV")

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise SystemExit("pocket fragment manifest missing counts object")
    if counts.get("targets") != len(targets):
        raise SystemExit("pocket fragment manifest target count does not match target CSV")
    if counts.get("accepted") != len(index_rows) or counts.get("excluded") != len(exclusions):
        raise SystemExit("pocket fragment manifest accepted/excluded counts do not match CSV inputs")

    target_prov = manifest.get("target_manifest_provenance")
    if not isinstance(target_prov, dict):
        raise SystemExit("pocket fragment manifest missing target manifest provenance")
    target_artifact = target_prov.get("artifact")
    if not isinstance(target_artifact, dict):
        raise SystemExit("pocket fragment manifest missing target artifact provenance")
    if not _manifest_path_matches(str(target_prov.get("path") or ""), target_manifest_path):
        raise SystemExit("pocket fragment manifest target manifest path does not match input")
    if str(target_prov.get("sha256") or "").strip() != _sha256(target_manifest_path):
        raise SystemExit("pocket fragment manifest target manifest sha256 mismatch")
    if not _manifest_path_matches(str(target_artifact.get("path") or ""), target_csv):
        raise SystemExit("pocket fragment manifest target CSV path does not match input")
    if str(target_artifact.get("sha256") or "").strip() != _sha256(target_csv):
        raise SystemExit("pocket fragment manifest target CSV sha256 mismatch")
    if target_artifact.get("rows") != len(targets):
        raise SystemExit("pocket fragment manifest target CSV row count mismatch")

    accepted = set(accepted_ids)
    excluded = set(excluded_ids)
    target_set = set(targets)
    if accepted & excluded:
        raise SystemExit("pocket fragment accepted/excluded sets overlap")
    if accepted | excluded != target_set:
        raise SystemExit("pocket fragment accepted/excluded sets must exactly partition target universe")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit("pocket fragment manifest missing artifacts object")
    out_dir = Path(str(artifacts.get("out_dir") or "")).resolve()
    if not out_dir.exists() or not out_dir.is_dir():
        raise SystemExit(f"pocket fragment output directory is missing: {out_dir}")
    expected_tree = str(artifacts.get("out_dir_tree_sha256") or "").strip()
    if expected_tree != _tree_digest(out_dir):
        raise SystemExit("pocket fragment output directory tree digest mismatch")

    by_uniprot: dict[str, dict[str, str]] = {}
    for row in index_rows:
        fragment_path = out_dir / row["fragment_path"]
        _require_file(fragment_path, "pocket fragment file")
        if row["fragment_sha256"] != _sha256(fragment_path):
            raise SystemExit(f"pocket fragment file sha256 mismatch: {row['uniprot']}")
        by_uniprot[row["uniprot"]] = row
    return by_uniprot, exclusion_reasons, manifest, out_dir


def _expected_foldseek_ids(index: dict[str, dict[str, str]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact: dict[str, str] = {}
    prefixes: dict[str, list[str]] = {}
    contexts: set[str] = set()
    for uniprot, row in index.items():
        stem = Path(row["fragment_path"]).stem
        match = FRAGMENT_NAME_RE.match(stem)
        if not match or match.group("uniprot") != uniprot:
            raise SystemExit(
                f"fragment_path for {uniprot} must derive from <uniprot>_p2rank_top1_contextN"
            )
        contexts.add(match.group("context"))
        exact[stem] = uniprot
        prefixes.setdefault(stem + "_", []).append(uniprot)
    if len(contexts) != 1:
        raise SystemExit("fragment index contains multiple contextN values; Foldseek IDs would be ambiguous")
    return exact, prefixes


def _foldseek_id_to_uniprot(
    value: str,
    *,
    exact: dict[str, str],
    prefixes: dict[str, list[str]],
    label: str,
    line_number: int,
) -> str:
    if value in exact:
        return exact[value]
    matches: list[str] = []
    for prefix, uniprots in prefixes.items():
        if value.startswith(prefix):
            matches.extend(uniprots)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"{label} has ambiguous chain-suffixed Foldseek identifier at line {line_number}: {value}"
        )
    raise SystemExit(
        f"{label} contains unknown Foldseek identifier at line {line_number}: {value}"
    )


def _cluster_assignments(
    path: Path,
    *,
    label: str,
    exact: dict[str, str],
    prefixes: dict[str, list[str]],
) -> dict[str, str]:
    _require_file(path, label)
    assignments: dict[str, str] = {}
    seen_raw_members: set[str] = set()
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
                raise SystemExit(
                    f"{label} requires exactly two nonblank TSV columns at line {line_number}: {path}"
                )
            representative_raw, member_raw = (field.strip() for field in fields)
            if member_raw in seen_raw_members:
                raise SystemExit(f"{label} assigns raw member {member_raw!r} more than once")
            seen_raw_members.add(member_raw)
            representative = _foldseek_id_to_uniprot(
                representative_raw,
                exact=exact,
                prefixes=prefixes,
                label=label,
                line_number=line_number,
            )
            member = _foldseek_id_to_uniprot(
                member_raw,
                exact=exact,
                prefixes=prefixes,
                label=label,
                line_number=line_number,
            )
            if member in assignments:
                raise SystemExit(f"{label} assigns member {member!r} more than once")
            assignments[member] = representative
    missing = sorted(set(exact.values()) - set(assignments))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise SystemExit(
            f"{label} is incomplete; {len(missing)} accepted pocket fragments lack a cluster assignment: {preview}{suffix}"
        )
    return assignments


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _clean_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def build_map(args: argparse.Namespace) -> None:
    if args.expected_target_count < 1:
        raise SystemExit("--expected-target-count must be >= 1")
    if not args.foldseek_version.strip():
        raise SystemExit("--foldseek-version must be nonblank")
    commands = {
        "tm40": args.foldseek_command_tm40.strip(),
        "tm50": args.foldseek_command_tm50.strip(),
        "tm60": args.foldseek_command_tm60.strip(),
    }
    if any(not command for command in commands.values()):
        raise SystemExit("all --foldseek-command-tm* values must be nonblank")

    _clean_outputs(args.out_csv, args.out_manifest)
    try:
        targets, target_manifest = _read_targets(
            args.target_clusters,
            args.target_cluster_manifest,
            args.expected_target_count,
        )
        fragments, exclusions, fragment_manifest, fragment_root = _read_fragment_universe(
            args.fragment_index,
            args.fragment_exclusions,
            args.fragment_manifest,
            targets,
            args.target_clusters,
            args.target_cluster_manifest,
        )
        exact, prefixes = _expected_foldseek_ids(fragments)
        cluster_inputs = {
            "tm40": args.foldseek_tm40_tsv,
            "tm50": args.foldseek_tm50_tsv,
            "tm60": args.foldseek_tm60_tsv,
        }
        clusters = {
            key: _cluster_assignments(
                path,
                label=f"Foldseek {key.upper()} cluster TSV",
                exact=exact,
                prefixes=prefixes,
            )
            for key, path in cluster_inputs.items()
        }
        rows: list[dict[str, str]] = []
        for uniprot in sorted(targets):
            available = uniprot in fragments
            rows.append(
                {
                    "uniprot": uniprot,
                    "pocket_available": "true" if available else "false",
                    "exclusion_reason": "" if available else exclusions[uniprot],
                    "pocket_cluster_tm40": clusters["tm40"][uniprot] if available else "",
                    "pocket_cluster_tm50": clusters["tm50"][uniprot] if available else "",
                    "pocket_cluster_tm60": clusters["tm60"][uniprot] if available else "",
                }
            )
        _write_csv(args.out_csv, rows)
        output_sha = _sha256(args.out_csv)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "definition": (
                "Foldseek clusters over top-P2Rank pocket-context fragments only; "
                "this artifact does not measure whole-protein structural similarity."
            ),
            "scientific_scope": {
                "included": "top-P2Rank pocket-context fragments",
                "excluded": "whole-protein structure and non-top-ranked predicted pockets",
            },
            "parameters": {
                "engine": "Foldseek easy-cluster",
                "foldseek_version": args.foldseek_version.strip(),
                "main_tm_threshold": 0.5,
                "sensitivity_tm_thresholds": [0.4, 0.6],
                "cluster_mode": "connected-component",
                "commands": commands,
            },
            "inputs": {
                "target_clusters": {
                    "path": str(args.target_clusters.resolve()),
                    "sha256": _sha256(args.target_clusters),
                    "rows": len(targets),
                },
                "target_cluster_manifest": {
                    "path": str(args.target_cluster_manifest.resolve()),
                    "sha256": _sha256(args.target_cluster_manifest),
                    "schema_version": target_manifest["schema_version"],
                },
                "fragment_index": {
                    "path": str(args.fragment_index.resolve()),
                    "sha256": _sha256(args.fragment_index),
                    "rows": len(fragments),
                },
                "fragment_exclusions": {
                    "path": str(args.fragment_exclusions.resolve()),
                    "sha256": _sha256(args.fragment_exclusions),
                    "rows": len(exclusions),
                },
                "fragment_manifest": {
                    "path": str(args.fragment_manifest.resolve()),
                    "sha256": _sha256(args.fragment_manifest),
                    "schema_version": fragment_manifest["schema_version"],
                },
                "fragment_files": {
                    "root": str(fragment_root),
                    "tree_sha256": _tree_digest(fragment_root),
                    "count": len(fragments),
                },
                "foldseek_cluster_tsvs": {
                    key: {
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "rows": len(clusters[key]),
                    }
                    for key, path in cluster_inputs.items()
                },
            },
            "counts": {
                "targets": len(targets),
                "pocket_available": len(fragments),
                "pocket_unavailable": len(exclusions),
                "clusters": {
                    key: len(set(assignments.values()))
                    for key, assignments in clusters.items()
                },
            },
            "partition_contract": {
                "expected_target_count": args.expected_target_count,
                "accepted_plus_excluded_equals_targets": True,
                "foldseek_membership_exactly_covers_available_pockets": True,
                "unavailable_targets_have_blank_cluster_columns": True,
            },
            "artifact": {
                "path": str(args.out_csv.resolve()),
                "sha256": output_sha,
                "rows": len(rows),
            },
        }
        _write_json(args.out_manifest, payload)
        print(
            "[pocket-clusters] wrote "
            f"rows={len(rows)} available={len(fragments)} unavailable={len(exclusions)} "
            f"tm50_clusters={payload['counts']['clusters']['tm50']}"
        )
    except BaseException:
        _clean_outputs(args.out_csv, args.out_manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-index", required=True, type=Path)
    parser.add_argument("--fragment-exclusions", required=True, type=Path)
    parser.add_argument("--fragment-manifest", required=True, type=Path)
    parser.add_argument("--target-clusters", required=True, type=Path)
    parser.add_argument("--target-cluster-manifest", required=True, type=Path)
    parser.add_argument("--foldseek-tm40-tsv", required=True, type=Path)
    parser.add_argument("--foldseek-tm50-tsv", required=True, type=Path)
    parser.add_argument("--foldseek-tm60-tsv", required=True, type=Path)
    parser.add_argument("--foldseek-version", required=True)
    parser.add_argument("--foldseek-command-tm40", required=True)
    parser.add_argument("--foldseek-command-tm50", required=True)
    parser.add_argument("--foldseek-command-tm60", required=True)
    parser.add_argument("--expected-target-count", type=int, default=4595)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        build_map(parse_args(argv))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
