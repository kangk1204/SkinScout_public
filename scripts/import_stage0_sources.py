#!/usr/bin/env python3
"""Import operator-supplied Stage 0 source files into canonical data paths."""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_readiness import (  # noqa: E402
    optional_stage0_source_artifacts,
    required_stage0_source_artifacts,
)
from stage0_source_checks import (  # noqa: E402
    stage0_source_key,
    validate_stage0_source,
)


class SourceImportError(RuntimeError):
    """Raised when a Stage 0 source cannot be imported safely."""


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    candidates: tuple[str, ...]
    glob_patterns: tuple[str, ...]
    zip_suffixes: tuple[str, ...]


SOURCE_SPECS = {
    "cosing_csv": SourceSpec(
        key="cosing_csv",
        label="Stage 0 source: CosIng CSV",
        candidates=(
            "cosing.csv",
            "cosing/cosing.csv",
            "data/cosing/cosing.csv",
            "cosing.csv.zip",
            "cosing/cosing.csv.zip",
            "CosIng - Glossary of Ingredients.csv",
            "CosIng - Glossary of Ingredients.csv.zip",
        ),
        glob_patterns=(
            "*cosing*.csv",
            "*cosing*.csv.zip",
            "*glossary*ingredient*.csv",
            "*glossary*ingredient*.csv.zip",
        ),
        zip_suffixes=(".csv",),
    ),
    "drugbank_xml": SourceSpec(
        key="drugbank_xml",
        label="Stage 0 source: DrugBank full database XML",
        candidates=(
            "drugbank_full_database.xml",
            "full_database.xml",
            "drugbank/drugbank_full_database.xml",
            "data/drugbank/drugbank_full_database.xml",
            "drugbank_full_database.xml.zip",
            "full database.xml.zip",
            "drugbank_all_full_database.xml.zip",
            "drugbank/full database.xml.zip",
            "drugbank/drugbank_all_full_database.xml.zip",
        ),
        glob_patterns=(
            "*drugbank*full*database*.xml",
            "*drugbank*full*database*.xml.zip",
            "*full*database*.xml",
            "*full*database*.xml.zip",
        ),
        zip_suffixes=(".xml",),
    ),
    "skin_proteome_tsv": SourceSpec(
        key="skin_proteome_tsv",
        label="Stage 0 source: skin proteome LFQ TSV",
        candidates=(
            "raw_lfq.tsv",
            "raw_lfq.tsv.gz",
            "skin_proteome/raw_lfq.tsv",
            "skin_proteome/raw_lfq.tsv.gz",
            "data/skin_proteome/raw_lfq.tsv",
            "data/skin_proteome/raw_lfq.tsv.gz",
        ),
        glob_patterns=(
            "*raw*lfq*.tsv",
            "*raw*lfq*.tsv.gz",
            "*skin*proteome*lfq*.tsv",
            "*skin*proteome*lfq*.tsv.gz",
        ),
        zip_suffixes=(".tsv",),
    ),
    "gtex_gct": SourceSpec(
        key="gtex_gct",
        label="Stage 0 source: GTEx gene TPM GCT",
        candidates=(
            "gtex_v10_gene_tpm.gct",
            "gtex_v10_gene_tpm.gct.gz",
            "gtex/gtex_v10_gene_tpm.gct",
            "gtex_v10/gtex_v10_gene_tpm.gct",
            "data/gtex_v10/gtex_v10_gene_tpm.gct",
            "gtex/gtex_v10_gene_tpm.gct.gz",
            "gtex_v10/gtex_v10_gene_tpm.gct.gz",
            "data/gtex_v10/gtex_v10_gene_tpm.gct.gz",
        ),
        glob_patterns=(
            "*gtex*v10*gene*tpm*.gct",
            "*gtex*v10*gene*tpm*.gct.gz",
            "*gtex*skin*gene*tpm*.gct",
            "*gtex*skin*gene*tpm*.gct.gz",
        ),
        zip_suffixes=(".gct",),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_cli_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _find_source(
    spec: SourceSpec,
    source_dir: Path | None,
    explicit: Mapping[str, Path],
) -> Path | None:
    if spec.key in explicit:
        return explicit[spec.key]
    if source_dir is None:
        return None
    for rel in spec.candidates:
        candidate = source_dir / rel
        if candidate.exists():
            return candidate
    matches = _glob_source_matches(spec, source_dir)
    if len(matches) == 1:
        return matches[0]
    return None


def _glob_source_matches(spec: SourceSpec, source_dir: Path) -> list[Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        return []
    patterns = tuple(pattern.lower() for pattern in spec.glob_patterns)
    matches: dict[Path, None] = {}
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir).as_posix().lower()
        name = path.name.lower()
        if any(
            fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern)
            for pattern in patterns
        ):
            matches[path] = None
    return sorted(matches)


def _source_search_description(spec: SourceSpec) -> str:
    exact = ", ".join(spec.candidates)
    patterns = ", ".join(spec.glob_patterns)
    return f"exact names: {exact}; recursive patterns: {patterns}"


def _source_not_found_error(spec: SourceSpec, label: str, source_dir: Path | None) -> str:
    if source_dir is None:
        return (
            f"{label}: no source found; provide --source-dir or "
            f"--{spec.key.replace('_', '-')}"
        )
    if not source_dir.exists():
        return f"{label}: --source-dir does not exist: {source_dir}"
    if not source_dir.is_dir():
        return f"{label}: --source-dir is not a directory: {source_dir}"
    matches = _glob_source_matches(spec, source_dir)
    if len(matches) > 1:
        shown = ", ".join(str(path.relative_to(source_dir)) for path in matches[:6])
        suffix = "..." if len(matches) > 6 else ""
        return (
            f"{label}: multiple recursive source candidates found: {shown}{suffix}; "
            "pass an explicit source path"
        )
    return (
        f"{label}: no source found; searched --source-dir for "
        f"{_source_search_description(spec)}"
    )


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _gunzip_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with gzip.open(src, "rb") as in_handle, tmp.open("wb") as out_handle:
        shutil.copyfileobj(in_handle, out_handle)
    tmp.replace(dst)


def _zip_member(spec: SourceSpec, src: Path) -> str:
    with zipfile.ZipFile(src) as archive:
        names = [
            name for name in archive.namelist()
            if not name.endswith("/")
            and name.lower().endswith(spec.zip_suffixes)
        ]
        if not names:
            raise SourceImportError(
                f"{src} contains no member ending with {spec.zip_suffixes}"
            )
        if len(names) > 1:
            shown = ", ".join(names[:5])
            suffix = "..." if len(names) > 5 else ""
            raise SourceImportError(
                f"{src} contains multiple candidate members: {shown}{suffix}"
            )
        return names[0]


def _unzip_atomic(spec: SourceSpec, src: Path, dst: Path) -> None:
    member = _zip_member(spec, src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(src) as archive, archive.open(member) as in_handle:
        with tmp.open("wb") as out_handle:
            shutil.copyfileobj(in_handle, out_handle)
    tmp.replace(dst)


def _materialize_source(spec: SourceSpec, src: Path, tmp_target: Path) -> str:
    if src.suffix == ".gz":
        _gunzip_atomic(src, tmp_target)
        return "decompress"
    if src.suffix == ".zip":
        _unzip_atomic(spec, src, tmp_target)
        return "extract"
    _copy_atomic(src, tmp_target)
    return "copy"


def _stage_copy(spec: SourceSpec, src: Path, target: Path) -> tuple[Path, str]:
    tmp_target = target.with_suffix(target.suffix + ".stage0_import")
    tmp_target.unlink(missing_ok=True)
    action = _materialize_source(spec, src, tmp_target)
    return tmp_target, action


def _stage_symlink(spec: SourceSpec, src: Path, target: Path) -> tuple[Path, str]:
    if src.suffix in {".gz", ".zip"}:
        raise SourceImportError(
            f"--mode symlink cannot install compressed source {src}; use copy mode"
        )
    return src, "symlink"


def _install_one(
    spec: SourceSpec,
    label: str,
    src: Path,
    target: Path,
    *,
    mode: str,
    force: bool,
) -> dict[str, Any]:
    if not src.exists() or not src.is_file() or src.stat().st_size == 0:
        raise SourceImportError(f"source is missing, not a file, or empty: {src}")

    if mode == "copy":
        staged, action = _stage_copy(spec, src, target)
    elif mode == "symlink":
        staged, action = _stage_symlink(spec, src.resolve(), target)
    else:
        raise AssertionError(f"unsupported mode: {mode}")

    validation = validate_stage0_source(label, staged)
    if not validation.ok:
        if staged != src:
            staged.unlink(missing_ok=True)
        raise SourceImportError(validation.detail)

    staged_sha = _sha256(staged)
    staged_size = staged.stat().st_size
    status = "installed"
    if target.exists() or target.is_symlink():
        if target.is_file() and _sha256(target) == staged_sha:
            status = "already_present"
            if staged != src:
                staged.unlink(missing_ok=True)
        elif not force:
            if staged != src:
                staged.unlink(missing_ok=True)
            raise SourceImportError(
                f"target exists with different content: {target}; rerun with --force"
            )
        else:
            if target.is_dir():
                if staged != src:
                    staged.unlink(missing_ok=True)
                raise SourceImportError(f"target is a directory: {target}")
            target.unlink(missing_ok=True)

    if status == "installed":
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            target.unlink(missing_ok=True)
            target.symlink_to(src)
        else:
            staged.replace(target)

    return {
        "key": spec.key,
        "label": label,
        "status": status,
        "action": action,
        "source_path": str(src),
        "source_bytes": src.stat().st_size,
        "source_sha256": _sha256(src),
        "target_path": str(target),
        "target_bytes": staged_size,
        "target_sha256": staged_sha,
        "detail": validation.detail,
    }


def import_stage0_sources(
    *,
    required_sources: Iterable[tuple[str, Path]] | None = None,
    source_dir: Path | None = None,
    source_paths: Mapping[str, Path] | None = None,
    manifest_path: Path,
    mode: str = "copy",
    force: bool = False,
) -> dict[str, Any]:
    explicit = {
        key: path
        for key, path in (source_paths or {}).items()
        if path is not None
    }
    required = list(required_sources or required_stage0_source_artifacts())
    if required_sources is None:
        required.extend(_present_optional_sources(source_dir, explicit))
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for label, target in required:
        key = stage0_source_key(label)
        if key is None:
            errors.append(f"no importer registered for source label: {label}")
            continue
        spec = SOURCE_SPECS[key]
        src = _find_source(spec, source_dir, explicit)
        if src is None:
            errors.append(_source_not_found_error(spec, label, source_dir))
            continue
        try:
            results.append(
                _install_one(
                    spec,
                    label,
                    src,
                    target,
                    mode=mode,
                    force=force,
                )
            )
        except SourceImportError as exc:
            errors.append(f"{label}: {exc}")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not errors else "failed",
        "mode": mode,
        "source_dir": str(source_dir) if source_dir is not None else None,
        "required_source_count": len(required),
        "imported_source_count": len(results),
        "sources": results,
        "errors": errors,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(manifest_path)
    return payload


def _present_optional_sources(
    source_dir: Path | None,
    explicit: Mapping[str, Path],
) -> list[tuple[str, Path]]:
    present: list[tuple[str, Path]] = []
    for label, target in optional_stage0_source_artifacts():
        key = stage0_source_key(label)
        if key is None or key in {
            stage0_source_key(existing_label)
            for existing_label, _existing_target in present
        }:
            continue
        spec = SOURCE_SPECS[key]
        if _find_source(spec, source_dir, explicit) is not None:
            present.append((label, target))
    return present


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing flat or canonical-layout Stage 0 source files.",
    )
    parser.add_argument("--cosing-csv", type=Path)
    parser.add_argument("--drugbank-xml", type=Path)
    parser.add_argument("--skin-proteome-tsv", type=Path)
    parser.add_argument("--gtex-gct", type=Path)
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Install sources by copying/extracting or by symlinking uncompressed files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing target file when content differs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/stage0_source_manifest.json"),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON payload")
    return parser.parse_args(argv)


def _explicit_source_paths(args: argparse.Namespace) -> dict[str, Path]:
    values = {
        "cosing_csv": _resolve_cli_path(args.cosing_csv),
        "drugbank_xml": _resolve_cli_path(args.drugbank_xml),
        "skin_proteome_tsv": _resolve_cli_path(args.skin_proteome_tsv),
        "gtex_gct": _resolve_cli_path(args.gtex_gct),
    }
    return {key: path for key, path in values.items() if path is not None}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = _resolve_cli_path(args.source_dir)
    manifest_path = _resolve_repo_path(args.manifest)
    payload = import_stage0_sources(
        source_dir=source_dir,
        source_paths=_explicit_source_paths(args),
        manifest_path=manifest_path,
        mode=args.mode,
        force=args.force,
    )
    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif payload["status"] == "ok":
        print(
            "Imported Stage 0 source files: "
            f"{payload['imported_source_count']}/{payload['required_source_count']}"
        )
        print(f"Manifest: {manifest_path}")
    else:
        print("Stage 0 source import failed:", file=sys.stderr)
        for error in payload["errors"]:
            print(f"  - {error}", file=sys.stderr)
        print(f"Manifest: {manifest_path}", file=sys.stderr)
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
