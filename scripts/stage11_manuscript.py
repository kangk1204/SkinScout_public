#!/usr/bin/env python3
"""stage11_manuscript.py — Emit a 6-file Markdown manuscript skeleton.

INSTRUCTIONS.md §17.4 layout:
    00_abstract.md / 01_introduction.md / 02_methods.md /
    03_results.md / 04_discussion.md / 05_references.bib
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

LOG = logging.getLogger("stage11.manuscript")

CAPTION_SOURCE_METRICS = {
    "bytes",
    "claim-ready eval steps",
    "html bytes",
    "items",
    "keys",
    "rows",
}
REPRO_METADATA_FILES = {
    "config_hash.txt",
    "git_commit.txt",
    "random_seeds.json",
    "runtime_log.txt",
    "tool_versions.lock",
}

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)

ABSTRACT = """\
# Abstract

(Auto-generated placeholder — fill in compound, target landscape, top-3 analog
properties, key findings.)
"""

INTRODUCTION = """\
# Introduction

Cosmetic R&D historically relies on serendipity, anecdotal evidence and slow
in-vivo iteration. This study introduces a license-aware computational pipeline
that combines proteome-wide target scoring, affinity-aware co-folding
(Boltz-2), ensemble sampling (BioEmu), and pose-supported interaction atom
analysis in a reproducible workflow for in-silico cosmetic ingredient
hypothesis generation. Planned REINVENT and retrosynthesis stages remain
behind explicit integration gates. Claims are limited to computational
evidence that passes fail-closed source and readiness checks.

(Auto-generated boilerplate — extend with motivation, novelty, contributions.)
"""

METHODS = """\
# Methods

## Stage 0 — Infrastructure

We mirrored the AlphaFold human proteome v4 (CC-BY-4.0), trimmed by pLDDT,
detected pockets with P2Rank, and converted to PDBQT via Meeko. We further
ingested ChEMBL37, BindingDB, CosIng, FDA Orange Book and the Human Protein
Atlas v25, with optional DrugBank licensed enrichment when available. We
computed a composite SkinScore per UniProt and built a NetworkX-based
skin-efficacy knowledge graph from PubTator 3.0.

## Stage 1-2 — Compound Preprocessing & ADMET Gate

Compounds are standardized (RDKit), protonated at physiological pH
(Dimorphite-DL), conformer-enumerated (ETKDGv3) and refined with xTB/GFN2.
A three-model skin-sensitization consensus (HuSSPred, STopTox, Pred-Skin)
gates the run.

## Stage 2.5 — Cosmetic Annotation & Drug Avoidance

The input ligand is compared by ECFP4 Tanimoto against CosIng INCI database
(EXACT / SIMILAR ≥ 0.85 / ANALOG ≥ 0.65 / NEW) and approved-drug DB
(STRICT / SCAFFOLD / SOFT warnings). The configurable `--drug-policy` decides
HALT / DOWNWEIGHT / PROCEED.

## Stage 3 — Dual-Mode Target Identification with Skin Weighting

We run MODE-COMPREHENSIVE through the AutoDock-GPU claim path only when every
selected receptor has precomputed AutoGrid4 map/field coverage (`.maps.fld` or
`.fld`). This repository checks that coverage but does not generate the maps.
MODE-FAST uses PSICHIC plus ligand similarity for throughput. Explicit Vina
docking is degraded diagnostic output and cannot support target/report claims.
A skin-expression weight `final = 0.7 × dock + 0.3 × skin` re-ranks candidates;
skin_score < 0.05 triggers a 70 % penalty.

## Stage 5.5 — Pose-Supported Interaction Atoms

Valid executable evidence is PLIP + ProLIF agreement over ligand atoms in bound
Boltz complex poses (2 of 2). The resulting atom set is used conservatively as
pose-supported interaction evidence in the 0-based bound-complex ligand atom
order. It is not mapped back to the original xTB SDF, is not a SMARTS or
generator constraint, and is not causal pharmacophore validation or
experimental efficacy evidence.

## Stage 5.6 — REINVENT Integration Blocker

Claim-capable analog generation is intentionally fail-closed. SkinScout does
not yet ship the required REINVENT 4 custom scoring plugins, trained prior and
agent artifacts, upstream-compatible staged-learning configuration, or a
validated mapping from bound-complex ligand atom order to the generator
representation. Diagnostic mode emits only a blocked-status record and an
empty SMILES file, which the downstream funnel rejects.

## Stage 7 / 7.5 / 8 — MD Blocker, Retrosynthesis, QM

Stage 7 is intentionally fail-closed until scientifically valid complex
topology/system preparation is implemented: protein+ligand topology merge,
solvation, ions, minimization, and restrained equilibration. Retrosynthesis
uses AiZynthFinder MCTS with USPTO policy and Zinc commercial stock; QM
readiness uses xTB + CREST + GPU4PySCF DFT when configured.

## Stage 9 / 11 — Reporting & Reproducibility

A single Mol*-based HTML report aggregates per-target panels (A–I).
The reproducibility pack (Stage 11) archives tool versions, config SHA-256,
git commit, and random seeds.
"""

RESULTS = """\
# Results

(Auto-generated placeholder — figures referenced via captions.json.)

- See `figures/fig01_workflow.svg` for the pipeline schematic.
- See `figures/fig07_eval_bars.png` for the benchmark summary
  (PoseBusters / PLINDER / cold-start / DTI-vs-docking disagreement).
- See `figures/fig08_case_study.png` for known-cosmetic recovery.
"""

DISCUSSION = """\
# Discussion

(Auto-generated placeholder — interpret findings, limitations, future work.)
"""

REFERENCES = """\
% Auto-generated by stage11_manuscript.py — extend as needed.
@article{Passaro2025Boltz2,
  title  = {Boltz-2: Co-folding and affinity prediction},
  author = {Passaro, et al.},
  year   = {2025},
  journal = {Recursion Tech Report},
}
@article{Lewis2025BioEmu,
  title  = {BioEmu: Equilibrium ensemble emulation},
  author = {Lewis, et al.},
  year   = {2025},
  journal = {Science},
  volume = {389},
  pages  = {eadv9817},
}
@article{Loeffler2024Reinvent4,
  title  = {REINVENT 4: Modern AI-driven design},
  author = {Loeffler, et al.},
  year   = {2024},
  journal = {J. Cheminform.},
  volume = {16},
  pages  = {20},
}
@article{Genheden2020Aizynth,
  title  = {AiZynthFinder: a fast, robust and flexible open-source software
            for retrosynthetic planning},
  author = {Genheden, et al.},
  year   = {2020},
  journal = {J. Cheminform.},
  volume = {12},
  pages  = {70},
}
"""

FILES = {
    "00_abstract.md":    ABSTRACT,
    "01_introduction.md": INTRODUCTION,
    "02_methods.md":     METHODS,
    "03_results.md":     RESULTS,
    "04_discussion.md":  DISCUSSION,
    "05_references.bib": REFERENCES,
}


def _reject_scaffold_markers(text: str, label: str) -> None:
    lowered = text.lower()
    for marker in ("placeholder", "stale", "xxxx", "{user}"):
        if marker in lowered:
            raise SystemExit(
                f"Manuscript source {label} contains scaffold marker "
                f"{marker!r}"
            )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Manuscript source {label} is required: {path}")
    try:
        text = path.read_text()
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            f"Manuscript source {label} failed to parse as JSON: {path}: {exc}"
        ) from exc
    _reject_scaffold_markers(text, label)
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f"Manuscript source {label} must be a non-empty object: {path}")
    return payload


def _validate_claim_ready_eval_manifest(path: Path, label: str) -> None:
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SystemExit(
            f"Manuscript source {label} failed to parse as JSON: {path}: {exc}"
        ) from exc
    validator_path = Path(__file__).with_name("stage11_make_figures.py")
    spec = importlib.util.spec_from_file_location(
        "stage11_make_figures_for_manuscript",
        validator_path,
    )
    if spec is None or spec.loader is None:
        raise SystemExit(
            "Manuscript could not load eval manifest validator: "
            f"{validator_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module._validate_iteration_manifest(doc, path)
    except ValueError as exc:
        raise SystemExit(
            "Manuscript reproducibility manifest eval artifact is not "
            f"claim-ready: {exc}"
        ) from exc


def _validate_captions(path: Path) -> dict[str, Any]:
    captions = _read_json_object(path, "captions")
    for name, entry in captions.items():
        if not isinstance(entry, dict):
            raise SystemExit(
                f"Manuscript captions entry must be an object: {name}: {path}"
            )
        caption = str(entry.get("caption", "")).strip()
        if not caption:
            raise SystemExit(f"Manuscript captions entry is missing text: {name}: {path}")
        if entry.get("status") != "source_backed":
            raise SystemExit(
                "Manuscript captions must be source-backed for claim-ready "
                f"drafts: {name}: {path}"
            )
        sources = entry.get("sources")
        if not isinstance(sources, list) or not sources:
            raise SystemExit(
                f"Manuscript captions entry has no sources: {name}: {path}"
            )
        seen_sources: set[tuple[str, str, str]] = set()
        for idx, source in enumerate(sources):
            if not isinstance(source, dict):
                raise SystemExit(
                    "Manuscript captions sources must contain objects: "
                    f"{name}: source index {idx}: {path}"
                )
            source_path = str(source.get("path", "")).strip()
            if not source_path:
                raise SystemExit(
                    "Manuscript captions source path must be non-empty: "
                    f"{name}: source index {idx}: {path}"
                )
            metric = str(source.get("metric", "")).strip()
            if metric not in CAPTION_SOURCE_METRICS:
                raise SystemExit(
                    "Manuscript captions source metric is not recognized: "
                    f"{name}: source index {idx}: {path}"
                )
            value = source.get("value")
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SystemExit(
                    "Manuscript captions source value must be a positive "
                    f"integer: {name}: source index {idx}: {path}"
                )
            sha = str(source.get("sha256", "")).strip()
            if len(sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in sha):
                raise SystemExit(
                    "Manuscript captions source has invalid sha256: "
                    f"{name}: source index {idx}: {path}"
                )
            source_key = (Path(source_path).as_posix(), metric, sha.lower())
            if source_key in seen_sources:
                raise SystemExit(
                    "Manuscript captions entry contains duplicate source: "
                    f"{name}: source index {idx}: {path}"
                )
            seen_sources.add(source_key)
    return captions


def _validate_repro_manifest(path: Path, expected_run_dir: Path) -> dict[str, Any]:
    manifest = _read_json_object(path, "reproducibility manifest")
    run_dir = str(manifest.get("run_dir", "")).strip()
    if not run_dir:
        raise SystemExit(
            "Manuscript reproducibility manifest run_dir must be non-empty: "
            f"{path}"
        )
    if Path(run_dir).resolve() != expected_run_dir.resolve():
        raise SystemExit(
            "Manuscript reproducibility manifest run_dir must match "
            f"--run-dir: {path}"
        )
    artifacts = manifest.get("artifacts")
    n_artifacts = manifest.get("n_artifacts")
    if not isinstance(n_artifacts, int) or isinstance(n_artifacts, bool) or n_artifacts <= 0:
        raise SystemExit(
            "Manuscript reproducibility manifest n_artifacts must be a "
            f"positive integer: {path}"
        )
    if not isinstance(artifacts, list) or len(artifacts) != n_artifacts:
        raise SystemExit(
            "Manuscript reproducibility manifest artifacts length must match "
            f"n_artifacts: {path}"
        )
    relative_paths: set[str] = set()
    for idx, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise SystemExit(
                "Manuscript reproducibility manifest artifacts must contain "
                f"objects: row index {idx}: {path}"
            )
        rel = str(artifact.get("relative_path", "")).strip()
        sha = str(artifact.get("sha256", "")).strip()
        size = artifact.get("bytes")
        artifact_path = str(artifact.get("path", "")).strip()
        if not rel:
            raise SystemExit(
                "Manuscript reproducibility manifest artifact missing "
                f"relative_path: row index {idx}: {path}"
            )
        if rel in relative_paths:
            raise SystemExit(
                "Manuscript reproducibility manifest contains duplicate "
                f"relative_path {rel!r}: row index {idx}: {path}"
            )
        if len(sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in sha):
            raise SystemExit(
                "Manuscript reproducibility manifest artifact has invalid "
                f"sha256: row index {idx}: {path}"
            )
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise SystemExit(
                "Manuscript reproducibility manifest artifact bytes must be "
                f"positive: row index {idx}: {path}"
            )
        if artifact_path and not rel.startswith("EVAL:"):
            expected_path = (Path(run_dir) / rel).resolve()
            if Path(artifact_path).resolve() != expected_path:
                raise SystemExit(
                    "Manuscript reproducibility manifest artifact path must "
                    "match relative_path under run_dir: "
                    f"row index {idx}: {path}"
                )
        if artifact_path:
            actual_path = Path(artifact_path)
            if not actual_path.exists() or not actual_path.is_file():
                raise SystemExit(
                    "Manuscript reproducibility manifest artifact path must "
                    f"exist: row index {idx}: {path}"
                )
            actual_bytes = actual_path.stat().st_size
            actual_sha = hashlib.sha256(actual_path.read_bytes()).hexdigest()
            if actual_bytes != size or actual_sha != sha.lower():
                raise SystemExit(
                    "Manuscript reproducibility manifest artifact bytes/sha256 "
                    f"must match existing file: row index {idx}: {path}"
                )
            if rel == "EVAL:iteration_manifest.json":
                _validate_claim_ready_eval_manifest(
                    actual_path,
                    "eval manifest artifact",
                )
        relative_paths.add(rel)
    if "EVAL:iteration_manifest.json" not in relative_paths:
        raise SystemExit(
            "Manuscript reproducibility manifest must include "
            f"EVAL:iteration_manifest.json: {path}"
        )
    if "metadata_files" not in manifest:
        raise SystemExit(
            "Manuscript reproducibility manifest metadata_files is required: "
            f"{path}"
        )
    _validate_repro_metadata_files(manifest["metadata_files"], path)
    return manifest


def _validate_repro_metadata_files(metadata_files: Any, manifest_path: Path) -> None:
    if not isinstance(metadata_files, list) or not metadata_files:
        raise SystemExit(
            "Manuscript reproducibility manifest metadata_files must be a "
            f"non-empty list: {manifest_path}"
        )
    seen: set[str] = set()
    for idx, record in enumerate(metadata_files):
        if not isinstance(record, dict):
            raise SystemExit(
                "Manuscript reproducibility manifest metadata_files must "
                f"contain objects: row index {idx}: {manifest_path}"
            )
        rel = str(record.get("relative_path", "")).strip()
        sha = str(record.get("sha256", "")).strip()
        size = record.get("bytes")
        rel_path = Path(rel)
        if not rel or rel_path.is_absolute() or ".." in rel_path.parts:
            raise SystemExit(
                "Manuscript reproducibility manifest metadata file has unsafe "
                f"relative_path: row index {idx}: {manifest_path}"
            )
        if rel in seen:
            raise SystemExit(
                "Manuscript reproducibility manifest metadata_files contains "
                f"duplicate relative_path {rel!r}: row index {idx}: "
                f"{manifest_path}"
            )
        if len(sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in sha):
            raise SystemExit(
                "Manuscript reproducibility manifest metadata file has "
                f"invalid sha256: row index {idx}: {manifest_path}"
            )
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise SystemExit(
                "Manuscript reproducibility manifest metadata file bytes "
                f"must be positive: row index {idx}: {manifest_path}"
            )
        actual_path = manifest_path.parent / rel_path
        if not actual_path.exists() or not actual_path.is_file():
            raise SystemExit(
                "Manuscript reproducibility manifest metadata file must "
                f"exist: row index {idx}: {manifest_path}"
            )
        actual_bytes = actual_path.stat().st_size
        actual_sha = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        if actual_bytes != size or actual_sha != sha.lower():
            raise SystemExit(
                "Manuscript reproducibility manifest metadata file "
                f"bytes/sha256 must match existing file: row index {idx}: "
                f"{manifest_path}"
            )
        seen.add(rel)
    missing = sorted(REPRO_METADATA_FILES.difference(seen))
    unexpected = sorted(seen.difference(REPRO_METADATA_FILES))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise SystemExit(
            "Manuscript reproducibility manifest metadata_files must match "
            "the required reproducibility sidecars "
            f"({'; '.join(details)}): {manifest_path}"
        )


def _validate_data_availability(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Manuscript data availability source is required: {path}")
    text = path.read_text()
    _reject_scaffold_markers(text, "data availability")
    if "# Data Availability" not in text:
        raise SystemExit(
            f"Manuscript data availability source missing heading: {path}"
        )
    if "DOI: pending" in text:
        raise SystemExit(
            f"Manuscript data availability source contains pending DOI: {path}"
        )
    doi_lines = [line.strip() for line in text.splitlines() if "DOI:" in line]
    if len(doi_lines) != 1:
        raise SystemExit(
            "Manuscript data availability source must contain exactly one DOI "
            f"line: {path}"
        )
    raw_doi = doi_lines[0].split("DOI:", 1)[1].strip().strip("`")
    doi = raw_doi.split(maxsplit=1)[0].strip("`),.;")
    if not DOI_RE.fullmatch(doi) or "XXXX" in doi.upper():
        raise SystemExit(
            f"Manuscript data availability source contains invalid DOI: {path}"
        )
    return text


def _caption_source_matches_artifact(
    source_path: str,
    artifact: dict[str, Any],
    run_dir: str,
) -> bool:
    rel = str(artifact.get("relative_path", "")).strip()
    if not rel:
        return False
    artifact_path = str(artifact.get("path", "")).strip()
    if artifact_path and source_path == artifact_path:
        return True
    rel_suffix = rel.split(":", 1)[1] if rel.startswith("EVAL:") else rel
    source_posix = Path(source_path).as_posix()
    rel_posix = Path(rel_suffix).as_posix()
    accepted_paths = {rel_posix}
    run_name = Path(run_dir).name
    if run_name:
        accepted_paths.add(f"{run_name}/{rel_posix}")
    return source_posix in accepted_paths


def _source_backed_files(
    captions: dict[str, Any],
    repro_manifest: dict[str, Any],
    data_availability: str,
) -> dict[str, str]:
    run_dir = str(repro_manifest.get("run_dir", "")).strip()
    repro_artifacts_by_sha: dict[str, list[dict[str, Any]]] = {}
    for artifact in repro_manifest["artifacts"]:
        if not isinstance(artifact, dict):
            continue
        sha = str(artifact.get("sha256", "")).strip().lower()
        repro_artifacts_by_sha.setdefault(sha, []).append(artifact)
    for name, entry in captions.items():
        for idx, source in enumerate(entry["sources"]):
            sha = str(source.get("sha256", "")).strip().lower()
            matching_artifacts = repro_artifacts_by_sha.get(sha, [])
            if not matching_artifacts:
                raise SystemExit(
                    "Manuscript captions source sha256 is missing from "
                    "reproducibility manifest: "
                    f"{name}: source index {idx}"
                )
            source_path = str(source.get("path", "")).strip()
            path_matches = [
                artifact for artifact in matching_artifacts
                if _caption_source_matches_artifact(source_path, artifact, run_dir)
            ]
            if not path_matches:
                raise SystemExit(
                    "Manuscript captions source path does not match "
                    "reproducibility manifest artifact: "
                    f"{name}: source index {idx}"
                )
            metric = str(source.get("metric", "")).strip()
            if metric == "bytes" and not any(
                int(artifact["bytes"]) == source["value"] for artifact in path_matches
            ):
                raise SystemExit(
                    "Manuscript captions source bytes value does not match "
                    "reproducibility manifest artifact: "
                    f"{name}: source index {idx}"
                )
    figure_lines = [
        f"- `{name}`: {entry['caption']}"
        for name, entry in sorted(captions.items())
    ]
    n_artifacts = repro_manifest["n_artifacts"]
    data_line = next(
        (line.strip() for line in data_availability.splitlines() if "DOI:" in line),
        "Data availability DOI recorded in the accompanying statement.",
    )
    return {
        "00_abstract.md": (
            "# Abstract\n\n"
            "SkinScout integrates source-backed target ranking, safety gates, "
            "analog design, retrosynthesis, and evaluation evidence into a "
            "reproducible cosmetic discovery workflow. The Stage 11 package "
            f"links {len(captions)} figure panels to checked source artifacts "
            f"and archives {n_artifacts} reproducibility records.\n"
        ),
        "01_introduction.md": INTRODUCTION.replace(
            "(Auto-generated boilerplate — extend with motivation, novelty, contributions.)",
            "This manuscript is generated from verified Stage 11 artifacts so "
            "figures, data availability, and reproducibility evidence remain "
            "traceable to the current run.",
        ),
        "02_methods.md": METHODS + (
            "\n## Stage 11 Evidence Binding\n\n"
            "The manuscript generator requires source-backed figure captions, "
            "a reproducibility artifact manifest, and a data-availability "
            "statement with a configured DOI before writing claim-ready text.\n"
        ),
        "03_results.md": (
            "# Results\n\n"
            "The publication figure package reports the following source-backed "
            "panels:\n\n"
            + "\n".join(figure_lines)
            + "\n"
        ),
        "04_discussion.md": (
            "# Discussion\n\n"
            "The current package emphasizes traceability: every reported panel "
            "is tied to checked source artifacts, and reproducibility metadata "
            "includes the claim-ready evaluation manifest digest. Remaining "
            "scientific interpretation should be written against these locked "
            "artifacts.\n\n"
            f"{data_line}\n"
        ),
        "05_references.bib": REFERENCES.replace(
            "% Auto-generated by stage11_manuscript.py — extend as needed.",
            "% Generated by stage11_manuscript.py from source-backed Stage 11 artifacts.",
        ),
    }


def expected_outputs(out_dir: Path) -> list[Path]:
    return [out_dir / name for name in FILES]


def remove_outputs(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--captions", default=None, type=Path)
    parser.add_argument("--repro-manifest", default=None, type=Path)
    parser.add_argument("--data-availability", default=None, type=Path)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="write scaffold manuscript text for explicit draft diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    remove_outputs(expected_outputs(args.out_dir))

    source_args = [args.captions, args.repro_manifest, args.data_availability]
    if all(source_args):
        files = _source_backed_files(
            _validate_captions(args.captions),
            _validate_repro_manifest(args.repro_manifest, args.run_dir),
            _validate_data_availability(args.data_availability),
        )
    else:
        files = FILES

    bodies = "\n".join(files.values())
    if "placeholder" in bodies.lower() and not args.allow_placeholders:
        raise SystemExit(
            "Manuscript draft contains placeholder scaffold text; "
            "use --allow-placeholders only for explicit drafts"
        )

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stage11_manuscript_",
        dir=args.out_dir.parent,
    ) as tmp:
        stage_dir = Path(tmp)
        for name, body in files.items():
            (stage_dir / name).write_text(body)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            (stage_dir / name).replace(args.out_dir / name)
    LOG.info("Wrote 6 manuscript files → %s", args.out_dir)


if __name__ == "__main__":
    main()
