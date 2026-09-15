#!/usr/bin/env python3
"""Validate Stage 0 data artifacts before a SkinScout compound run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage0_source_checks import validate_stage0_source  # noqa: E402

PRESETS = {"stage0", "safety", "target-id", "report"}
MODES = {"comprehensive", "fast", "both"}


@dataclass(frozen=True)
class ArtifactCheck:
    label: str
    path: str
    kind: str
    status: str
    size_bytes: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ReferenceQualitySpec:
    required_cols: set[str]
    min_rows: int
    placeholder_markers: dict[str, set[str]]


REFERENCE_QUALITY: dict[str, ReferenceQualitySpec] = {
    "CosIng reference": ReferenceQualitySpec(
        required_cols={"inci_name", "smiles", "inchikey", "ecfp4", "scaffold_smiles"},
        min_rows=100,
        placeholder_markers={
            "inci_name": {"Niacinamide", "Retinol", "Glycerin"},
        },
    ),
    "drug reference": ReferenceQualitySpec(
        required_cols={"drug_id", "name", "smiles", "inchikey", "ecfp4"},
        min_rows=100,
        placeholder_markers={
            "drug_id": {"PLACEHOLDER", "PLACEHOLDER1", "PLACEHOLDER2"},
            "name": {"no_drug_db_yet", "Aspirin", "Ibuprofen"},
        },
    ),
    "drug scaffold reference": ReferenceQualitySpec(
        required_cols={"scaffold_smiles", "n_drugs"},
        min_rows=20,
        placeholder_markers={},
    ),
    "skin-expression score table": ReferenceQualitySpec(
        required_cols={"uniprot", "skin_score"},
        min_rows=1000,
        placeholder_markers={},
    ),
}
MIN_SKIN_KG_GENE_EDGES = 50
HARD_PLACEHOLDER_TOKENS = ("PLACEHOLDER", "no_drug_db_yet")
STAGE0_SOURCE_PREFIX = "Stage 0 source:"
EMPTY_OK_FILE_ARTIFACTS = {
    "cleaned AlphaFold receptor marker",
    "PDBQT receptor marker",
}


def _workflow_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or ROOT / "workflow/config.yaml"
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"Workflow config must be a YAML object: {path}")
    return payload


def _config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Config boolean must be true/false-like, got {value!r}")


def _path_from_config(
    config: Mapping[str, Any],
    section: str,
    key: str,
    *,
    root: Path = ROOT,
) -> Path:
    value = config[section][key]  # type: ignore[index]
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path


def _requires_drugbank_source(stage0_cfg: Mapping[str, Any]) -> bool:
    return _config_bool(stage0_cfg.get("require_drugbank_source"), False)


def _requires_skin_proteome_source(skin_cfg: Mapping[str, Any]) -> bool:
    return _config_bool(skin_cfg.get("require_proteome_source"), False)


def _requires_gtex_source(skin_cfg: Mapping[str, Any]) -> bool:
    return _config_bool(skin_cfg.get("require_gtex_source"), False)


def _cosing_auto_mirror_enabled(cosing_cfg: Mapping[str, Any]) -> bool:
    return _config_bool(cosing_cfg.get("auto_mirror_public_api"), True)


def _has_config_path(
    workflow_config: Mapping[str, Any],
    section: str,
    key: str,
) -> bool:
    section_cfg = workflow_config.get(section, {})
    return isinstance(section_cfg, Mapping) and key in section_cfg


def _cosing_source_artifact(
    workflow_config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> tuple[str, Path]:
    cosing = _path_from_config(workflow_config, "paths_v3", "cosing", root=root)
    return (f"{STAGE0_SOURCE_PREFIX} CosIng CSV", cosing / "cosing.csv")


def _drugbank_source_artifact(
    workflow_config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> tuple[str, Path]:
    drugbank = _path_from_config(workflow_config, "paths", "drugbank", root=root)
    return (
        f"{STAGE0_SOURCE_PREFIX} DrugBank full database XML",
        drugbank / "drugbank_full_database.xml",
    )


def _skin_proteome_source_artifact(
    workflow_config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> tuple[str, Path]:
    proteome = _path_from_config(
        workflow_config,
        "paths_v3",
        "skin_proteome",
        root=root,
    )
    return (
        f"{STAGE0_SOURCE_PREFIX} skin proteome LFQ TSV",
        proteome / "raw_lfq.tsv",
    )


def _gtex_source_artifact(
    workflow_config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> tuple[str, Path]:
    gtex = _path_from_config(workflow_config, "paths_v3", "gtex", root=root)
    return (
        f"{STAGE0_SOURCE_PREFIX} GTEx gene TPM GCT",
        gtex / "gtex_v10_gene_tpm.gct",
    )


def required_stage0_source_artifacts(
    *,
    config: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> list[tuple[str, Path]]:
    workflow_config = config or _workflow_config()
    required: list[tuple[str, Path]] = []

    cosmetic_mode = str(workflow_config.get("cosmetic_db_mode", "full")).strip()
    if cosmetic_mode == "full":
        cosing_cfg = workflow_config.get("cosing", {})
        if not isinstance(cosing_cfg, Mapping):
            raise ValueError("cosing config must be a mapping")
        if not _cosing_auto_mirror_enabled(cosing_cfg):
            required.append(_cosing_source_artifact(workflow_config, root=root))
    elif cosmetic_mode != "placeholder":
        raise ValueError(
            "cosmetic_db_mode must be 'full' or 'placeholder', "
            f"got {cosmetic_mode!r}"
        )

    stage0_cfg = workflow_config.get("stage0", {})
    if not isinstance(stage0_cfg, Mapping):
        raise ValueError("stage0 config must be a mapping")
    if _requires_drugbank_source(stage0_cfg) and not _config_bool(
        stage0_cfg.get("allow_drugbank_placeholder"),
        False,
    ):
        required.append(_drugbank_source_artifact(workflow_config, root=root))

    skin_cfg = workflow_config.get("skin_expression", {})
    if not isinstance(skin_cfg, Mapping):
        raise ValueError("skin_expression config must be a mapping")
    if not _config_bool(skin_cfg.get("allow_empty_sources"), False):
        if _requires_skin_proteome_source(skin_cfg):
            required.append(_skin_proteome_source_artifact(workflow_config, root=root))
        if _requires_gtex_source(skin_cfg):
            required.append(_gtex_source_artifact(workflow_config, root=root))

    unique: dict[Path, tuple[str, Path]] = {}
    for label, path in required:
        unique.setdefault(path, (label, path))
    return list(unique.values())


def optional_stage0_source_artifacts(
    *,
    config: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> list[tuple[str, Path]]:
    workflow_config = config or _workflow_config()
    stage0_cfg = workflow_config.get("stage0", {})
    if not isinstance(stage0_cfg, Mapping):
        raise ValueError("stage0 config must be a mapping")
    optional: list[tuple[str, Path]] = []

    cosmetic_mode = str(workflow_config.get("cosmetic_db_mode", "full")).strip()
    if cosmetic_mode == "full":
        cosing_cfg = workflow_config.get("cosing", {})
        if not isinstance(cosing_cfg, Mapping):
            raise ValueError("cosing config must be a mapping")
        if _cosing_auto_mirror_enabled(cosing_cfg) and _has_config_path(
            workflow_config,
            "paths_v3",
            "cosing",
        ):
            optional.append(_cosing_source_artifact(workflow_config, root=root))
    elif cosmetic_mode != "placeholder":
        raise ValueError(
            "cosmetic_db_mode must be 'full' or 'placeholder', "
            f"got {cosmetic_mode!r}"
        )

    if not _requires_drugbank_source(stage0_cfg) and _has_config_path(
        workflow_config,
        "paths",
        "drugbank",
    ):
        optional.append(_drugbank_source_artifact(workflow_config, root=root))

    skin_cfg = workflow_config.get("skin_expression", {})
    if not isinstance(skin_cfg, Mapping):
        raise ValueError("skin_expression config must be a mapping")
    skin_sources_can_be_empty = _config_bool(
        skin_cfg.get("allow_empty_sources"),
        False,
    )
    if (
        (skin_sources_can_be_empty or not _requires_skin_proteome_source(skin_cfg))
        and _has_config_path(workflow_config, "paths_v3", "skin_proteome")
    ):
        optional.append(_skin_proteome_source_artifact(workflow_config, root=root))
    if (
        (skin_sources_can_be_empty or not _requires_gtex_source(skin_cfg))
        and _has_config_path(workflow_config, "paths_v3", "gtex")
    ):
        optional.append(_gtex_source_artifact(workflow_config, root=root))
    return optional


def required_data_artifacts(
    preset: str,
    mode: str = "comprehensive",
    *,
    config: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> list[tuple[str, Path]]:
    if preset not in PRESETS:
        raise ValueError(f"Unsupported preset: {preset}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    workflow_config = config or _workflow_config()
    if preset == "stage0":
        return required_stage0_source_artifacts(config=workflow_config, root=root)

    paths = {
        "cosing": _path_from_config(workflow_config, "paths_v3", "cosing", root=root),
        "drug_avoidance": _path_from_config(
            workflow_config,
            "paths_v3",
            "drug_avoidance",
            root=root,
        ),
        "skin_expression": _path_from_config(
            workflow_config,
            "paths_v3",
            "skin_expression",
            root=root,
        ),
        "skin_kg": _path_from_config(workflow_config, "paths_v3", "skin_kg", root=root),
        "alphafold_clean": _path_from_config(
            workflow_config,
            "paths",
            "alphafold_clean",
            root=root,
        ),
        "docking_boxes": _path_from_config(
            workflow_config,
            "paths",
            "docking_boxes",
            root=root,
        ),
        "pdbqt": _path_from_config(workflow_config, "paths", "pdbqt", root=root),
        "no_pocket_list": _path_from_config(
            workflow_config,
            "paths",
            "no_pocket_list",
            root=root,
        ),
    }
    required = [
        ("CosIng reference", paths["cosing"] / "cosing.parquet"),
        ("drug reference", paths["drug_avoidance"] / "drugs.parquet"),
        ("drug scaffold reference", paths["drug_avoidance"] / "scaffolds.parquet"),
    ]
    if preset in {"target-id", "report"}:
        required.extend(
            [
                (
                    "cleaned AlphaFold receptor marker",
                    paths["alphafold_clean"] / ".clean_complete",
                ),
                ("PDBQT receptor marker", paths["pdbqt"] / ".pdbqt_complete"),
                ("docking box directory", paths["docking_boxes"]),
                ("no-pocket target list", paths["no_pocket_list"]),
                (
                    "skin-expression score table",
                    paths["skin_expression"] / "skin_score.tsv",
                ),
                ("skin efficacy KG", paths["skin_kg"] / "skin_efficacy.graphml"),
            ]
        )

    unique: dict[Path, tuple[str, Path]] = {}
    for label, path in required:
        unique.setdefault(path, (label, path))
    return list(unique.values())


def check_artifact(label: str, path: Path) -> ArtifactCheck:
    if not path.exists():
        return ArtifactCheck(label, str(path), "missing", "missing")
    if path.is_file():
        size = path.stat().st_size
        return ArtifactCheck(
            label,
            str(path),
            "file",
            "present" if size > 0 else "empty",
            size,
        )
    if path.is_dir():
        try:
            has_child = any(path.iterdir())
        except OSError:
            return ArtifactCheck(label, str(path), "directory", "unreadable")
        return ArtifactCheck(
            label,
            str(path),
            "directory",
            "present" if has_child else "empty",
        )
    return ArtifactCheck(label, str(path), "other", "present")


def _table_quality(label: str, path: Path, spec: ReferenceQualitySpec) -> ArtifactCheck:
    base = check_artifact(label, path)
    if base.status != "present":
        return base
    try:
        import pandas as pd

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    except Exception as exc:  # noqa: BLE001
        return ArtifactCheck(
            label,
            str(path),
            base.kind,
            "invalid",
            base.size_bytes,
            f"failed to parse: {exc}",
        )

    missing_cols = sorted(spec.required_cols - set(df.columns))
    markers: list[str] = []
    for col, values in spec.placeholder_markers.items():
        if col not in df.columns:
            continue
        observed = set(df[col].astype(str))
        hits = sorted(observed & values)
        markers.extend(f"{col}={hit}" for hit in hits[:5])
    hard_markers = [
        marker
        for marker in markers
        if any(token in marker for token in HARD_PLACEHOLDER_TOKENS)
    ]
    detail = (
        f"rows={len(df)} min_rows={spec.min_rows} "
        f"missing_cols={missing_cols or 'none'} "
        f"marker_hits={markers or 'none'}"
    )
    status = "present"
    if missing_cols:
        status = "invalid"
    elif hard_markers:
        status = "placeholder"
    elif len(df) < spec.min_rows:
        status = "placeholder" if markers else "too_small"
    return ArtifactCheck(label, str(path), base.kind, status, base.size_bytes, detail)


def _skin_kg_quality(label: str, path: Path) -> ArtifactCheck:
    base = check_artifact(label, path)
    if base.status != "present":
        return base
    try:
        import networkx as nx

        graph = nx.read_graphml(path)
    except Exception as exc:  # noqa: BLE001
        return ArtifactCheck(
            label,
            str(path),
            base.kind,
            "invalid",
            base.size_bytes,
            f"failed to parse: {exc}",
        )

    gene_edges = 0
    pubtator_backed_nodes = 0
    for src, _dst, _attrs in graph.edges(data=True):
        if str(src).startswith("gene:"):
            gene_edges += 1
    for _node, attrs in graph.nodes(data=True):
        if attrs.get("pmid_count") or attrs.get("sample_pmids"):
            pubtator_backed_nodes += 1
    detail = (
        f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
        f"gene_edges={gene_edges} min_gene_edges={MIN_SKIN_KG_GENE_EDGES} "
        f"pubtator_backed_nodes={pubtator_backed_nodes}"
    )
    status = (
        "present"
        if gene_edges >= MIN_SKIN_KG_GENE_EDGES and pubtator_backed_nodes > 0
        else "placeholder"
    )
    return ArtifactCheck(label, str(path), base.kind, status, base.size_bytes, detail)


def check_required_artifact(label: str, path: Path) -> ArtifactCheck:
    if label.startswith(STAGE0_SOURCE_PREFIX):
        base = check_artifact(label, path)
        if base.status != "present":
            return base
        validation = validate_stage0_source(label, path)
        if not validation.ok:
            return ArtifactCheck(
                label,
                str(path),
                base.kind,
                "invalid",
                base.size_bytes,
                validation.detail,
            )
        return ArtifactCheck(
            label,
            str(path),
            base.kind,
            "present",
            base.size_bytes,
            validation.detail,
        )
    if label in EMPTY_OK_FILE_ARTIFACTS:
        if not path.exists():
            return ArtifactCheck(label, str(path), "missing", "missing")
        if path.is_file():
            return ArtifactCheck(
                label,
                str(path),
                "file",
                "present",
                path.stat().st_size,
            )
        return ArtifactCheck(
            label,
            str(path),
            "invalid",
            "invalid",
            detail="marker artifact must be a file",
        )
    if label in REFERENCE_QUALITY:
        return _table_quality(label, path, REFERENCE_QUALITY[label])
    if label == "skin efficacy KG":
        return _skin_kg_quality(label, path)
    return check_artifact(label, path)


def artifact_missing(path: Path) -> bool:
    return check_artifact("", path).status != "present"


def failed_check_entry(check: ArtifactCheck | Mapping[str, Any]) -> tuple[str, Path]:
    if isinstance(check, ArtifactCheck):
        label = check.label
        path = Path(check.path)
        status = check.status
        detail = check.detail
    else:
        label = str(check.get("label", ""))
        path = Path(str(check.get("path", "")))
        status = str(check.get("status", ""))
        detail = check.get("detail")
    annotated = f"{label} [{status}]"
    if detail:
        annotated += f" {detail}"
    return annotated, path


def missing_contains_stage0_sources(missing: list[tuple[str, Path]]) -> bool:
    return any(label.startswith(STAGE0_SOURCE_PREFIX) for label, _path in missing)


def _failed_entry_status(label: str) -> str:
    if "[" not in label or "]" not in label:
        return ""
    return label.split("[", 1)[1].split("]", 1)[0]


def failed_entries_are_stage0_buildable(missing: list[tuple[str, Path]]) -> bool:
    return bool(missing) and all(
        _failed_entry_status(label) == "missing"
        and not label.startswith(STAGE0_SOURCE_PREFIX)
        for label, _path in missing
    )


def stage0_source_failures_for_build(
    *,
    config: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> list[tuple[str, Path]]:
    checks = [
        check_required_artifact(label, path)
        for label, path in required_stage0_source_artifacts(config=config, root=root)
    ]
    return [failed_check_entry(check) for check in checks if check.status != "present"]


def data_readiness_payload(
    preset: str,
    mode: str,
    run_dti_sanity: bool,
    *,
    config: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    checks = [
        check_required_artifact(label, path)
        for label, path in required_data_artifacts(
            preset,
            mode,
            config=config,
            root=root,
        )
    ]
    failed = [check for check in checks if check.status != "present"]
    return {
        "status": "ok" if not failed else "failed",
        "preset": preset,
        "mode": mode,
        "run_dti_sanity": run_dti_sanity,
        "checks": [asdict(check) for check in checks],
    }


def missing_from_payload(payload: Mapping[str, Any]) -> list[tuple[str, Path]]:
    missing = []
    for check in payload.get("checks", []):
        if not isinstance(check, Mapping) or check.get("status") == "present":
            continue
        missing.append(failed_check_entry(check))
    return missing


def failure_message(missing: list[tuple[str, Path]]) -> str:
    if missing_contains_stage0_sources(missing):
        lines = [
            "Stage 0 source readiness preflight failed; required source inputs "
            "are missing or invalid:",
            *[f"  - {label}: {path}" for label, path in missing],
            "",
            "Import them with:",
            "  python scripts/import_stage0_sources.py --source-dir /path/to/stage0_sources",
            "",
            "Or place the source files at the listed paths. Switch the relevant "
            "workflow/config.yaml setting to an explicit placeholder/degraded "
            "mode only for diagnostics.",
        ]
        return "\n".join(lines)

    lines = [
        "Data readiness preflight failed; missing or invalid Stage 0 artifacts:",
        *[f"  - {label}: {path}" for label, path in missing],
        "",
        "Run the infrastructure build first, for example:",
        "  python scripts/run_skinscout.py --preset stage0 --run-id stage0_bootstrap",
        "",
    ]
    if failed_entries_are_stage0_buildable(missing):
        lines.extend(
            [
                "To let this run build missing Stage 0 artifacts automatically, add:",
                "  --allow-stage0-build",
            ]
        )
    else:
        lines.extend(
            [
                "--allow-stage0-build only bypasses missing generated outputs; "
                "replace or rebuild invalid, empty, placeholder, or too-small "
                "artifacts before running.",
            ]
        )
    return "\n".join(lines)


def allow_stage0_build_message(missing: list[tuple[str, Path]]) -> str:
    lines = [
        "Data readiness preflight found missing Stage 0 artifacts, but "
        "--allow-stage0-build was set:",
        *[f"  - {label}: {path}" for label, path in missing],
        "",
        "Snakemake is allowed to generate these missing artifacts during this run.",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="target-id")
    parser.add_argument("--mode", choices=sorted(MODES), default="comprehensive")
    parser.add_argument("--no-dti-sanity", action="store_true")
    parser.add_argument(
        "--allow-stage0-build",
        action="store_true",
        help="Exit zero even when compound-run Stage 0 artifacts are missing.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = data_readiness_payload(
        args.preset,
        args.mode,
        not args.no_dti_sanity,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(args.json_out)

    missing = missing_from_payload(payload)
    ok = not missing
    if args.json:
        sys.stdout.write(text)
    elif ok:
        print(
            "Data readiness preflight passed "
            f"for preset={args.preset} mode={args.mode}"
        )
    elif args.allow_stage0_build and failed_entries_are_stage0_buildable(missing):
        source_missing = stage0_source_failures_for_build()
        if source_missing:
            print(failure_message(source_missing), file=sys.stderr)
        else:
            print(allow_stage0_build_message(missing), file=sys.stderr)
    else:
        print(failure_message(missing), file=sys.stderr)

    if ok:
        return 0
    if args.allow_stage0_build and failed_entries_are_stage0_buildable(missing):
        return 0 if not stage0_source_failures_for_build() else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
