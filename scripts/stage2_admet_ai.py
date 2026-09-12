#!/usr/bin/env python3
"""stage2_admet_ai.py — Chemprop-RDKit ADMET-AI (41 TDC tasks)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from rdkit import Chem

LOG = logging.getLogger("stage2.admet_ai")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def smiles_from_sdf(sdf: Path) -> str:
    sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for mol in sup:
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    raise SystemExit(f"No parseable mol in {sdf}")


def _is_missing_admet_ai(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", None)
    return missing_name == "admet_ai" or "No module named 'admet_ai'" in str(exc)


def _load_admet_model() -> type[Any] | None:
    try:
        from admet_ai import ADMETModel  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        if _is_missing_admet_ai(exc):
            return None
        raise SystemExit(
            "admet-ai is installed, but one of its dependencies failed to import: "
            f"{exc}. Fix envs/dti.yml rather than running in degraded mode."
        ) from exc
    except ImportError as exc:
        raise SystemExit(
            "admet-ai is installed, but its Python API could not be imported: "
            f"{exc}. Update the Stage 2 adapter or dependency pins."
        ) from exc
    return ADMETModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Emit an explicit unavailable ADMET-AI record instead of failing.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "auto"),
        default="cpu",
        help=(
            "Execution device policy. CPU is the workflow default so this small "
            "prediction cannot exhaust GPU memory needed by target models."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_json)
    smiles = smiles_from_sdf(args.in_sdf)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ADMETModel = _load_admet_model()
    if ADMETModel is None:
        if not args.allow_unavailable:
            raise SystemExit(
                "admet-ai is required for Stage 2 ADMET evidence; install it or set "
                "admet.allow_admet_ai_unavailable=true for an explicit degraded run"
            )
        LOG.warning("admet-ai not installed; emitting explicit unavailable result")
        _write_json_atomic({
            "smiles": smiles,
            "status": "unavailable",
            "degraded": True,
            "degraded_reason": "admet_ai_unavailable",
            "predictions": {},
        }, args.out_json)
        return

    model = ADMETModel()
    preds = model.predict(smiles=smiles)
    if hasattr(preds, "to_dict"):
        preds = preds.to_dict()
    _write_json_atomic({
        "smiles": smiles,
        "status": "ok",
        "predictions": preds,
    }, args.out_json)
    LOG.info("ADMET-AI predictions written → %s", args.out_json)


if __name__ == "__main__":
    main()
