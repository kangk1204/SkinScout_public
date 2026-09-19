#!/usr/bin/env python3
"""stage1_protonate.py — physiological-pH protonation via Dimorphite-DL 2.x.

Reads the standardized single-record SDF, extracts SMILES, enumerates the
dominant protonation state in the pH window via dimorphite_dl.protonate_smiles,
and writes the top variant back as SDF (2D coords; 3D is built in Stage 1 ETKDG).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

LOG = logging.getLogger("stage1.protonate")
DEPROTONATED_CARBOXAMIDE_N = Chem.MolFromSmarts("[N-][CX3](=[OX1])")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_sdf_atomic(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer = Chem.SDWriter(str(tmp))
    writer.write(mol)
    writer.close()
    tmp.replace(path)


def smiles_from_sdf(sdf: Path) -> str:
    sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for mol in sup:
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    raise SystemExit(f"No parseable molecule in {sdf}")


def has_implausible_deprotonated_amide_n(mol: Chem.Mol) -> bool:
    return bool(
        DEPROTONATED_CARBOXAMIDE_N is not None
        and mol.HasSubstructMatch(DEPROTONATED_CARBOXAMIDE_N)
    )


def select_protonation_variant(variants: list[str]) -> str | None:
    """pH 7.2–7.6 범위에서 가장 이온화된(지배적일 가능성이 큰) 변형을 고른다.

    Dimorphite-DL 2.0.2는 변형별 존재비를 주지 않는다. 사전순 정렬은 염기성 아민을
    중성형(`N...` < `[NH+]...`)으로, 산을 중성형으로 편향시켜 pH 7.4와 어긋난
    미세상태를 골랐다(실측: 모르폴린 pKa 8.4 → 중성 선택). 이 pH 대역에서는 산·염기
    모두 이온화 형태가 우세하므로, 전하를 가진 원자가 많은 변형을 먼저 고르고
    동점은 문자열로 고정해 실행 간 재현성을 유지한다.
    """
    ranked: list[tuple[int, str]] = []
    for variant in variants:
        mol = Chem.MolFromSmiles(variant)
        if mol is None:
            continue
        if has_implausible_deprotonated_amide_n(mol):
            continue
        charged_atoms = sum(
            1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0
        )
        ranked.append((-charged_atoms, variant))
    if ranked:
        ranked.sort()
        return ranked[0][1]
    # 모든 변형이 부적절(예: 카복사마이드 음이온)하면 배제한 구조를 되돌리지 않는다.
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-sdf", required=True, type=Path)
    parser.add_argument("--ph-min", type=float, default=7.2)
    parser.add_argument("--ph-max", type=float, default=7.6)
    parser.add_argument(
        "--allow-unprotonated-fallback",
        action="store_true",
        help="Allow neutral-input passthrough when Dimorphite-DL is unavailable or unusable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_sdf)
    smiles = smiles_from_sdf(args.in_sdf)

    try:
        from dimorphite_dl import protonate_smiles
    except ImportError:
        if not args.allow_unprotonated_fallback:
            raise SystemExit(
                "dimorphite-dl is required for Stage 1 protonation; install it or set "
                "stage1.allow_unprotonated_fallback=true for an explicit degraded run"
            )
        LOG.warning("dimorphite-dl not installed — explicit neutral-form fallback enabled.")
        protonate_smiles = None

    out_smiles = smiles
    method = "dimorphite_dl"
    fallback_reason = ""
    if protonate_smiles is not None:
        variants = protonate_smiles(
            smiles,
            ph_min=args.ph_min,
            ph_max=args.ph_max,
            max_variants=16,
        )
        if variants:
            # Dimorphite-DL 2.0.2은 내부적으로 list(set(...))을 돌려줘 순서가
            # PYTHONHASHSEED에 따라 달라진다(동일 입력이 실행마다 다른 미세상태로
            # 선택됨). 정렬·중복 제거로 실행 간 재현성을 고정한다.
            variants = sorted(set(variants))
            selected = select_protonation_variant(list(variants))
            if selected is None:
                raise SystemExit(
                    "Dimorphite가 생성한 모든 양성자 변형이 부적절(카복사마이드 음이온 등)입니다. "
                    "자동 선택 대신 명시적 검토가 필요합니다."
                )
            if selected != variants[0]:
                LOG.info(
                    "Selected protonation variant %s after rejecting implausible amide-N anion %s",
                    selected,
                    variants[0],
                )
            out_smiles = selected
        else:
            if not args.allow_unprotonated_fallback:
                raise SystemExit(
                    "Dimorphite-DL returned no protonation variant; set "
                    "stage1.allow_unprotonated_fallback=true for an explicit degraded run"
                )
            LOG.warning("Dimorphite returned no variant — explicit neutral-form fallback enabled.")
            method = "fallback_unprotonated"
            fallback_reason = "no_dimorphite_variant"
    else:
        method = "fallback_unprotonated"
        fallback_reason = "dimorphite_dl_unavailable"

    mol = Chem.MolFromSmiles(out_smiles)
    if mol is None:
        if not args.allow_unprotonated_fallback:
            raise SystemExit(f"Protonated SMILES {out_smiles!r} is unparseable")
        LOG.warning(
            "Protonated SMILES %r unparseable — explicit input fallback enabled.",
            out_smiles,
        )
        mol = Chem.MolFromSmiles(smiles)
        method = "fallback_unprotonated"
        fallback_reason = "unparseable_protonated_smiles"
    if mol is None:
        sys.exit("Could not build a molecule from either protonated or input SMILES")

    AllChem.Compute2DCoords(mol)
    mol.SetProp("protonated_smiles", out_smiles)
    mol.SetProp("protonation_method", method)
    mol.SetProp("protonation_fallback", "yes" if fallback_reason else "no")
    if fallback_reason:
        mol.SetProp("protonation_fallback_reason", fallback_reason)
    _write_sdf_atomic(mol, args.out_sdf)
    LOG.info("Protonated (pH %.1f-%.1f): %s -> %s",
             args.ph_min, args.ph_max, smiles, out_smiles)


if __name__ == "__main__":
    main()
