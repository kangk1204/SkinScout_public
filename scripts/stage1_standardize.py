#!/usr/bin/env python3
"""stage1_standardize.py — RDKit standardization for a single input compound.

Accepts EITHER --smiles "..."  OR --sdf-in path. Produces a single-record SDF
with the canonical, neutralized, salt-stripped form plus a metadata JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors

LOG = logging.getLogger("stage1.standardize")


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


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_mol(smiles: str, sdf_in: Path | None) -> Chem.Mol:
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise SystemExit(f"Could not parse SMILES: {smiles!r}")
        return mol
    if sdf_in and sdf_in.exists():
        supplier = Chem.SDMolSupplier(str(sdf_in), removeHs=False)
        for m in supplier:
            if m is not None:
                return m
        raise SystemExit(f"No parseable molecule in {sdf_in}")
    raise SystemExit("Provide --smiles or --sdf-in.")


def standardize(mol: Chem.Mol) -> Chem.Mol:
    """염·전하만 정리하고 입력 구조(토토머 포함)는 보존한다.

    인덱스·CosIng·벤치마크는 토토머를 **서로 다른 명시적 구조**로 유지한다
    (test_build_activity_retrieval_index의 tautomer 테스트). 여기서
    TautomerEnumerator.Canonicalize로 합치면 실행 경로만 구조가 달라져 신원
    키가 갈린다 - ilomastat 하이드록삼산이 `C(=O)NO`에서 `C(O)=NO`로 바뀌어
    InChIKey 입체층까지 달라진 실측 사례가 있다. 검색 인덱스와 같은
    `standardize_parent` 하나로 맞춘다.
    """
    from build_activity_retrieval_index import standardize_parent

    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        stripped = Chem.RemoveHs(mol)
        if stripped is not None and stripped.GetNumAtoms():
            mol = stripped
    return standardize_parent(mol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", default="", type=str)
    parser.add_argument("--sdf-in", default="", type=str)
    parser.add_argument("--out-sdf", required=True, type=Path)
    parser.add_argument("--out-meta", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_sdf, args.out_meta)
    input_smiles = args.smiles.strip()
    sdf_in = Path(args.sdf_in) if args.sdf_in else None
    mol = load_mol(input_smiles, sdf_in)
    input_meta: dict[str, object] = {}
    if input_smiles:
        input_meta["input_type"] = "smiles"
        input_meta["input_smiles"] = input_smiles
        input_meta["input_canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
    elif sdf_in is not None:
        input_meta["input_type"] = "sdf"
        input_meta["input_sdf"] = str(sdf_in)
    mol = standardize(mol)

    AllChem.Compute2DCoords(mol)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi = Chem.MolToInchi(mol)
    inchikey = Chem.MolToInchiKey(mol)

    mol.SetProp("_Name", inchikey)
    mol.SetProp("canonical_smiles", canonical_smiles)
    _write_sdf_atomic(mol, args.out_sdf)

    meta = {
        **input_meta,
        "canonical_smiles": canonical_smiles,
        "inchi": inchi,
        "inchikey": inchikey,
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": Descriptors.MolWt(mol),
        "logp_crippen": Crippen.MolLogP(mol),
    }
    _write_json_atomic(meta, args.out_meta)
    LOG.info("Standardized → %s  inchikey=%s", args.out_sdf, inchikey)


if __name__ == "__main__":
    main()
