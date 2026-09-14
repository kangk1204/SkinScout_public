#!/usr/bin/env python3
"""Discovery ligand canonicalization helpers.

The Discovery key is intentionally stricter than ordinary RDKit canonical
SMILES so exact-evidence exclusions fail closed across salts, charges,
tautomers, and stereo-only aliases.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

RDKIT_VERSION_CONTRACT = "exact-runtime"
CANONICAL_PIPELINE = (
    "Cleanup",
    "FragmentParent",
    "Uncharger",
    "TautomerParent",
    "RemoveStereochemistry",
    "CanonicalNonIsomericSmiles",
    "SHA256",
)


def current_rdkit_version() -> str:
    return str(rdBase.rdkitVersion)


def validate_rdkit_version_contract(observed: object, *, label: str = "RDKit version") -> str:
    version = str(observed).strip()
    expected = current_rdkit_version()
    if not version:
        raise ValueError(f"{label} is missing")
    if version != expected:
        raise ValueError(
            f"{label} must match active RDKit runtime exactly: {version} != {expected}"
        )
    return version


@dataclass(frozen=True)
class DiscoveryKey:
    input_smiles: str
    parent_canonical_smiles: str
    discovery_key_sha256: str
    parent_inchikey: str
    parent_connectivity_inchikey: str
    rdkit_version: str
    rdkit_version_contract: str = RDKIT_VERSION_CONTRACT
    pipeline: tuple[str, ...] = CANONICAL_PIPELINE

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pipeline"] = list(self.pipeline)
        return payload


def discovery_key(smiles: object, *, label: str = "SMILES") -> DiscoveryKey:
    text = str(smiles).strip()
    if not text:
        raise ValueError(f"{label} is blank")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"{label} is not parseable: {text!r}")
    return discovery_key_from_mol(mol, input_text=text, label=label)


def discovery_key_from_mol(
    mol: Chem.Mol,
    *,
    input_text: str,
    label: str = "molecule",
) -> DiscoveryKey:
    if mol is None:
        raise ValueError(f"{label} is not parseable")
    try:
        parent = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(parent)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        parent = rdMolStandardize.TautomerParent(parent)
    except Exception as exc:
        raise ValueError(f"{label} failed RDKit standardization: {input_text!r}") from exc
    Chem.RemoveStereochemistry(parent)
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    if not canonical:
        raise ValueError(f"{label} produced an empty canonical parent")
    inchi_key = Chem.MolToInchiKey(parent)
    if not inchi_key:
        raise ValueError(f"{label} failed parent InChIKey generation")
    return DiscoveryKey(
        input_smiles=input_text,
        parent_canonical_smiles=canonical,
        discovery_key_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        parent_inchikey=inchi_key,
        parent_connectivity_inchikey=inchi_key.split("-", 1)[0],
        rdkit_version=current_rdkit_version(),
    )
