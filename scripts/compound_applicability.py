#!/usr/bin/env python3
"""Decide whether a compound is the kind of input this pipeline can answer for.

`INSTRUCTIONS.md` §14-15 states the scope: single small molecules only, with
mixtures and extracts separated into components first, and surfactants,
polymers and UV filters excluded because they act through physical mechanisms
rather than protein binding. Nothing in the code enforced any of it, so a
peptide or a formulation blend ran to completion and produced a ranked target
list that looked exactly like a valid one.

Two kinds of finding are reported and they are not interchangeable. A category
exclusion is a documented scope boundary and blocks the run. A property warning
only says the input sits outside the range the validation panel covers, which
is 15 compounds - it is a caution, never a verdict about the compound.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

# Measured over data/validation/skin_known_target_panel.csv (n=15). These are
# the bounds the pipeline has actually been exercised across, not a claim about
# what docking can handle in general.
PANEL_HEAVY_ATOMS = (6, 33)
PANEL_ROTATABLE_BONDS = (0, 9)
PANEL_MOLECULAR_WEIGHT = (99.0, 459.0)
PANEL_SIZE = 15

# A residue-to-residue link: an alpha carbon carrying a backbone nitrogen and a
# carbonyl that amides onto the next residue. One amide bond is ordinary in a
# small molecule (capsaicin has one); a chain of them is a peptide.
PEPTIDE_BOND = Chem.MolFromSmarts("[NX3][CX4][CX3](=[OX1])[NX3]")
PEPTIDE_BOND_LIMIT = 2
# Eight consecutive methylenes: the lipophilic tail of a surfactant.
LONG_ALKYL_CHAIN = Chem.MolFromSmarts("[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2]")
IONIC_HEAD_GROUPS = tuple(
    Chem.MolFromSmarts(pattern)
    for pattern in (
        "S(=[OX1])(=[OX1])[OX1H0-,OX2H1]",
        "P(=[OX1])([OX1H0-,OX2H1])[OX1H0-,OX2H1]",
        "[CX3](=[OX1])[OX1H0-]",
        "[NX4+]",
    )
)
POLYMER_MOLECULAR_WEIGHT = 1000.0

# UV filters are a regulatory category, not a structural class, so they are
# matched against a curated list rather than inferred. The list is NOT
# exhaustive - it covers the common organic filters and says nothing about the
# rest, so a compound absent from it is "not recognised", never "not a filter".
UV_FILTER_REFERENCE = Path(__file__).resolve().parents[1] / "data" / "validation" / "uv_filter_reference.csv"
UV_FILTER_COLUMNS = ("inci_name", "common_name", "smiles", "inchikey_connectivity", "source")

OUT_OF_SCOPE = "out_of_scope"
REVIEW = "review"
IN_SCOPE = "in_scope"


def physchem(molecule: Chem.Mol) -> dict[str, float]:
    return {
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "hbd": float(Lipinski.NumHDonors(molecule)),
        "hba": float(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(molecule)),
        "heavy_atoms": float(molecule.GetNumHeavyAtoms()),
        "rings": float(rdMolDescriptors.CalcNumRings(molecule)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        "qed": float(QED.qed(molecule)),
    }


@lru_cache(maxsize=1)
def uv_filter_index() -> dict[str, dict[str, str]]:
    """Curated UV filters keyed by InChIKey connectivity, failing closed."""
    path = UV_FILTER_REFERENCE
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"UV filter reference is required and must be non-empty: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(UV_FILTER_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(
                f"UV filter reference {path} is missing column(s): " + ", ".join(missing)
            )
        index: dict[str, dict[str, str]] = {}
        for row in reader:
            for column in UV_FILTER_COLUMNS:
                if not (row.get(column) or "").strip():
                    raise SystemExit(
                        f"UV filter reference {path} has a blank {column} row"
                    )
            key = row["inchikey_connectivity"].strip()
            if key in index:
                raise SystemExit(f"UV filter reference {path} repeats {key}")
            index[key] = {column: row[column].strip() for column in UV_FILTER_COLUMNS}
    return index


def _connectivity(molecule: Chem.Mol) -> str:
    return Chem.MolToInchiKey(molecule).split("-")[0]


def _finding(code: str, detail: str, basis: str) -> dict[str, str]:
    return {"code": code, "detail": detail, "basis": basis}


def _category_exclusions(smiles: str, molecule: Chem.Mol) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if "." in smiles:
        findings.append(
            _finding(
                "mixture",
                "여러 성분이 한 입력에 들어 있습니다. 주요 성분을 분리해 각각 실행하세요.",
                "INSTRUCTIONS.md §14",
            )
        )
    peptide_bonds = len(molecule.GetSubstructMatches(PEPTIDE_BOND))
    if peptide_bonds >= PEPTIDE_BOND_LIMIT:
        findings.append(
            _finding(
                "peptide",
                f"펩타이드 결합이 {peptide_bonds}개 이어져 있습니다. 펩타이드는 적용 범위 밖입니다.",
                "INSTRUCTIONS.md §14",
            )
        )
    if "*" in smiles:
        findings.append(
            _finding(
                "polymer",
                "반복 단위 표기(*)가 있습니다. 고분자는 적용 범위 밖입니다.",
                "INSTRUCTIONS.md §15",
            )
        )
    elif float(Descriptors.MolWt(molecule)) >= POLYMER_MOLECULAR_WEIGHT:
        findings.append(
            _finding(
                "polymer",
                f"분자량이 {Descriptors.MolWt(molecule):.0f} Da입니다. 고분자는 적용 범위 밖입니다.",
                "INSTRUCTIONS.md §15",
            )
        )
    uv_filter = uv_filter_index().get(_connectivity(molecule))
    if uv_filter is not None:
        findings.append(
            _finding(
                "uv_filter",
                f"자외선차단 성분({uv_filter['common_name']})입니다. 물리화학적 흡수로 "
                "작용하므로 적용 범위 밖입니다.",
                uv_filter["source"],
            )
        )
    if molecule.HasSubstructMatch(LONG_ALKYL_CHAIN) and any(
        molecule.HasSubstructMatch(head) for head in IONIC_HEAD_GROUPS
    ):
        findings.append(
            _finding(
                "surfactant",
                "긴 알킬 사슬과 이온성 머리기가 함께 있습니다. 계면활성제는 단백질 결합이 아니라 "
                "물리화학적 작용을 하므로 적용 범위 밖입니다.",
                "INSTRUCTIONS.md §15",
            )
        )
    return findings


def _property_warnings(properties: dict[str, float]) -> list[dict[str, str]]:
    basis = f"검증 패널 실측 범위 (n={PANEL_SIZE})"
    warnings: list[dict[str, str]] = []
    heavy = properties["heavy_atoms"]
    if not PANEL_HEAVY_ATOMS[0] <= heavy <= PANEL_HEAVY_ATOMS[1]:
        warnings.append(
            _finding(
                "heavy_atoms_outside_panel",
                f"무거운 원자 {heavy:.0f}개로, 검증 패널 범위"
                f"({PANEL_HEAVY_ATOMS[0]}–{PANEL_HEAVY_ATOMS[1]}) 밖입니다.",
                basis,
            )
        )
    rotatable = properties["rotatable_bonds"]
    if rotatable > PANEL_ROTATABLE_BONDS[1]:
        warnings.append(
            _finding(
                "rotatable_bonds_outside_panel",
                f"회전 가능 결합 {rotatable:.0f}개로, 검증 패널 최대"
                f"({PANEL_ROTATABLE_BONDS[1]})를 넘습니다. 도킹 탐색이 더 어려워집니다.",
                basis,
            )
        )
    weight = properties["molecular_weight"]
    if not PANEL_MOLECULAR_WEIGHT[0] <= weight <= PANEL_MOLECULAR_WEIGHT[1]:
        warnings.append(
            _finding(
                "molecular_weight_outside_panel",
                f"분자량 {weight:.0f} Da로, 검증 패널 범위"
                f"({PANEL_MOLECULAR_WEIGHT[0]:.0f}–{PANEL_MOLECULAR_WEIGHT[1]:.0f} Da) 밖입니다.",
                basis,
            )
        )
    return warnings


def assess(smiles: str) -> dict[str, object]:
    """Classify an input compound against the documented scope."""
    text = (smiles or "").strip()
    if not text:
        return {
            "verdict": OUT_OF_SCOPE,
            "exclusions": [
                _finding("unparseable", "SMILES가 비어 있습니다.", "RDKit")
            ],
            "warnings": [],
            "properties": {},
        }
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return {
            "verdict": OUT_OF_SCOPE,
            "exclusions": [
                _finding("unparseable", "SMILES를 해석할 수 없습니다.", "RDKit")
            ],
            "warnings": [],
            "properties": {},
        }
    exclusions = _category_exclusions(text, molecule)
    properties = physchem(molecule)
    warnings = _property_warnings(properties)
    if exclusions:
        verdict = OUT_OF_SCOPE
    elif warnings:
        verdict = REVIEW
    else:
        verdict = IN_SCOPE
    return {
        "verdict": verdict,
        "exclusions": exclusions,
        "warnings": warnings,
        "properties": properties,
    }


def assess_sdf(path: Path) -> dict[str, object]:
    """Classify an SDF input against the same scope as a SMILES input.

    Stage 1 takes the first parseable record and silently ignores the rest, so
    a multi-record file is a mixture that reaches the pipeline as whichever
    molecule happened to come first. That is reported here rather than left to
    be discovered in the results.
    """
    if not path.exists() or path.stat().st_size == 0:
        return {
            "verdict": OUT_OF_SCOPE,
            "exclusions": [
                _finding("unparseable", f"SDF 파일이 비어 있거나 없습니다: {path}", "RDKit")
            ],
            "warnings": [],
            "properties": {},
        }
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    records = list(supplier)
    molecules = [mol for mol in records if mol is not None]
    # RDKit yields None for a record it cannot read, but silently drops trailing
    # content it never recognises as a record at all. Counting the "$$$$"
    # terminators is what makes that content visible: an SDF declares one per
    # record, so more terminators than molecules means the file holds something
    # unaccounted for. A single record written without a trailing terminator is
    # common and stays valid.
    declared = sum(
        1
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() == "$$$$"
    )
    unreadable = max(declared, len(records)) - len(molecules)
    if molecules and unreadable > 0:
        return {
            "verdict": OUT_OF_SCOPE,
            "exclusions": [
                _finding(
                    "unparseable",
                    f"SDF의 레코드 {unreadable}개를 해석할 수 없습니다. 파일에 확인되지 "
                    "않은 내용이 있으므로 실행하지 않습니다.",
                    "RDKit",
                )
            ],
            "warnings": [],
            "properties": {},
        }
    if not molecules:
        return {
            "verdict": OUT_OF_SCOPE,
            "exclusions": [
                _finding("unparseable", f"SDF에 해석 가능한 분자가 없습니다: {path}", "RDKit")
            ],
            "warnings": [],
            "properties": {},
        }
    if len(molecules) > 1:
        return {
            "verdict": OUT_OF_SCOPE,
            "exclusions": [
                _finding(
                    "mixture",
                    f"SDF에 분자가 {len(molecules)}개 있습니다. 파이프라인은 첫 번째만 "
                    "사용하므로 성분을 분리해 각각 실행하세요.",
                    "INSTRUCTIONS.md §14",
                )
            ],
            "warnings": [],
            "properties": {},
        }
    return assess(Chem.MolToSmiles(molecules[0]))


def refusal_message(result: dict[str, object]) -> str | None:
    """One line naming why an input is out of scope, or None if it is not."""
    exclusions = result.get("exclusions") or []
    if not exclusions:
        return None
    reasons = " ".join(str(item.get("detail", "")) for item in exclusions)  # type: ignore[union-attr]
    sources = ", ".join(
        sorted({str(item.get("basis", "")) for item in exclusions if item.get("basis")})  # type: ignore[union-attr]
    )
    return (
        f"이 화합물은 적용 범위 밖입니다. {reasons}"
        + (f" (근거: {sources})" if sources else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles")
    parser.add_argument("--sdf", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    if bool(args.smiles) == bool(args.sdf):
        raise SystemExit("Provide exactly one of --smiles or --sdf")
    result = assess(args.smiles) if args.smiles else assess_sdf(args.sdf)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(1 if result["verdict"] == OUT_OF_SCOPE else 0)


if __name__ == "__main__":
    main()
