#!/usr/bin/env python3
"""stage6_bioemu.py — BioEmu apo ensemble + rigid-aligned k-means clustering."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boltz2_runner import pdb_sequence  # noqa: E402

LOG = logging.getLogger("stage6.bioemu")
ALLOWED_STRUCTURE_SOURCES = {
    "alphafold_cleaned",
    "alphafold_cleaned_holo_download_failed",
    "rcsb_holo_post_cutoff",
}


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_boltz_report(path: Path) -> pd.DataFrame:
    if not nonempty(path):
        raise SystemExit(f"Boltz-2 report is required and must be non-empty: {path}")
    try:
        report = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"Boltz-2 report failed to parse: {path}: {exc}") from exc
    required = {"target_id", "kept"}
    missing_cols = sorted(required - set(report.columns))
    if missing_cols:
        raise SystemExit(f"Boltz-2 report missing required columns {missing_cols}")
    if report.empty:
        raise SystemExit("Boltz-2 report contains no rows")
    for col in sorted(required):
        invalid = [
            int(idx) for idx, value in report[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"Boltz-2 report column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        report[col] = report[col].astype(str).str.strip()
    invalid_kept = sorted(set(report.loc[~report["kept"].isin({"yes", "no"}), "kept"]))
    if invalid_kept:
        shown = ", ".join(invalid_kept[:10])
        suffix = "..." if len(invalid_kept) > 10 else ""
        raise SystemExit(f"Boltz-2 report column 'kept' contains invalid values: {shown}{suffix}")
    duplicate_ids = report["target_id"][report["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(f"Boltz-2 report contains duplicate target_id values: {shown}{suffix}")
    _validate_kept_quality_evidence(report, path)
    return report


def _validate_kept_quality_evidence(report: pd.DataFrame, path: Path) -> None:
    kept = report[report["kept"] == "yes"]
    if kept.empty:
        return
    plddt_col = "complex_plddt" if "complex_plddt" in report.columns else "pocket_plddt"
    required = {
        "source",
        "n_residues",
        "iptm",
        plddt_col,
        "affinity_log_uM",
        "pb_valid",
    }
    missing_cols = sorted(required - set(report.columns))
    if missing_cols:
        raise SystemExit(
            "Boltz-2 kept rows missing required quality evidence columns "
            f"{missing_cols}: {path}"
        )
    for col in ("source", "pb_valid"):
        invalid = [
            int(idx) for idx, value in kept[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"Boltz-2 kept rows column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
    sources = kept["source"].astype(str).str.strip()
    invalid_sources = sorted(set(sources[~sources.isin(ALLOWED_STRUCTURE_SOURCES)]))
    if invalid_sources:
        shown = ", ".join(invalid_sources[:10])
        suffix = "..." if len(invalid_sources) > 10 else ""
        allowed = ", ".join(sorted(ALLOWED_STRUCTURE_SOURCES))
        raise SystemExit(
            "Boltz-2 kept rows column 'source' contains invalid values: "
            f"{shown}{suffix}; expected one of: {allowed}"
        )
    pb = kept["pb_valid"].astype(str).str.strip()
    invalid_pb = sorted(set(pb[~pb.isin({"yes", "no", "not_available"})]))
    if invalid_pb:
        shown = ", ".join(invalid_pb[:10])
        suffix = "..." if len(invalid_pb) > 10 else ""
        raise SystemExit(
            f"Boltz-2 kept rows column 'pb_valid' contains invalid values: {shown}{suffix}"
        )
    pb_failed = pb == "no"
    if pb_failed.any():
        bad_ids = kept.loc[pb_failed, "target_id"].astype(str).tolist()
        shown = ", ".join(bad_ids[:10])
        suffix = "..." if len(bad_ids) > 10 else ""
        raise SystemExit(
            "Boltz-2 kept rows must not have pb_valid=no for target_id values: "
            f"{shown}{suffix}"
        )
    _validate_numeric_kept_column(kept, "n_residues", path, min_value=1.0)
    _validate_numeric_kept_column(kept, "iptm", path, min_value=0.0, max_value=1.0)
    _validate_numeric_kept_column(
        kept,
        plddt_col,
        path,
        min_value=0.0,
        max_value=100.0,
    )
    _validate_numeric_kept_column(kept, "affinity_log_uM", path)


def _validate_numeric_kept_column(
    kept: pd.DataFrame,
    col: str,
    path: Path,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    bool_like = [
        int(idx)
        for idx, value in kept[col].items()
        if (
            isinstance(value, bool)
            or type(value).__name__ == "bool_"
            or (
                isinstance(value, str)
                and value.strip().lower() in {"true", "false"}
            )
        )
    ]
    if bool_like:
        shown = ", ".join(str(idx) for idx in bool_like[:10])
        suffix = "..." if len(bool_like) > 10 else ""
        raise SystemExit(
            f"Boltz-2 kept rows column '{col}' must be numeric at row index(es) "
            f"{shown}{suffix}: {path}"
        )
    values = pd.to_numeric(kept[col], errors="coerce")
    invalid = values[values.isna()].index.tolist()
    if invalid:
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            f"Boltz-2 kept rows column '{col}' must be numeric at row index(es) "
            f"{shown}{suffix}: {path}"
        )
    nonfinite = [
        int(idx)
        for idx, value in values.items()
        if not math.isfinite(float(value))
    ]
    if nonfinite:
        shown = ", ".join(str(idx) for idx in nonfinite[:10])
        suffix = "..." if len(nonfinite) > 10 else ""
        raise SystemExit(
            f"Boltz-2 kept rows column '{col}' must be finite at row index(es) "
            f"{shown}{suffix}: {path}"
        )
    if min_value is not None:
        below = values[values < min_value].index.tolist()
        if below:
            shown = ", ".join(str(idx) for idx in below[:10])
            suffix = "..." if len(below) > 10 else ""
            raise SystemExit(
                f"Boltz-2 kept rows column '{col}' must be >= {min_value} at "
                f"row index(es) {shown}{suffix}: {path}"
            )
    if max_value is not None:
        above = values[values > max_value].index.tolist()
        if above:
            shown = ", ".join(str(idx) for idx in above[:10])
            suffix = "..." if len(above) > 10 else ""
            raise SystemExit(
                f"Boltz-2 kept rows column '{col}' must be <= {max_value} at "
                f"row index(es) {shown}{suffix}: {path}"
            )


def bioemu_command(sequence: str, out_dir: Path, n_conf: int) -> list[str]:
    """The command line BioEmu actually accepts.

    The package ships no ``bioemu`` console script - it is invoked as
    ``python -m bioemu.sample`` - and ``sample`` takes three positional
    arguments (sequence, sample count, output directory), not the
    ``--receptor/--num-conformers/--out`` flags this used to pass. Both
    mistakes failed the same silent way: ``shutil.which("bioemu")`` returned
    None, ``run_bioemu`` returned False, and the stage reported a sampling
    failure rather than a wiring failure.

    BioEmu samples from a *sequence*, not from a receptor structure, so the
    caller extracts the sequence from the prepared PDB.
    """
    return [sys.executable, "-m", "bioemu.sample",
            sequence, str(n_conf), str(out_dir)]


def bioemu_available() -> bool:
    """Whether BioEmu can be imported by the interpreter running this script."""
    return importlib.util.find_spec("bioemu") is not None


def run_bioemu(receptor: Path, out_dir: Path, n_conf: int) -> bool:
    if not nonempty(receptor):
        return False
    if not bioemu_available():
        return False
    sequence = pdb_sequence(receptor)
    if not sequence:
        LOG.warning("No protein sequence in %s; BioEmu cannot sample it", receptor)
        return False
    res = subprocess.run(bioemu_command(sequence, out_dir, n_conf),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        LOG.warning("BioEmu sampling failed for %s: %s",
                    receptor.name, res.stderr[-400:])
    return res.returncode == 0


# 곁사슬을 놓은 뒤 완화할 때 백본을 붙드는 세기. 세게 잡으면 BioEmu 가 뽑은
# 형태가 그대로 남고, 너무 세면 왜곡된 결합 길이도 그대로 남는다. 1000 에서는
# 실측으로 펩타이드 결합이 1.15 A 에서 정상값 1.34 A 로 돌아왔고 접힘은 유지됐다.
# 곁사슬 재구성의 난수 씨앗. 고정해 두어야 같은 medoid 가 늘 같은 결과를 낸다.
RECONSTRUCTION_SEED = 20260906
# 재현을 위해 플랫폼도 고정한다. CPU 는 스레드 하나로 둔다.
# CUDA 는 `DeterministicForces` 를 켜면 같은 기기에서 축약 순서를 고정한다.
# CPU 는 스레드 하나로 두면 재현되지만 가장 큰 계에서 2분을 넘겼다.
RECONSTRUCTION_PLATFORM = "CUDA"
RECONSTRUCTION_PLATFORM_PROPERTIES = {
    "CUDA": {"DeterministicForces": "true"},
    "CPU": {"Threads": "1"},
}

BACKBONE_RESTRAINT_K = 1000.0
# 0 은 "수렴할 때까지". 실측으로 1초 미만이고, 500회로 끊으면 프레임에 따라
# 국소 최소에 갇혀 카복실기 산소 두 개가 1.59 A(정상 2.2 A)로 남았다. 그 잔기
# 하나 때문에 RDKit 이 산소에 결합 3개를 매기고 RTMScore 가 수용체 전체를 버린다.
SIDECHAIN_MIN_ITERATIONS = 0
# 쓴 구조를 다시 재서 확인한다. 최소화가 늘 성공하지는 않으므로, 성공했다고
# 가정하는 대신 결과를 본다 - 스테이지 4 에서 같은 규칙이 통했다.
MAX_MINIMISATION_ATTEMPTS = 3


def _write_heavy_atoms(app: Any, topology: Any, positions: Any, out: Path) -> None:
    """중원자만 PDB 로 쓴다. 수소는 완화에만 쓰고 하류에는 넘기지 않는다."""
    modeller = app.Modeller(topology, positions)
    modeller.delete(
        [atom for atom in modeller.topology.atoms() if atom.element.symbol == "H"]
    )
    with out.open("w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)


# 표준 아미노산의 무거운 원자 결합. 백본(N-CA, CA-C, C-O, C-OXT, CA-CB)은
# 모든 잔기가 공유하므로 따로 두고, 여기에는 곁사슬만 적는다. 이 표가 있으면
# 어떤 원자쌍이 결합인지 거리로 짐작하지 않아도 된다 - 짐작하면 순환논법이
# 된다(붙어 버린 두 원자는 거리로는 결합처럼 보인다).
BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"), ("CA", "CB"))
SIDECHAIN_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "GLY": (), "ALA": (),
    "SER": (("CB", "OG"),),
    "CYS": (("CB", "SG"),),
    "THR": (("CB", "OG1"), ("CB", "CG2")),
    "VAL": (("CB", "CG1"), ("CB", "CG2")),
    "LEU": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "ILE": (("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
    "MET": (("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "PRO": (("CB", "CG"), ("CG", "CD"), ("CD", "N")),
    "PHE": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"),
            ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")),
    "TYR": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"),
            ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")),
    "TRP": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"),
            ("NE1", "CE2"), ("CD2", "CE2"), ("CD2", "CE3"), ("CE3", "CZ3"),
            ("CZ3", "CH2"), ("CH2", "CZ2"), ("CZ2", "CE2")),
    "ASP": (("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "ASN": (("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "GLU": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "GLN": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
    "LYS": (("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "ARG": (("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"),
            ("CZ", "NH1"), ("CZ", "NH2")),
    "HIS": (("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"),
            ("NE2", "CD2"), ("CD2", "CG")),
}

# 두 결합 건너(1-3) 원자쌍의 하한. 실제 실험 구조에서 잰 정상값이 2.18~2.23 A 다.
GEMINAL_FLOOR_ANGSTROM = 1.9
# 결합도 1-3 도 아닌 무거운 원자쌍의 하한. 수소결합조차 2.7 A 이상이다.
NONBONDED_FLOOR_ANGSTROM = 2.2
# 펩타이드 결합 C(i)-N(i+1) 과 이황화 SG-SG 는 잔기를 건너 실제로 이어져 있다.
PEPTIDE_BOND_MAX_ANGSTROM = 1.8
DISULFIDE_MAX_ANGSTROM = 2.5


def _read_heavy_atoms(pdb: Path):
    """PDB 에서 무거운 원자만 (잔기키, 잔기이름, 원자이름, 좌표) 로 읽는다."""
    atoms = []
    for line in pdb.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip().upper()
        element = line[76:78].strip().upper()
        if element == "H" or (not element and name.startswith("H")):
            continue
        atoms.append((
            (line[21:22], line[22:27].strip()),
            line[17:20].strip().upper(),
            name,
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        ))
    return atoms


def _bond_graph(atoms) -> dict[int, set[int]]:
    """원자 색인 사이의 공유결합 그래프.

    거리로 결합을 매기지 않는다. 잔기 표에서 어떤 원자쌍이 이어져 있는지 읽고,
    잔기를 건너는 것은 펩타이드 결합과 이황화만 거리로 확인한다. 거리로 결합을
    정하면 겹쳐 버린 두 원자가 결합으로 오인돼 검사 자체가 무력해진다.
    """
    import math

    index: dict[tuple[tuple[str, str], str], int] = {}
    for i, (key, _resname, name, _xyz) in enumerate(atoms):
        index[(key, name)] = i
    graph: dict[int, set[int]] = {i: set() for i in range(len(atoms))}

    def link(a: int, b: int) -> None:
        graph[a].add(b)
        graph[b].add(a)

    residues: dict[tuple[str, str], str] = {}
    for key, resname, _name, _xyz in atoms:
        residues[key] = resname
    for key, resname in residues.items():
        for a, b in BACKBONE_BONDS + SIDECHAIN_BONDS.get(resname, ()):
            ia, ib = index.get((key, a)), index.get((key, b))
            if ia is not None and ib is not None:
                link(ia, ib)
    # 잔기를 잇는 결합. 사슬 안에서 이웃한 잔기의 C-N 과, 임의의 두 CYS 의 SG-SG.
    ordered = sorted(residues, key=lambda k: (k[0], int(k[1]) if k[1].lstrip("-").isdigit() else 0, k[1]))
    for first, second in zip(ordered, ordered[1:]):
        if first[0] != second[0]:
            continue
        ic, jn = index.get((first, "C")), index.get((second, "N"))
        if ic is not None and jn is not None and \
                math.dist(atoms[ic][3], atoms[jn][3]) <= PEPTIDE_BOND_MAX_ANGSTROM:
            link(ic, jn)
    sulfurs = [i for i, (_k, resname, name, _x) in enumerate(atoms)
               if resname == "CYS" and name == "SG"]
    for a_i, ia in enumerate(sulfurs):
        for ib in sulfurs[a_i + 1:]:
            if math.dist(atoms[ia][3], atoms[ib][3]) <= DISULFIDE_MAX_ANGSTROM:
                link(ia, ib)
    return graph


def strained_residues(pdb: Path) -> list[str]:
    """기하가 물리적으로 불가능한 원자쌍이 있는 잔기 목록.

    RDKit 은 PDB 를 읽을 때 거리로 결합을 매긴다. 카복실기 산소 두 개가 1.59 A
    (정상 2.2 A)로 붙어 있으면 산소에 결합 3개가 매겨지고, RTMScore 는 그
    수용체 전체에서 포켓 그래프를 만들지 못한다. 잔기 하나가 구조 하나를 버린다.

    이 검사는 잔기·원자 이름을 나열한 표가 아니라 결합 그래프에서 나온다.
    표를 쓰면 표에 적힌 11쌍 밖은 전부 보이지 않는다 - LYS 의 CD-NZ 가 겹쳐도,
    다른 잔기의 원자가 파고들어도 통과한다. 여기서는 두 가지를 본다:
      (a) 두 결합 건너(1-3) 무거운 원자쌍이 1.9 A 보다 가까운 경우
      (b) 결합도 1-3 도 아닌 무거운 원자쌍이 2.2 A 보다 가까운 경우
          - 잔기를 넘는 충돌이 여기에 걸린다
    """
    import math

    atoms = _read_heavy_atoms(pdb)
    if not atoms:
        return []
    graph = _bond_graph(atoms)
    geminal: set[tuple[int, int]] = set()
    for centre, neighbours in graph.items():
        ordered = sorted(neighbours)
        for a_i, ia in enumerate(ordered):
            for ib in ordered[a_i + 1:]:
                geminal.add((ia, ib))

    coords = [atom[3] for atom in atoms]
    pairs = _close_pairs(coords, NONBONDED_FLOOR_ANGSTROM)
    found: dict[tuple[int, int], float] = {}
    for ia, ib in pairs:
        if ib in graph[ia]:
            continue
        distance = math.dist(coords[ia], coords[ib])
        floor = (GEMINAL_FLOOR_ANGSTROM if (min(ia, ib), max(ia, ib)) in geminal
                 else NONBONDED_FLOOR_ANGSTROM)
        if distance < floor:
            found[(min(ia, ib), max(ia, ib))] = distance

    # 잔기 하나가 여러 쌍에서 뒤틀릴 수 있다. 소비하는 쪽(`_pull_apart`, 재시도
    # 판정)은 잔기 단위로 세므로, 잔기마다 가장 심한 쌍 하나를 대표로 적고 나머지
    # 개수는 뒤에 붙인다.
    worst: dict[tuple[str, str], tuple[float, str]] = {}
    extra: dict[tuple[str, str], int] = {}
    for (ia, ib), distance in sorted(found.items()):
        key, resname, name_a, _ = atoms[ia]
        other_key, other_resname, name_b, _ = atoms[ib]
        if key == other_key:
            label = f"{name_a}-{name_b}"
        else:
            label = f"{name_a}-{other_resname}{other_key[1]}.{name_b}"
        floor = (GEMINAL_FLOOR_ANGSTROM if (ia, ib) in geminal
                 else NONBONDED_FLOOR_ANGSTROM)
        severity = floor - distance
        token = f"{key[0]}:{resname}{key[1]}({label} {distance:.2f}A)"
        extra[key] = extra.get(key, 0) + 1
        if key not in worst or severity > worst[key][0]:
            worst[key] = (severity, token)
    bad: list[str] = []
    for key in sorted(worst, key=lambda k: -worst[k][0]):
        token = worst[key][1]
        others = extra[key] - 1
        bad.append(token if others == 0 else f"{token} +{others}")
    return bad


def _close_pairs(coords, cutoff: float):
    """cutoff 안에 든 원자쌍. 원자 수가 만 단위라 격자로 나눠 본다."""
    cells: dict[tuple[int, int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(coords):
        cells.setdefault((int(x // cutoff), int(y // cutoff), int(z // cutoff)),
                         []).append(i)
    neighbourhood = [(dx, dy, dz)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    seen: set[tuple[int, int]] = set()
    for cell, members in cells.items():
        for dx, dy, dz in neighbourhood:
            other = cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
            if not other:
                continue
            for ia in members:
                for ib in other:
                    if ia < ib:
                        seen.add((ia, ib))
    return seen


# 한 원자에 붙은 두 말단 원자의 정상 거리. 실제 실험 구조 40개에서 잰 값이다
# (ASN 2.19, ASP 2.21, GLU 2.21, ARG 2.23, GLN 2.18 A; 1.9 A 아래는 0건).
IDEAL_TERMINAL_PAIR_ANGSTROM = 2.20


def _pull_apart(positions: list, fixer: Any, unit: Any,
                strained: list[str]) -> list:
    """뒤틀린 말단 원자쌍을 정상 거리로 벌려 놓는다.

    중점은 그대로 두고 두 원자를 양쪽으로 민다. 완벽한 기하를 만드는 것이
    목적이 아니라, 최소화가 옳은 골짜기에서 출발하게 하는 것이 목적이다.
    """
    from openmm import Vec3

    pairs = {
        "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
        "ASN": ("OD1", "ND2"), "GLN": ("OE1", "NE2"),
        "ARG": ("NH1", "NH2"), "VAL": ("CG1", "CG2"),
        "LEU": ("CD1", "CD2"), "ILE": ("CG1", "CG2"),
        "THR": ("OG1", "CG2"), "PHE": ("CD1", "CD2"), "TYR": ("CD1", "CD2"),
    }
    wanted = {token.split("(", 1)[0] for token in strained}
    nm = unit.nanometer
    target = IDEAL_TERMINAL_PAIR_ANGSTROM / 10.0  # nm

    by_residue: dict[Any, dict[str, int]] = {}
    for atom in fixer.topology.atoms():
        key = f"{atom.residue.chain.id}:{atom.residue.name}{atom.residue.id}"
        if key in wanted:
            by_residue.setdefault(key, {})[atom.name] = atom.index

    for key, atoms in by_residue.items():
        resname = "".join(ch for ch in key.split(":", 1)[1] if ch.isalpha())
        names = pairs.get(resname)
        if not names or names[0] not in atoms or names[1] not in atoms:
            continue
        i, j = atoms[names[0]], atoms[names[1]]
        a = positions[i].value_in_unit(nm)
        b = positions[j].value_in_unit(nm)
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        length = (dx * dx + dy * dy + dz * dz) ** 0.5
        if length <= 0 or length >= target:
            continue
        push = (target - length) / 2.0 / length
        positions[i] = Vec3(a[0] - dx * push, a[1] - dy * push, a[2] - dz * push) * nm
        positions[j] = Vec3(b[0] + dx * push, b[1] + dy * push, b[2] + dz * push) * nm
        LOG.info("  %s 의 %s-%s 를 %.2f → %.2f A 로 벌려 다시 내려갑니다",
                 key, names[0], names[1], length * 10, target * 10)
    return positions


def _relax_only(simulation: Any, fixer: Any, app: Any, unit: Any,
                out: Path, strained: list[str], pdb: Path) -> list[str]:
    """뒤틀린 잔기만 자유롭게 두고 나머지를 묶은 채 다시 내려간다.

    실측으로 남는 것은 언제나 같은 종류다 - 한 원자에 붙은 두 말단 원자가
    1.55~1.60 A 로 붙는다(실제 구조에서는 2.14~2.26 A). 이웃 곁사슬이 만든
    국소 최소에서 그 잔기만 빠져나오면 되는데, 전체를 함께 움직이면 이웃이
    도로 밀어 넣는다.
    """
    from openmm import CustomExternalForce

    wanted = {token.split("(", 1)[0] for token in strained}

    def _residue_key(residue: Any) -> str:
        return f"{residue.chain.id}:{residue.name}{residue.id}"

    free = {
        atom.index
        for atom in fixer.topology.atoms()
        if _residue_key(atom.residue) in wanted and atom.name not in ("N", "CA", "C", "O")
    }
    if not free:
        return strained

    state = simulation.context.getState(getPositions=True)
    positions = list(state.getPositions())

    # 최소화만으로는 안 된다. 두 말단 원자가 1.55~1.60 A 로 붙으면 그 자리가
    # 국소 최소이고, 정상 거리(실측 2.14~2.26 A)로 가려면 각도 장벽을 넘어야
    # 하는데 내리막만 따라가는 최소화는 그것을 못 넘는다. 그래서 **먼저 벌려
    # 놓고** 내려간다 - 장벽 반대편에서 출발하면 최소화가 제 일을 한다.
    positions = _pull_apart(positions, fixer, unit, strained)
    simulation.context.setPositions(positions)
    # 전역 매개변수 이름은 계 안에서 유일해야 한다. 백본 구속이 이미 `k` 를
    # 쓰고 있어서, 같은 이름으로 두 번째 힘을 넣으면 OpenMM 이
    # "Two Forces define different default values for the parameter 'k'" 로
    # 거부한다 - 그러면 이 잔기 하나를 고치려던 시도가 구조 전체의 복원을
    # 실패시키고, 백본만 있는 파일이 남는다. 실측으로 Q14393 의 medoid 하나가
    # 그렇게 곁사슬 0% 로 나갔다.
    hold = CustomExternalForce("0.5*k_hold*((x-x1)^2+(y-y1)^2+(z-z1)^2)")
    hold.addGlobalParameter(
        "k_hold", 50_000.0 * unit.kilojoule_per_mole / unit.nanometer**2
    )
    for name in ("x1", "y1", "z1"):
        hold.addPerParticleParameter(name)
    for atom in fixer.topology.atoms():
        if atom.index not in free:
            hold.addParticle(
                atom.index, positions[atom.index].value_in_unit(unit.nanometer)
            )
    simulation.system.addForce(hold)
    simulation.context.reinitialize(preserveState=True)
    simulation.minimizeEnergy(maxIterations=SIDECHAIN_MIN_ITERATIONS)
    relaxed = simulation.context.getState(getPositions=True)
    _write_heavy_atoms(app, fixer.topology, relaxed.getPositions(), out)
    remaining = strained_residues(out)
    LOG.info("%s: 뒤틀린 잔기만 풀어 다시 완화 → %d개 남음",
             pdb.name, len(remaining))
    return remaining


def reconstruct_sidechains(pdb: Path) -> str:
    """백본만 있는 구조에 곁사슬을 채우고, 충돌만 푼다. 상태 문자열을 돌려준다.

    BioEmu 는 백본(N/CA/C/O)과 CB 만 낸다. 실측으로 257잔기 구조에 곁사슬 중원자
    785개가 비어 있었다. 그 상태로 하류에 넘기면 RTMScore 는 포켓 그래프를 만들지
    못해 멈추지만(정직한 실패) GNINA 는 **멈추지 않고 점수를 낸다.** 곁사슬이 없는
    포켓은 실제보다 훨씬 열려 있어 그 점수는 다른 구조와 견줄 수 없는데, 화면에는
    정상적인 숫자로 나온다. 조용히 틀린 답이 멈추는 것보다 나쁘다.

    두 단계다.

    1. `pdbfixer` 가 빠진 중원자를 채운다. 다만 기본 로타머를 놓을 뿐 충돌을 풀지
       않아서, 실측으로 PHE 16 과 TRP 12 의 곁사슬이 1.73 A 로 겹쳤다. RDKit 은
       그 거리를 결합으로 읽어 탄소에 결합 5개를 매기고, RTMScore 가 거기서 멈춘다.
    2. OpenMM 으로 **백본을 붙든 채** 에너지를 최소화해 충돌만 푼다. 접힘은
       BioEmu 가 뽑은 것이어야 하므로 백본은 제자리에 둔다. 실측으로 결합 아닌
       2.2 A 미만 접촉이 129개에서 0개가 됐고 4초가 걸렸다.

    빠진 **잔기**는 만들지 않는다. 없는 잔기를 지어내는 것은 복원이 아니라 창작이고,
    앙상블은 BioEmu 가 낸 잔기 집합 그대로여야 프레임끼리 비교된다.
    """
    try:
        import openmm
        from openmm import CustomExternalForce, LangevinIntegrator, app, unit
        from pdbfixer import PDBFixer
    except ImportError:
        LOG.warning(
            "pdbfixer/openmm 이 없어 곁사슬을 복원하지 못했습니다: %s. "
            "백본만 있는 수용체의 도킹 점수는 신뢰할 수 없습니다.", pdb,
        )
        return "backbone_only_pdbfixer_missing"

    try:
        fixer = PDBFixer(filename=str(pdb))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        n_missing = sum(len(v) for v in fixer.missingAtoms.values())
        # pdbfixer 는 새로 놓은 원자를 자기 랑주뱅 적분기로 다듬는다. 씨앗을 주지
        # 않으면 그 단계가 매번 다른 난수를 쓰고, 뒤에서 적분기와 플랫폼을 아무리
        # 고정해도 출발 좌표가 이미 달라져 있다. 실측: 씨앗 없이 같은 medoid 를
        # 두 번 돌리면 원자 하나가 1.74 A 옮겨졌다.
        # pdbfixer 의 두 단계 모두 스레드 하나짜리 CPU 에서 돌린다. 기본 플랫폼
        # (CUDA)이나 다중 스레드 CPU 는 축약 순서가 실행마다 달라, 원자를 놓는
        # 최소화가 매번 조금씩 다른 곳에서 멈춘다.
        fixer.platform = openmm.Platform.getPlatformByName("CPU")
        fixer.platform.setPropertyDefaultValue("Threads", "1")
        fixer.addMissingAtoms(seed=RECONSTRUCTION_SEED)
        # 힘장은 수소가 있어야 계를 만들 수 있다. 완화가 끝나면 다시 뺀다.
        # Modeller 은 수소를 난수 위치에 놓고 최소화한다(`random.random()`).
        # 씨앗을 고정하지 않으면 그 출발점이 매번 다르다.
        random.seed(RECONSTRUCTION_SEED)
        fixer.addMissingHydrogens(7.0)

        forcefield = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
        system = forcefield.createSystem(
            fixer.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
        restraint = CustomExternalForce(
            "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k", BACKBONE_RESTRAINT_K * unit.kilojoule_per_mole / unit.nanometer**2
        )
        for name in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)
        for atom in fixer.topology.atoms():
            if atom.name in ("N", "CA", "C", "O"):
                restraint.addParticle(
                    atom.index,
                    fixer.positions[atom.index].value_in_unit(unit.nanometer),
                )
        system.addForce(restraint)

        # 적분기와 속도 생성에 고정 씨앗을 준다. 주지 않으면 OpenMM 이 매번 다른
        # 난수를 뽑아, **같은 medoid 파일이 실행마다 다른 기하와 다른 판정**을
        # 낸다 - 한 번은 통과하고 다음엔 뒤틀린 잔기가 남아 버려진다. 그러면
        # 버려진 구조를 다시 재현해 원인을 볼 수 없다.
        integrator = LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
        )
        integrator.setRandomNumberSeed(RECONSTRUCTION_SEED)
        # 씨앗만으로는 부족하다. 플랫폼을 정해 주지 않으면 OpenMM 은 가장 빠른
        # 것(여기서는 CUDA)을 잡는데, GPU 의 축약 순서가 실행마다 달라 같은 입력이
        # 다른 좌표를 낸다 - 실측으로 원자 하나가 1.74 A 옮겨졌고, 그것은 뒤틀림
        # 판정선(1.9 A)을 뒤집기에 충분하다. 스레드 하나짜리 CPU 로 고정하면
        # 재현된다. 가장 큰 계(9,334 원자)에서도 비용은 초 단위다.
        platform = openmm.Platform.getPlatformByName(RECONSTRUCTION_PLATFORM)
        properties = RECONSTRUCTION_PLATFORM_PROPERTIES.get(
            RECONSTRUCTION_PLATFORM, {})
        simulation = app.Simulation(fixer.topology, system, integrator,
                                    platform, properties)
        simulation.context.setPositions(fixer.positions)

        # 최소화가 늘 되는 것은 아니다. pdbfixer 가 놓은 기본 로타머에서
        # 출발하면 프레임에 따라 국소 최소에 갇혀, 카복실기 산소 두 개가
        # 1.59 A(정상 2.2 A)로 남는다. 잔기 하나가 그러면 RTMScore 는 수용체
        # 전체를 버린다. 그래서 성공을 가정하지 않고 결과를 재 보고, 남아 있으면
        # 살짝 흔들어 다시 내려간다.
        tmp = pdb.with_suffix(".pdb.tmp")
        strained: list[str] = []
        for attempt in range(1, MAX_MINIMISATION_ATTEMPTS + 1):
            if attempt > 1:
                # 같은 자리에서 다시 내려가면 같은 곳에 갇힌다. 짧게 데워
                # 갇힌 골짜기에서 빼낸 뒤 다시 내려간다. 백본은 계속 묶여 있다.
                # 시도마다 다른 씨앗을 주되, 그 씨앗도 정해져 있다. 같은 입력을
                # 다시 돌리면 같은 시도에서 같은 흔들림을 받는다.
                simulation.context.setVelocitiesToTemperature(
                    300 * unit.kelvin, RECONSTRUCTION_SEED + attempt
                )
                simulation.step(200)
            simulation.minimizeEnergy(maxIterations=SIDECHAIN_MIN_ITERATIONS)
            state = simulation.context.getState(getPositions=True)
            _write_heavy_atoms(app, fixer.topology, state.getPositions(), tmp)
            strained = strained_residues(tmp)
            if not strained:
                break
            LOG.info("%s: 뒤틀린 잔기 %d개, 다시 완화합니다 (%d/%d): %s",
                     pdb.name, len(strained), attempt,
                     MAX_MINIMISATION_ATTEMPTS, ", ".join(strained[:3]))
        if strained:
            # 마지막 수단: 뒤틀린 잔기만 풀어 주고 나머지를 전부 묶는다. 앞의
            # 시도들은 백본만 묶고 곁사슬 전체를 함께 움직였는데, 그러면 이웃
            # 곁사슬들이 만드는 국소 최소에서 문제의 잔기가 빠져나오지 못한다.
            strained = _relax_only(
                simulation, fixer, app, unit, tmp, strained, pdb
            )
        if strained:
            # 여기서도 안 되면 **구조는 남기고** 사실을 알린다. 잔기 하나를
            # 피하려고 복원한 곁사슬 수백 개를 버리면 백본만 있는 파일이
            # 남는데, 그것이야말로 stage 6.5 의 게이트가 막는 상태다.
            # 작은 문제를 완전한 실패로 바꾸는 교환이다.
            LOG.warning(
                "%s: 완화 후에도 뒤틀린 잔기 %d개가 남았습니다(구조는 유지): %s",
                pdb.name, len(strained), ", ".join(strained),
            )
            tmp.replace(pdb)
            return "pdbfixer_sidechains_openmm_relaxed_with_strain"
    except Exception as exc:  # noqa: BLE001 - pdbfixer/openmm 예외가 넓다
        LOG.warning("곁사슬 복원 실패 %s: %s", pdb, exc)
        return "sidechain_reconstruction_failed"
    tmp.replace(pdb)
    LOG.info("Reconstructed %d side-chain atom(s) and relaxed clashes in %s",
             n_missing, pdb.name)
    return "pdbfixer_sidechains_openmm_relaxed"


def rigid_invariant_coordinates(xyz: Any) -> Any:
    """Align coordinate frames to frame zero with the Kabsch transform."""
    import numpy as np

    points = np.asarray(xyz, dtype=float)
    if points.ndim != 3 or points.shape[0] < 1 or points.shape[1] < 1:
        raise ValueError("xyz must have shape (frames, atoms, 3)")
    reference = points[0] - points[0].mean(axis=0)
    aligned = []
    for frame in points:
        centered = frame - frame.mean(axis=0)
        covariance = centered.T @ reference
        left, _, right = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(left @ right))
        rotation = left @ correction @ right
        aligned.append(centered @ rotation)
    return np.asarray(aligned)


def cluster_conformers(conformer_dir: Path, k: int) -> list[Path]:
    """BioEmu 가 낸 앙상블을 k 개로 묶고 각 묶음의 대표 구조를 쓴다.

    BioEmu 는 샘플을 **궤적**으로 낸다: `samples.xtc` 한 개와 `topology.pdb`
    한 개다. 예전에는 이 디렉터리에서 `*.pdb` 를 찾아 컨포머로 삼았는데, 거기
    걸리는 것은 `topology.pdb` 하나뿐이다. 그래서 KMeans 가 구조 한 개를
    한 묶음으로 묶고, 출발 구조 자신을 "앙상블 대표"로 돌려주었다 - 스테이지가
    성공으로 끝나고 매니페스트도 나오지만, 앙상블은 어디에도 없었다.

    (이 함수는 이름과 달리 pyemma 를 쓴 적이 없다. tICA 도 하지 않는다.
    CA 좌표에 KMeans 를 걸 뿐이다. 이름을 사실에 맞췄다.)
    """
    try:
        import mdtraj as md
        import numpy as np
        from sklearn.cluster import KMeans
    except ImportError:
        return []

    topology = conformer_dir / "topology.pdb"
    trajectories = sorted(conformer_dir.glob("*.xtc"))
    if trajectories and nonempty(topology):
        frames = []
        for xtc in trajectories:
            if not nonempty(xtc):
                continue
            try:
                frames.append(md.load(str(xtc), top=str(topology)))
            except Exception as exc:  # noqa: BLE001 - mdtraj 예외가 넓다
                LOG.warning("Could not read %s: %s", xtc, exc)
        if not frames:
            return []
        traj = md.join(frames) if len(frames) > 1 else frames[0]
        source = "trajectory"
    else:
        # 컨포머를 파일 하나씩 내는 도구를 쓰는 경우. topology.pdb 는 샘플이
        # 아니므로 뺀다.
        pdbs = sorted(p for p in conformer_dir.glob("*.pdb")
                      if nonempty(p) and p.name != "topology.pdb")
        if not pdbs:
            return []
        try:
            traj = md.join([md.load(str(p)) for p in pdbs])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not read conformer PDBs in %s: %s", conformer_dir, exc)
            return []
        source = "pdb_files"

    ca = traj.atom_slice(traj.topology.select("name CA"))
    n_frames = ca.n_frames
    if n_frames == 0:
        return []
    if ca.n_atoms == 0:
        return []
    # Cartesian coordinates only describe conformation after quotienting out
    # rigid translation and rotation.  Align every frame to the first CA trace
    # before clustering so a moved copy is not treated as a new conformation.
    X = rigid_invariant_coordinates(ca.xyz).reshape(n_frames, -1)
    n_clusters = min(k, n_frames)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)

    medoids: list[Path] = []
    for ci, center in enumerate(km.cluster_centers_):
        members = [i for i, lab in enumerate(km.labels_) if lab == ci]
        if not members:
            continue
        dists = [float(np.linalg.norm(X[i] - center)) for i in members]
        frame = members[int(np.argmin(dists))]
        out = conformer_dir / f"medoid_{ci:02d}.pdb"
        traj[frame].save_pdb(str(out))
        reconstruction_status = reconstruct_sidechains(out)
        if reconstruction_status != "pdbfixer_sidechains_openmm_relaxed":
            rejected = out.with_name(f"{out.stem}.rejected.pdb")
            if nonempty(out):
                out.replace(rejected)
            LOG.warning(
                "Rejected BioEmu medoid %s after side-chain validation: %s",
                rejected, reconstruction_status,
            )
            continue
        medoids.append(out)
    LOG.info("Clustered %d %s frame(s) into %d medoid(s) in %s",
             n_frames, source, len(medoids), conformer_dir)
    return medoids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boltz-report", required=True, type=Path)
    parser.add_argument("--struct-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--num-conformers", type=int, default=1000)
    parser.add_argument("--kmeans-k", type=int, default=8)
    parser.add_argument("--out-manifest", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.out_manifest.exists():
        args.out_manifest.unlink()

    if args.top_n < 1:
        raise SystemExit(f"--top-n must be >= 1: {args.top_n}")
    if args.num_conformers < 1:
        raise SystemExit(f"--num-conformers must be >= 1: {args.num_conformers}")
    if args.kmeans_k < 1:
        raise SystemExit(f"--kmeans-k must be >= 1: {args.kmeans_k}")

    boltz = read_boltz_report(args.boltz_report)
    boltz = boltz[boltz["kept"] == "yes"].head(args.top_n)
    if boltz.empty:
        raise SystemExit("No Boltz-2 kept targets available for BioEmu")
    if not bioemu_available():
        raise SystemExit(
            "bioemu is not importable by this interpreter; run this stage in "
            "the environment built from envs/bioemu.yml"
        )
    LOG.info("Top-N for BioEmu: %d", len(boltz))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    for uid in boltz["target_id"].astype(str):
        receptor = args.struct_dir / uid / f"{uid}_input.pdb"
        if not nonempty(receptor):
            raise SystemExit(f"BioEmu receptor PDB is missing or empty for {uid}: {receptor}")
        target_out = args.out_dir / uid
        target_out.mkdir(parents=True, exist_ok=True)
        ok = run_bioemu(receptor, target_out, args.num_conformers)
        if not ok:
            raise SystemExit(f"BioEmu sample failed for {uid}")
        medoids = cluster_conformers(target_out, args.kmeans_k)
        if not medoids:
            raise SystemExit(f"BioEmu clustering produced no medoids for {uid}")
        for medoid in medoids:
            if not nonempty(medoid):
                raise SystemExit(f"BioEmu medoid PDB is missing or empty for {uid}: {medoid}")
            rows.append([uid, str(medoid)])
    if not rows:
        raise SystemExit("BioEmu produced no ensemble medoids")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.out_manifest.with_suffix(args.out_manifest.suffix + ".tmp")
    with tmp_manifest.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["target_id", "medoid_pdb"])
        w.writerows(rows)
    tmp_manifest.replace(args.out_manifest)
    LOG.info("Wrote %s", args.out_manifest)


if __name__ == "__main__":
    main()
