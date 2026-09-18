#!/usr/bin/env python3
"""Prepare protein-ligand systems for Stage 7 GROMACS production MD.

Protocol: minimization, position-restrained NVT/NPT equilibration
(``define = -DPOSRES`` against the ``grompp -r`` reference), then unrestrained
production. Ions are added with ``genion -neutral -conc <--ion-concentration-molar>``
so the explicit-solvent background salt matches the 0.15 M ``saltcon`` used by
the implicit-solvent MM-GBSA evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from mmgbsa_policy import SALT_CONCENTRATION_MOLAR

LOG = logging.getLogger("stage7.prep")

MANIFEST_COLUMNS = [
    "target_id",
    "complex_pose",
    "complex_pose_sha256",
    "selected_medoid_pdb",
    "selected_medoid_sha256",
    "selected_receptor_pdb",
    "selected_receptor_sha256",
    "selected_pose_sdf",
    "selected_pose_sha256",
    "ligand_topology",
    "ligand_topology_sha256",
    "parameterization_status",
    "parameterization_command",
    "system_gro",
    "topology_top",
    "tpr",
    "npt_gro",
    "prod_mdp",
    "status",
    "system_gro_sha256",
    "topology_top_sha256",
    "tpr_sha256",
    "npt_gro_sha256",
    "prod_mdp_sha256",
    "equilibration_restraints",
    "ion_concentration_molar",
]

MINIM_MDP = """; deterministic SkinScout Stage 7 minimization
integrator = steep
emtol = 1000.0
emstep = 0.01
nsteps = 50000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
"""

NVT_MDP = """; deterministic SkinScout Stage 7 NVT equilibration
; POSRES restrains protein heavy atoms to the reference structure passed via
; `grompp -r`. Without the define grompp ignores -r and this is unrestrained
; 300 K dynamics, which the manuscript must not call restrained equilibration.
define = -DPOSRES
integrator = md
dt = 0.002
nsteps = 50000
continuation = no
constraint_algorithm = lincs
constraints = h-bonds
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
tcoupl = V-rescale
tc-grps = System
tau_t = 0.1
ref_t = 300
pcoupl = no
gen_vel = yes
gen_temp = 300
gen_seed = 17391
pbc = xyz
"""

NPT_MDP = """; deterministic SkinScout Stage 7 NPT equilibration
define = -DPOSRES
integrator = md
dt = 0.002
nsteps = 50000
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
tcoupl = V-rescale
tc-grps = System
tau_t = 0.1
ref_t = 300
pcoupl = Parrinello-Rahman
pcoupltype = isotropic
tau_p = 2.0
ref_p = 1.0
compressibility = 4.5e-5
gen_vel = no
pbc = xyz
"""

# 프로덕션 mdp 에 출력 주기가 하나도 없었다. GROMACS 의 `nstxout-compressed`
# 기본값은 0 이라 궤적을 아예 쓰지 않는다 - mdrun 은 종료코드 0 으로 끝나고
# .gro/.edr/.log/.cpt 만 남는다. 궤적을 만들려고 있는 단계가 궤적을 만들지 않은
# 것이고, 그것을 먹는 MM-GBSA 는 프레임 없이 시작할 수 없다. 실패가 "0/2 복제
# 완료"로만 보여서 원인이 mdrun 쪽에 있는 것처럼 읽혔다.
#
# 10 ps 마다(5,000걸음) 좌표를 남긴다. 0.1 ns 면 10프레임, 50 ns 면 5,000프레임이라
# MM-GBSA 가 앙상블 평균을 낼 만하고, 47,000 원자 기준으로 50 ns 에 약 2 GB 다.
PROD_TRAJECTORY_STRIDE_STEPS = 5_000

PROD_MDP_TEMPLATE = """; deterministic SkinScout Stage 7 production
integrator = md
dt = 0.002
nsteps = {nsteps}
nstxout-compressed = {stride}
compressed-x-grps = System
nstenergy = {stride}
nstlog = {stride}
continuation = yes
constraint_algorithm = lincs
constraints = h-bonds
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
tcoupl = V-rescale
tc-grps = System
tau_t = 0.1
ref_t = 300
pcoupl = Parrinello-Rahman
pcoupltype = isotropic
tau_p = 2.0
ref_p = 1.0
compressibility = 4.5e-5
gen_vel = no
pbc = xyz
"""


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_nonempty(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not _nonempty(resolved):
        raise SystemExit(f"{label} is required and must be non-empty: {path}")
    return resolved


def read_receptor_manifest(path: Path,
                           chosen: dict[str, str] | None = None) -> dict[str, Path]:
    """앙상블 매니페스트에서 표적당 수용체 하나를 고른다.

    스테이지 6 은 표적 하나에 medoid 를 여러 개 낸다 - 앙상블이므로 그것이 정상
    상태다. 여기서 중복 target_id 를 거부하면 앙상블 경로 전체가 막힌다. 대신
    스테이지 6.5 의 합의가 고른 구조(`best_medoid_pdb`)를 받아 그것을 쓴다.
    합의가 그 열을 주지 않았다면 어느 구조로 MD 를 걸어야 하는지 알 수 없으므로,
    임의로 하나를 집는 대신 그 사실을 말하고 멈춘다.
    """
    if not _nonempty(path):
        raise SystemExit(f"Receptor manifest is required and must be non-empty: {path}")
    try:
        manifest = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"Receptor manifest failed to parse: {path}: {exc}") from exc
    required = {"target_id", "medoid_pdb"}
    missing_cols = sorted(required - set(manifest.columns))
    if missing_cols:
        raise SystemExit(f"Receptor manifest missing required columns {missing_cols}")
    if manifest.empty:
        raise SystemExit("Receptor manifest contains no rows")
    for col in sorted(required):
        invalid = [
            int(idx)
            for idx, value in manifest[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"Receptor manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        manifest[col] = manifest[col].astype(str).str.strip()
    duplicate_ids = sorted(
        set(manifest["target_id"][manifest["target_id"].duplicated()].tolist())
    )
    if duplicate_ids and not chosen:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(
            f"Receptor manifest has {len(duplicate_ids)} target(s) with multiple "
            f"ensemble medoids ({shown}{suffix}) and the ensemble consensus did "
            "not say which one it chose. Stage 6.5 must record "
            "'best_medoid_pdb'; picking one arbitrarily would run MD on a "
            "structure nothing selected."
        )
    paths: dict[str, Path] = {}
    for _, row in manifest.iterrows():
        uid = str(row["target_id"])
        medoid = str(row["medoid_pdb"])
        resolved_medoid = Path(medoid).expanduser().resolve(strict=False)
        resolved_choice = (
            Path(chosen[uid]).expanduser().resolve(strict=False)
            if chosen and uid in chosen
            else None
        )
        if uid in paths:
            # 합의가 고른 것만 남긴다.
            if resolved_choice is not None and resolved_choice != resolved_medoid:
                continue
        elif resolved_choice is not None and resolved_choice != resolved_medoid:
            continue
        paths[uid] = require_nonempty(
            resolved_medoid, f"Receptor medoid PDB for {uid}"
        )
    missing = sorted(set(manifest["target_id"].astype(str)) - set(paths))
    if missing:
        shown = ", ".join(missing[:10])
        raise SystemExit(
            "Ensemble consensus named a receptor that is not in the receptor "
            f"manifest for target(s): {shown}"
        )
    return paths


def read_complex_pose_manifest(path: Path) -> dict[str, Path]:
    """Read target-specific protein-ligand complex poses.

    A single complex pose cannot be safely reused across different receptors;
    the manifest makes that provenance explicit for top-N MD preparation.
    """
    if not _nonempty(path):
        raise SystemExit(f"Complex pose manifest is required and must be non-empty: {path}")
    try:
        manifest = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"Complex pose manifest failed to parse: {path}: {exc}") from exc
    pose_column = "complex_pose" if "complex_pose" in manifest.columns else "complex_pdb"
    required = {"target_id", pose_column}
    missing_cols = sorted(required - set(manifest.columns))
    if missing_cols:
        raise SystemExit(f"Complex pose manifest missing required columns {missing_cols}")
    if manifest.empty:
        raise SystemExit("Complex pose manifest contains no rows")
    for col in sorted(required):
        invalid = [
            int(idx)
            for idx, value in manifest[col].items()
            if pd.isna(value) or not str(value).strip()
        ]
        if invalid:
            shown = ", ".join(str(idx) for idx in invalid[:10])
            suffix = "..." if len(invalid) > 10 else ""
            raise SystemExit(
                f"Complex pose manifest column '{col}' contains blank values at "
                f"row index(es) {shown}{suffix}: {path}"
            )
        manifest[col] = manifest[col].astype(str).str.strip()
    duplicate_ids = manifest["target_id"][manifest["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(f"Complex pose manifest contains duplicate target_id values: {shown}{suffix}")
    paths = {
        str(row["target_id"]): require_nonempty(
            Path(str(row[pose_column])),
            f"Complex pose for {row['target_id']}",
        )
        for _, row in manifest.iterrows()
    }
    return paths


def read_ensemble_consensus(path: Path) -> pd.DataFrame:
    if not _nonempty(path):
        raise SystemExit(f"Ensemble consensus is required and must be non-empty: {path}")
    try:
        consensus = pd.read_csv(path, sep="\t", skip_blank_lines=False)
    except Exception as exc:
        raise SystemExit(f"Ensemble consensus failed to parse: {path}: {exc}") from exc
    required = {"target_id", "consensus_score"}
    missing_cols = sorted(required - set(consensus.columns))
    if missing_cols:
        raise SystemExit(f"Ensemble consensus missing required columns {missing_cols}")
    if consensus.empty:
        raise SystemExit("Ensemble consensus contains no rows")
    invalid_ids = [
        int(idx)
        for idx, value in consensus["target_id"].items()
        if pd.isna(value) or not str(value).strip()
    ]
    if invalid_ids:
        shown = ", ".join(str(idx) for idx in invalid_ids[:10])
        suffix = "..." if len(invalid_ids) > 10 else ""
        raise SystemExit(
            "Ensemble consensus column 'target_id' contains blank values at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    consensus = consensus.copy()
    consensus["target_id"] = consensus["target_id"].astype(str).str.strip()
    bool_like_scores = [
        int(idx)
        for idx, value in consensus["consensus_score"].items()
        if (
            isinstance(value, bool)
            or type(value).__name__ == "bool_"
            or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
        )
    ]
    if bool_like_scores:
        shown = ", ".join(str(idx) for idx in bool_like_scores[:10])
        suffix = "..." if len(bool_like_scores) > 10 else ""
        raise SystemExit(
            "Ensemble consensus column 'consensus_score' contains non-numeric "
            f"values at row index(es) {shown}{suffix}: {path}"
        )
    consensus["consensus_score"] = pd.to_numeric(
        consensus["consensus_score"], errors="coerce"
    )
    invalid_scores = consensus["consensus_score"].isna()
    if invalid_scores.any():
        invalid = [int(idx) for idx in consensus.index[invalid_scores]]
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            "Ensemble consensus column 'consensus_score' contains non-numeric "
            f"values at row index(es) {shown}{suffix}: {path}"
        )
    non_finite_scores = [
        int(idx)
        for idx, value in consensus["consensus_score"].items()
        if not math.isfinite(float(value))
    ]
    if non_finite_scores:
        shown = ", ".join(str(idx) for idx in non_finite_scores[:10])
        suffix = "..." if len(non_finite_scores) > 10 else ""
        raise SystemExit(
            "Ensemble consensus column 'consensus_score' contains non-finite "
            f"values at row index(es) {shown}{suffix}: {path}"
        )
    non_positive_scores = consensus["consensus_score"] <= 0
    if non_positive_scores.any():
        invalid = [int(idx) for idx in consensus.index[non_positive_scores]]
        shown = ", ".join(str(idx) for idx in invalid[:10])
        suffix = "..." if len(invalid) > 10 else ""
        raise SystemExit(
            "Ensemble consensus column 'consensus_score' must be > 0 at "
            f"row index(es) {shown}{suffix}: {path}"
        )
    duplicate_ids = consensus["target_id"][consensus["target_id"].duplicated()].tolist()
    if duplicate_ids:
        shown = ", ".join(duplicate_ids[:10])
        suffix = "..." if len(duplicate_ids) > 10 else ""
        raise SystemExit(f"Ensemble consensus contains duplicate target_id values: {shown}{suffix}")
    return consensus.sort_values("consensus_score", ascending=False)


ENSEMBLE_LINEAGE_COLUMNS = {
    "best_medoid_pdb",
    "best_medoid_sha256",
    "best_receptor_pdb",
    "best_receptor_sha256",
    "best_pose_sdf",
    "best_pose_sha256",
    "best_complex_pose",
    "best_complex_pose_sha256",
}


def verified_ensemble_lineage(consensus: pd.DataFrame) -> dict[str, dict[str, Path | str]]:
    """Verify the exact medoid, aligned receptor and docked pose selected by Stage 6."""
    missing = sorted(ENSEMBLE_LINEAGE_COLUMNS - set(consensus.columns))
    if missing:
        raise SystemExit(
            "Ensemble consensus has no immutable docking lineage; missing columns "
            f"{missing}. Re-run Stage 6 ensemble docking."
        )
    output: dict[str, dict[str, Path | str]] = {}
    for index, row in consensus.iterrows():
        target_id = str(row["target_id"])
        record: dict[str, Path | str] = {}
        for path_col, digest_col in (
            ("best_medoid_pdb", "best_medoid_sha256"),
            ("best_receptor_pdb", "best_receptor_sha256"),
            ("best_pose_sdf", "best_pose_sha256"),
            ("best_complex_pose", "best_complex_pose_sha256"),
        ):
            raw_path = str(row[path_col]).strip()
            expected = str(row[digest_col]).strip().lower()
            if not raw_path or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise SystemExit(
                    f"Ensemble consensus row {index} has invalid {path_col}/{digest_col} lineage"
                )
            resolved = require_nonempty(Path(raw_path), f"Ensemble {path_col} for {target_id}")
            actual = sha256_file(resolved)
            if actual != expected:
                raise SystemExit(
                    f"Ensemble {path_col} SHA-256 mismatch for {target_id}: {resolved}"
                )
            record[path_col] = resolved
            record[digest_col] = expected
        output[target_id] = record
    return output


def validate_ligand_topology(path: Path) -> None:
    text = path.read_text(errors="replace")
    lower = text.lower()
    required_sections = ("[ moleculetype ]", "[ atoms ]")
    missing = [section for section in required_sections if section not in lower]
    if missing:
        raise SystemExit(
            f"Ligand topology is missing required section(s) {missing}: {path}"
        )
    if "[ system ]" in lower or "[ molecules ]" in lower:
        raise SystemExit(
            "Ligand topology must be an include-compatible .itp without [ system ] "
            f"or [ molecules ] sections: {path}"
        )


def ligand_molecule_name(path: Path) -> str:
    in_section = False
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_section = line.strip("[] ").lower() == "moleculetype"
            continue
        if in_section:
            name = line.split()[0]
            if name:
                return name
    raise SystemExit(f"Ligand topology contains no molecule name: {path}")


def ligand_topology_atom_count(path: Path) -> int:
    """Count the ligand atoms declared by an include-compatible .itp."""
    in_atoms = False
    count = 0
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_atoms = line.strip("[] ").lower() == "atoms"
            continue
        if not in_atoms:
            continue
        fields = line.split()
        if fields and fields[0].isdigit():
            count += 1
    if count == 0:
        raise SystemExit(
            "Ligand topology [ atoms ] section contains no atom records: "
            f"{path}"
        )
    return count


def split_complex_pose(path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Split a bound pose into protein and one ligand residue for GROMACS.

    ``pdb2gmx`` must process the protein without an unknown ligand residue.
    The ligand coordinates are therefore carried separately and merged after
    protein hydrogenation, preserving the pose while keeping topology creation
    explicit and auditable.
    """
    pose = require_nonempty(path, "Protein-ligand complex pose")
    lines = pose.read_text(errors="replace").splitlines()
    protein_lines = [line for line in lines if line[:6].strip().upper() == "ATOM"]
    ligand_lines = [line for line in lines if line[:6].strip().upper() == "HETATM"]
    if not protein_lines:
        raise SystemExit(f"Complex pose contains no protein ATOM records: {pose}")
    if not ligand_lines:
        raise SystemExit(f"Complex pose contains no ligand HETATM records: {pose}")

    residues = {
        (
            line[21:22].strip(),
            line[22:26].strip(),
            line[26:27].strip(),
            line[17:20].strip(),
        )
        for line in ligand_lines
    }
    if len(residues) != 1:
        shown = ", ".join("/".join(item) for item in sorted(residues))
        raise SystemExit(
            "Complex pose must contain exactly one non-protein ligand residue; "
            f"found {len(residues)} ({shown}) in {pose}"
        )

    protein_pdb = out_dir / "protein_input.pdb"
    ligand_pdb = out_dir / "ligand_pose.pdb"
    protein_pdb.write_text("\n".join((*protein_lines, "TER", "END")) + "\n")
    ligand_pdb.write_text("\n".join((*ligand_lines, "END")) + "\n")
    return protein_pdb, ligand_pdb


def _ligand_topology_atom_names(ligand_topology: Path) -> list[str]:
    """ACPYPE 토폴로지의 `[ atoms ]` 이름을 선언 순서대로."""
    names: list[str] = []
    inside = False
    for line in ligand_topology.read_text(errors="replace").splitlines():
        stripped = line.split(";", 1)[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped.lower() == "[ atoms ]"
            continue
        if not inside or not stripped:
            continue
        fields = stripped.split()
        if len(fields) >= 5 and fields[0].isdigit():
            names.append(fields[4])
    return names


def require_explicit_hydrogens(ligand_sdf: Path) -> None:
    """리간드에 수소가 있는지 확인한다. 없으면 MD 는 돌지만 답이 나오지 않는다.

    Stage 1 은 두 개의 SDF 를 남긴다: `standardized.sdf`(수소 없음)와 xTB 최적화를
    거친 `compound_canonical.sdf`(수소 포함). MD 는 후자를 받아야 한다.

    전자를 주면 어디서도 멈추지 않는다. ACPYPE 는 중원자만의 토폴로지를 만들고,
    GROMACS 는 그것으로 용매화와 평형화와 프로덕션 MD 를 끝까지 돈다. 실측으로
    세 표적이 각각 2복제씩 6개 궤적을 만들었다. 그리고 그 뒤 MM-GBSA 가
    "Some energy terms are undefined" 로 멈춘다 - 수소 없는 리간드에는 정의되지
    않는 항이 있기 때문이다. 몇 시간을 쓰고 나서야, 그것도 원인을 말해 주지 않는
    메시지로 드러난다.

    수소는 MD 의 선택 사항이 아니다. 시작할 때 확인한다.
    """
    from rdkit import Chem

    mol = Chem.MolFromMolFile(str(ligand_sdf), removeHs=False)
    if mol is None:
        raise SystemExit(f"Ligand SDF could not be read: {ligand_sdf}")
    n_hydrogen = mol.GetNumAtoms() - mol.GetNumHeavyAtoms()
    if n_hydrogen == 0 and mol.GetNumHeavyAtoms() > 1:
        raise SystemExit(
            f"Ligand SDF has no explicit hydrogens ({mol.GetNumHeavyAtoms()} "
            f"heavy atoms, 0 hydrogens): {ligand_sdf}. MD needs the "
            "hydrogen-bearing structure - use the Stage 1 xTB-optimised SDF "
            "(compound_canonical.sdf), not standardized.sdf. Without them the "
            "run completes and MM-GBSA then fails with undefined energy terms."
        )


def align_pose_to_topology(ligand_pdb: Path, reference_sdf: Path,
                       ligand_topology: Path | None = None) -> int:
    """포즈를 토폴로지의 원자 순서·이름에 맞춰 다시 쓴다. 필요하면 수소도 채운다.

    공동 접힘(Boltz-2)과 도킹은 리간드를 **중원자만** 내놓는다. ACPYPE 토폴로지는
    Stage 1 의 SDF 에서 만들어져 수소를 포함한다 - 아다팔렌에서 31 대 58 이었다.
    두 수가 다르면 MD 준비가 정직하게 멈추지만, 멈추기만 하면 앙상블 경로가
    거기서 끝난다.

    수소를 새로 만들지 않는다. Stage 1 의 SDF 를 그대로 쓰되 그 **중원자 좌표만**
    포즈의 것으로 바꾸고, 중원자를 붙든 채 수소 위치만 MMFF 로 푼다. 그래서
    - 원자 순서가 토폴로지와 같고(같은 SDF 에서 나왔다),
    - 결합한 자세는 포즈 그대로이며(중원자 편차 0 을 단언한다),
    - 수소만 그 자세에 맞게 놓인다.

    돌려주는 값은 쓴 원자 수다.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    reference = Chem.MolFromMolFile(str(reference_sdf), removeHs=False)
    if reference is None or reference.GetNumConformers() == 0:
        raise SystemExit(
            f"Ligand reference SDF could not be read with 3D coordinates: {reference_sdf}"
        )
    pose_block = ligand_pdb.read_text(errors="replace")
    pose = Chem.MolFromPDBBlock(pose_block, removeHs=False, sanitize=False)
    if pose is None:
        raise SystemExit(f"Ligand pose PDB could not be read: {ligand_pdb}")

    heavy_reference = Chem.RemoveHs(Chem.Mol(reference))
    if pose.GetNumAtoms() != heavy_reference.GetNumAtoms():
        raise SystemExit(
            f"Ligand pose has {pose.GetNumAtoms()} atoms but the Stage 1 ligand "
            f"has {heavy_reference.GetNumAtoms()} heavy atoms: {ligand_pdb}. "
            "These must be the same molecule."
        )
    try:
        # PDB 에는 결합 차수가 없다. 참조 분자에서 이식해야 부분구조 매칭이 선다.
        pose = AllChem.AssignBondOrdersFromTemplate(heavy_reference, pose)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(
            f"Ligand pose does not match the Stage 1 ligand: {ligand_pdb}: {exc}"
        ) from exc
    match = pose.GetSubstructMatch(heavy_reference)
    if len(match) != heavy_reference.GetNumAtoms():
        raise SystemExit(
            f"Ligand pose could not be mapped onto the Stage 1 ligand: {ligand_pdb}"
        )

    heavy_indices = [
        atom.GetIdx() for atom in reference.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    conformer = reference.GetConformer()
    pose_conformer = pose.GetConformer()
    for position, reference_index in enumerate(heavy_indices):
        point = pose_conformer.GetAtomPosition(match[position])
        conformer.SetAtomPosition(
            reference_index, Point3D(point.x, point.y, point.z)
        )

    # 수소 위치만 푼다. 참조 분자에 수소가 없으면(카페인의 Stage 1 SDF 가
    # 그렇다) 고정할 것이 전부라 움직일 원자가 없고, RDKit 의 BFGS 가
    # "Failed Expression: status >= 0" 으로 죽는다. 그때는 최적화할 것이
    # 없으므로 건너뛰는 것이 맞다.
    movable = reference.GetNumAtoms() - len(heavy_indices)
    properties = (
        AllChem.MMFFGetMoleculeProperties(reference) if movable else None
    )
    if properties is not None:
        field = AllChem.MMFFGetMoleculeForceField(reference, properties)
        for index in heavy_indices:
            field.AddFixedPoint(index)
        field.Minimize(maxIts=2000)

    # 중원자가 정말로 포즈 그대로인지 확인한다. 여기가 어긋나면 MD 는 도킹이
    # 찾은 자세가 아니라 다른 자세에서 출발한다.
    for position, reference_index in enumerate(heavy_indices):
        a = conformer.GetAtomPosition(reference_index)
        b = pose_conformer.GetAtomPosition(match[position])
        if max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z)) > 1e-3:
            raise SystemExit(
                f"Adding hydrogens moved a heavy atom of the ligand pose: {ligand_pdb}"
            )

    resname = "LIG"
    chain = "L"
    for line in pose_block.splitlines():
        if line.startswith(("HETATM", "ATOM")):
            resname = line[17:20].strip() or resname
            chain = line[21:22].strip() or chain
            break
    # 원자 이름은 **토폴로지에서** 가져온다. 직접 지어내면 grompp 가
    # "non-matching atom names" 경고를 원자 수만큼 쏟아내고, 이온 넣기 단계의
    # -maxwarn 1 을 넘겨 실행이 멈춘다. 그리고 경고를 늘려 넘기는 것은 이름이
    # 어긋난 채로 좌표와 토폴로지를 짝지어도 좋다는 뜻이 되므로 하지 않는다.
    topology_names = (
        _ligand_topology_atom_names(ligand_topology) if ligand_topology else []
    )
    if topology_names and len(topology_names) != reference.GetNumAtoms():
        raise SystemExit(
            f"Ligand topology declares {len(topology_names)} atom names but the "
            f"Stage 1 ligand has {reference.GetNumAtoms()} atoms: {ligand_topology}"
        )
    rows = []
    for index, atom in enumerate(reference.GetAtoms(), start=1):
        point = conformer.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        name = (topology_names[index - 1] if topology_names
                else f"{symbol}{index}")[:4]
        rows.append(
            f"HETATM{index:5d} {name:<4s} {resname:>3s} {chain:>1s}   1    "
            f"{point.x:8.3f}{point.y:8.3f}{point.z:8.3f}  1.00  0.00"
            f"          {symbol:>2s}"
        )
    tmp = ligand_pdb.with_suffix(".pdb.tmp")
    tmp.write_text("\n".join((*rows, "END")) + "\n")
    tmp.replace(ligand_pdb)
    return len(rows)


def _read_gro_atoms(path: Path) -> tuple[list[str], str]:
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 3:
        raise SystemExit(f"GROMACS coordinate file is malformed: {path}")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise SystemExit(f"GROMACS coordinate atom count is invalid: {path}") from exc
    atom_lines = lines[2 : 2 + atom_count]
    if atom_count <= 0 or len(atom_lines) != atom_count:
        raise SystemExit(f"GROMACS coordinate atom records are incomplete: {path}")
    box_line = lines[2 + atom_count] if len(lines) > 2 + atom_count else "0.00000 0.00000 0.00000"
    return atom_lines, box_line


def _pdb_xyz_angstrom(line: str, path: Path) -> tuple[float, float, float]:
    try:
        xyz = tuple(float(line[start:end]) / 10.0 for start, end in ((30, 38), (38, 46), (46, 54)))
    except ValueError as exc:
        raise SystemExit(f"Ligand PDB contains invalid coordinates: {path}") from exc
    if not all(math.isfinite(value) for value in xyz):
        raise SystemExit(f"Ligand PDB contains non-finite coordinates: {path}")
    return xyz


def merge_protein_and_ligand_gro(
    protein_gro: Path,
    ligand_pdb: Path,
    out_gro: Path,
    expected_ligand_atoms: int,
) -> None:
    """Create a pose-preserving GRO with processed protein plus ligand atoms."""
    protein_atoms, box_line = _read_gro_atoms(protein_gro)
    ligand_lines = [
        line for line in ligand_pdb.read_text(errors="replace").splitlines()
        if line[:6].strip().upper() == "HETATM"
    ]
    if len(ligand_lines) != expected_ligand_atoms:
        raise SystemExit(
            "Ligand coordinate/topology atom-count mismatch: complex pose has "
            f"{len(ligand_lines)} atoms but topology declares {expected_ligand_atoms}: "
            f"{ligand_pdb}"
        )

    merged: list[str] = []
    for atom_number, line in enumerate(protein_atoms, start=1):
        if len(line) < 20:
            raise SystemExit(f"Processed protein GRO atom line is malformed: {protein_gro}")
        merged.append(f"{line[:15]}{atom_number:5d}{line[20:]}")

    for offset, line in enumerate(ligand_lines, start=len(protein_atoms) + 1):
        residue_name = line[17:20].strip()[:5] or "LIG"
        atom_name = line[12:16].strip()[:5] or f"C{offset}"
        x, y, z = _pdb_xyz_angstrom(line, ligand_pdb)
        merged.append(
            f"{1:5d}{residue_name:<5}{atom_name:>5}{offset:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )

    out_gro.parent.mkdir(parents=True, exist_ok=True)
    out_gro.write_text(
        "SkinScout processed protein-ligand pose\n"
        f"{len(merged)}\n"
        + "\n".join(merged)
        + "\n"
        + (box_line.strip() or "0.00000 0.00000 0.00000")
        + "\n"
    )


def require_tool(command: str, label: str) -> str:
    resolved = shutil.which(command)
    if not resolved:
        raise SystemExit(f"{label} command is not available on PATH: {command}")
    return command


def run_command(cmd: list[str], cwd: Path, input_text: str | None = None) -> str:
    LOG.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
            + (f"\n{detail}" if detail else "")
        )
    return " ".join(cmd)


def write_mdp_templates(target_dir: Path, duration_ns: float) -> dict[str, Path]:
    # `--duration-ns` 는 float 이라 곱하면 float 이 되고, GROMACS 는 mdp 의
    # `nsteps` 에 실수를 받지 않는다("Right hand side '50000.0' ... is not an
    # integer value"). 최소화·NVT·NPT 를 다 돌고 나서 프로덕션 grompp 에서
    # 멈추므로, 실패가 몇 분 뒤에 온다.
    #
    # 2 fs 걸음이므로 1 ns = 500,000 걸음이다. 걸음 하나에 못 미치는 시간은
    # 요청한 것보다 짧게 도는 것이니 올림하고, 0 걸음은 거부한다.
    steps = math.ceil(duration_ns * 500_000)
    if steps < 1:
        raise SystemExit(
            f"--duration-ns must be long enough for at least one 2 fs step: {duration_ns}"
        )
    nsteps = int(steps)
    templates = {
        "ions": MINIM_MDP,
        "minim": MINIM_MDP,
        "nvt": NVT_MDP,
        "npt": NPT_MDP,
        "prod": PROD_MDP_TEMPLATE.format(
            nsteps=nsteps,
            # 짧은 시험 실행에서도 프레임이 최소 하나는 나와야 한다.
            stride=min(PROD_TRAJECTORY_STRIDE_STEPS, max(1, nsteps // 10)),
        ),
    }
    paths: dict[str, Path] = {}
    for name, content in templates.items():
        path = target_dir / f"{name}.mdp"
        path.write_text(content)
        paths[name] = path
    return paths


def _split_ligand_atomtypes(ligand_itp: Path) -> tuple[Path, Path]:
    """ACPYPE ITP 를 `[ atomtypes ]` 와 나머지로 가른다.

    GROMACS 는 `[ atomtypes ]` 가 어떤 `[ moleculetype ]` 보다 **앞에** 오기를
    요구한다. ACPYPE 는 둘을 한 파일에 담아 주는데, 그 파일을 단백질
    `[ moleculetype ]` 뒤에 include 하면 `Invalid order for directive atomtypes`
    로 멈춘다. 그래서 두 조각으로 갈라 각각 제자리에 넣는다.
    """
    lines = ligand_itp.read_text(errors="replace").splitlines()
    atomtypes: list[str] = []
    rest: list[str] = []
    target = rest
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("[") and stripped.endswith("]"):
            target = atomtypes if stripped == "[ atomtypes ]" else rest
        target.append(line)
    atomtypes_itp = ligand_itp.with_name(f"{ligand_itp.stem}_atomtypes.itp")
    rest_itp = ligand_itp.with_name(f"{ligand_itp.stem}_moleculetype.itp")
    atomtypes_itp.write_text("\n".join(atomtypes).rstrip() + "\n")
    rest_itp.write_text("\n".join(rest).rstrip() + "\n")
    return atomtypes_itp, rest_itp


def append_ligand_include(topology: Path, ligand_topology: Path) -> Path:
    ligand_copy = topology.parent / ligand_topology.name
    if ligand_topology.resolve(strict=False) != ligand_copy.resolve(strict=False):
        shutil.copyfile(ligand_topology, ligand_copy)
    atomtypes_itp, rest_itp = _split_ligand_atomtypes(ligand_copy)
    text = topology.read_text(errors="replace")
    include_line = f'#include "{rest_itp.name}"'
    atomtypes_line = f'#include "{atomtypes_itp.name}"'
    molecule_name = ligand_molecule_name(ligand_topology)
    if atomtypes_line not in text:
        lines = text.splitlines()
        # 힘장 include 바로 뒤. 여기가 첫 [ moleculetype ] 보다 앞이다.
        forcefield_index = next(
            (idx for idx, line in enumerate(lines)
             if line.strip().startswith("#include") and "forcefield.itp" in line),
            None,
        )
        if forcefield_index is None:
            raise SystemExit(
                "Main GROMACS topology has no forcefield include; cannot place "
                f"the ligand atomtypes before the first moleculetype: {topology}"
            )
        lines.insert(forcefield_index + 1, atomtypes_line)
        text = "\n".join(lines) + "\n"
    if include_line not in text:
        lines = text.splitlines()
        system_index = next(
            (idx for idx, line in enumerate(lines) if line.strip().lower() == "[ system ]"),
            0,
        )
        lines.insert(system_index, include_line)
        text = "\n".join(lines) + "\n"
    lowered = text.lower()
    if "[ molecules ]" not in lowered:
        text = text.rstrip() + "\n\n[ molecules ]\n; molecule_name  count\n"
    molecules_index = next(
        (idx for idx, line in enumerate(text.splitlines()) if line.strip().lower() == "[ molecules ]"),
        None,
    )
    lines = text.splitlines()
    if molecules_index is None:
        raise SystemExit(f"Main GROMACS topology lost its [ molecules ] section: {topology}")
    existing = any(
        line.split(";", 1)[0].split()[0:1] == [molecule_name]
        for line in lines[molecules_index + 1 :]
        if line.split(";", 1)[0].split()
    )
    if not existing:
        lines.append(f"{molecule_name:<16} 1")
    topology.write_text("\n".join(lines) + "\n")
    return ligand_copy


def prepare_ligand_topology(
    args: argparse.Namespace,
    target_dir: Path,
    acpype_cmd: str | None,
) -> tuple[Path, str, str]:
    """Return an include-compatible ligand topology and its provenance."""
    if args.ligand_topology is not None:
        topology = require_nonempty(args.ligand_topology, "Ligand topology")
        validate_ligand_topology(topology)
        return topology, "provided_topology", "operator-provided topology"

    if acpype_cmd is None:
        raise SystemExit("ACPYPE is required when --ligand-topology is not supplied")
    ligand_sdf = require_nonempty(args.ligand_sdf, "Ligand SDF")
    # ACPYPE writes fixed names. Run each attempt in a fresh directory so a
    # successful exit that produced nothing cannot pick up ligand.acpype from
    # an earlier attempt.
    with tempfile.TemporaryDirectory(dir=target_dir, prefix=".acpype-attempt-") as raw_attempt:
        attempt = Path(raw_attempt)
        command = [acpype_cmd, "-i", str(ligand_sdf), "-b", "ligand", "-o", "gmx"]
        command_text = run_command(command, cwd=attempt)
        candidates = [
            attempt / "ligand.acpype" / "ligand_GMX.itp",
            attempt / "ligand_GMX.itp",
        ]
        generated = next((path for path in candidates if _nonempty(path)), None)
        if generated is None:
            shown = ", ".join(str(path) for path in candidates)
            raise SystemExit(
                "ACPYPE completed without a topology from the current isolated attempt; "
                f"checked: {shown}"
            )
        validate_ligand_topology(generated)
        published = target_dir / "ligand_GMX.itp"
        temporary = published.with_suffix(published.suffix + f".tmp.{os.getpid()}")
        shutil.copyfile(generated, temporary)
        temporary.replace(published)
    validate_ligand_topology(published)
    return published, "acpype_generated", command_text


def prepare_target(
    target_id: str,
    args: argparse.Namespace,
    gmx_cmd: str,
    acpype_cmd: str | None,
    complex_pose: Path,
    ligand_topology: Path | None,
    lineage: dict[str, Path | str] | None = None,
) -> dict[str, str]:
    target_dir = (args.out_dir / target_id).resolve(strict=False)
    target_dir.mkdir(parents=True, exist_ok=True)
    mdp = write_mdp_templates(target_dir, args.duration_ns)
    topology = target_dir / "topol.top"
    system_gro = target_dir / "system.gro"
    prod_tpr = target_dir / "prod.tpr"

    resolved_ligand_topology, parameterization_status, parameterization_command = (
        prepare_ligand_topology(args, target_dir, acpype_cmd)
    )
    ligand_atom_count = ligand_topology_atom_count(resolved_ligand_topology)
    protein_pdb, ligand_pdb = split_complex_pose(complex_pose, target_dir)
    # 포즈를 **언제나** 토폴로지에 맞춰 다시 쓴다. 개수가 다를 때만 손대면
    # 안 된다: 공동 접힘이 붙인 이름(C22, N19, ...)과 ACPYPE 가 붙인 이름
    # (C1, N1, ...)은 다른데, 개수가 우연히 같으면(카페인은 양쪽 14) 이 코드가
    # 건너뛴다. 그러면 좌표와 토폴로지가 **파일 순서로만** 짝지어지고, 두 도구가
    # 원자를 같은 순서로 나열할 이유는 없다. GROMACS 는 "non-matching atom
    # names ... atom names from topol.top will be used" 라고 경고만 하고 계속
    # 간다 - 탄소 자리에 질소 좌표가 들어간 계가 조용히 돌아간다.
    #
    # RDKit 부분구조 매칭으로 맞추므로 순서가 아니라 분자 구조가 대응을 정하고,
    # 수소가 모자라면 그때 함께 채워진다.
    pose_atoms = sum(
        1 for line in ligand_pdb.read_text(errors="replace").splitlines()
        if line[:6].strip().upper() == "HETATM"
    )
    written = align_pose_to_topology(
        ligand_pdb, args.ligand_sdf, resolved_ligand_topology
    )
    if written != ligand_atom_count:
        raise SystemExit(
            f"Ligand pose has {written} atoms after alignment but the topology "
            f"declares {ligand_atom_count}: {ligand_pdb}"
        )
    LOG.info(
        "Ligand pose had %d atoms; rewrote it against the topology's %d "
        "(matched by substructure, not by file order)",
        pose_atoms, written,
    )
    _validate_forcefield_selection(args)
    run_command(
        [
            gmx_cmd,
            "pdb2gmx",
            "-f",
            str(protein_pdb),
            "-o",
            str(target_dir / "protein_processed.gro"),
            "-p",
            str(topology),
            "-ff",
            args.forcefield,
            "-water",
            args.water_model,
        ],
        cwd=target_dir,
    )
    if not _nonempty(topology):
        raise SystemExit(f"pdb2gmx did not create a non-empty topology: {topology}")
    included_ligand_topology = append_ligand_include(topology, resolved_ligand_topology)
    merge_protein_and_ligand_gro(
        target_dir / "protein_processed.gro",
        ligand_pdb,
        target_dir / "complex_processed.gro",
        ligand_atom_count,
    )
    run_command(
        [
            gmx_cmd,
            "editconf",
            "-f",
            str(target_dir / "complex_processed.gro"),
            "-o",
            str(target_dir / "boxed.gro"),
            "-c",
            "-d",
            str(args.box_distance_nm),
            "-bt",
            args.box_type,
        ],
        cwd=target_dir,
    )
    run_command(
        [
            gmx_cmd,
            "solvate",
            "-cp",
            str(target_dir / "boxed.gro"),
            "-cs",
            args.solvent_model,
            "-o",
            str(target_dir / "solvated.gro"),
            "-p",
            str(topology),
        ],
        cwd=target_dir,
    )
    # 용매화가 끝나면 계의 크기가 정해진다. 그 수를 여기서 남겨야, 104만 원자
    # 짜리 계가 몇 시간 뒤 GPU 메모리 오류로 드러나는 대신 시작할 때 보인다.
    # 실측으로 표적에 따라 4.1만에서 104만까지 25배 차이가 났다 - 펼쳐진
    # 구조는 상자를 훨씬 크게 만든다.
    solvated = target_dir / "solvated.gro"
    if _nonempty(solvated):
        try:
            n_atoms = int(solvated.read_text(errors="replace").splitlines()[1].strip())
        except (IndexError, ValueError):
            n_atoms = -1
        box = solvated.read_text(errors="replace").splitlines()[-1].split()
        LOG.info("Solvated system for %s: %d atoms, box %s nm",
                 target_dir.name, n_atoms, " x ".join(box[:3]))
    run_command(
        [
            gmx_cmd,
            "grompp",
            "-f",
            str(mdp["ions"]),
            "-c",
            str(target_dir / "solvated.gro"),
            "-p",
            str(topology),
            "-o",
            str(target_dir / "ions.tpr"),
            "-maxwarn",
            str(args.maxwarn),
        ],
        cwd=target_dir,
    )
    run_command(
        [
            gmx_cmd,
            "genion",
            "-s",
            str(target_dir / "ions.tpr"),
            "-o",
            str(system_gro),
            "-p",
            str(topology),
            "-pname",
            args.positive_ion,
            "-nname",
            args.negative_ion,
            "-neutral",
            "-conc",
            f"{args.ion_concentration_molar:g}",
        ],
        cwd=target_dir,
        input_text=f"{args.solvent_group}\n",
    )
    run_command(
        [
            gmx_cmd,
            "grompp",
            "-f",
            str(mdp["minim"]),
            "-c",
            str(system_gro),
            "-p",
            str(topology),
            "-o",
            str(target_dir / "em.tpr"),
            "-maxwarn",
            str(args.maxwarn),
        ],
        cwd=target_dir,
    )
    run_command([gmx_cmd, "mdrun", "-deffnm", str(target_dir / "em")], cwd=target_dir)
    run_command(
        [
            gmx_cmd,
            "grompp",
            "-f",
            str(mdp["nvt"]),
            "-c",
            str(target_dir / "em.gro"),
            "-r",
            str(system_gro),
            "-p",
            str(topology),
            "-o",
            str(target_dir / "nvt.tpr"),
            "-maxwarn",
            str(args.maxwarn),
        ],
        cwd=target_dir,
    )
    run_command([gmx_cmd, "mdrun", "-deffnm", str(target_dir / "nvt")], cwd=target_dir)
    run_command(
        [
            gmx_cmd,
            "grompp",
            "-f",
            str(mdp["npt"]),
            "-c",
            str(target_dir / "nvt.gro"),
            "-r",
            str(system_gro),
            "-p",
            str(topology),
            "-o",
            str(target_dir / "npt.tpr"),
            "-maxwarn",
            str(args.maxwarn),
        ],
        cwd=target_dir,
    )
    run_command([gmx_cmd, "mdrun", "-deffnm", str(target_dir / "npt")], cwd=target_dir)
    run_command(
        [
            gmx_cmd,
            "grompp",
            "-f",
            str(mdp["prod"]),
            "-c",
            str(target_dir / "npt.gro"),
            "-p",
            str(topology),
            "-o",
            str(prod_tpr),
            "-maxwarn",
            str(args.maxwarn),
        ],
        cwd=target_dir,
    )

    for label, path in (
        ("system.gro", system_gro),
        ("topol.top", topology),
        ("prod.tpr", prod_tpr),
        ("npt.gro", target_dir / "npt.gro"),
        ("prod.mdp", mdp["prod"]),
    ):
        if not _nonempty(path):
            raise SystemExit(f"Prepared {label} is missing or empty for {target_id}: {path}")

    return {
        "target_id": target_id,
        "complex_pose": str(complex_pose),
        "complex_pose_sha256": sha256_file(complex_pose),
        "selected_medoid_pdb": str((lineage or {}).get("best_medoid_pdb", "")),
        "selected_medoid_sha256": str((lineage or {}).get("best_medoid_sha256", "")),
        "selected_receptor_pdb": str((lineage or {}).get("best_receptor_pdb", "")),
        "selected_receptor_sha256": str((lineage or {}).get("best_receptor_sha256", "")),
        "selected_pose_sdf": str((lineage or {}).get("best_pose_sdf", "")),
        "selected_pose_sha256": str((lineage or {}).get("best_pose_sha256", "")),
        "ligand_topology": str(included_ligand_topology),
        "ligand_topology_sha256": sha256_file(included_ligand_topology),
        "parameterization_status": parameterization_status,
        "parameterization_command": parameterization_command,
        "system_gro": str(system_gro),
        "topology_top": str(topology),
        "tpr": str(prod_tpr),
        "npt_gro": str(target_dir / "npt.gro"),
        "prod_mdp": str(mdp["prod"]),
        "status": "ready",
        "system_gro_sha256": sha256_file(system_gro),
        "topology_top_sha256": sha256_file(topology),
        "tpr_sha256": sha256_file(prod_tpr),
        "npt_gro_sha256": sha256_file(target_dir / "npt.gro"),
        "prod_mdp_sha256": sha256_file(mdp["prod"]),
        # 문구가 아니라 파일이 조건을 말하게 한다. NVT/NPT mdp 의 -DPOSRES 와
        # genion 이 실제로 넣은 염 농도가 매니페스트에 함께 남는다.
        "equilibration_restraints": "POSRES",
        "ion_concentration_molar": f"{args.ion_concentration_molar:g}",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_diagnostic_empty_manifest(path: Path) -> None:
    write_manifest(
        path,
        [
            {
                "target_id": "diagnostic_empty",
                "complex_pose": "",
                "complex_pose_sha256": "",
                "selected_medoid_pdb": "",
                "selected_medoid_sha256": "",
                "selected_receptor_pdb": "",
                "selected_receptor_sha256": "",
                "selected_pose_sdf": "",
                "selected_pose_sha256": "",
                "ligand_topology": "",
                "ligand_topology_sha256": "",
                "parameterization_status": "diagnostic_empty",
                "parameterization_command": "",
                "system_gro": "",
                "topology_top": "",
                "tpr": "",
                "npt_gro": "",
                "prod_mdp": "",
                "status": "diagnostic_empty",
                "system_gro_sha256": "",
                "topology_top_sha256": "",
                "tpr_sha256": "",
                "npt_gro_sha256": "",
                "prod_mdp_sha256": "",
            }
        ],
    )


def remove_stale_manifest(path: Path) -> None:
    path.unlink(missing_ok=True)


def fail_closed(message: str) -> SystemExit:
    return SystemExit(f"Stage7 GROMACS preparation is fail-closed: {message}")


SUPPORTED_LIGAND_FF = {"acpype_gaff2", "gaff2", "gaff", "acpype"}
SUPPORTED_TIMESTEPS = {1.0, 2.0}


def _gromacs_top_dir(gmx_command: str) -> Path | None:
    env = os.environ.get("GMXDATA")
    if env and (Path(env) / "top").is_dir():
        return Path(env) / "top"
    import shutil

    exe = shutil.which(gmx_command)
    if not exe:
        return None
    candidate = Path(exe).resolve().parent.parent / "share" / "gromacs" / "top"
    return candidate if candidate.is_dir() else None


def _validate_forcefield_selection(args) -> None:
    """설치본에 없는 force field/미지원 설정은 실행 전에 fail-fast한다(D04)."""
    top = _gromacs_top_dir(args.gmx_command)
    if top is None:
        raise SystemExit(
            f"GROMACS share/top 디렉터리를 찾을 수 없습니다(gmx={args.gmx_command!r}); GMXDATA를 설정하세요."
        )
    if not (top / f"{args.forcefield}.ff").is_dir():
        available = sorted(path.name[: -len(".ff")] for path in top.glob("*.ff"))
        raise SystemExit(
            f"forcefield {args.forcefield!r}가 설치본에 없습니다. 사용 가능: {available}"
        )
    if args.ligand_ff not in SUPPORTED_LIGAND_FF:
        raise SystemExit(
            f"ligand_ff {args.ligand_ff!r}는 아직 구현되지 않았습니다: {sorted(SUPPORTED_LIGAND_FF)}"
        )
    if args.timestep_fs not in SUPPORTED_TIMESTEPS:
        raise SystemExit(
            f"timestep_fs {args.timestep_fs!r}는 아직 구현되지 않았습니다: {sorted(SUPPORTED_TIMESTEPS)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-consensus", required=True, type=Path)
    parser.add_argument("--receptor-manifest", required=True, type=Path)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--complex-pose", required=False, type=Path)
    parser.add_argument("--complex-pose-manifest", required=False, type=Path)
    parser.add_argument("--ligand-topology", required=False, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--duration-ns", type=float, default=50.0)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--gmx-command", default="gmx")
    parser.add_argument("--acpype-command", default="acpype")
    parser.add_argument("--forcefield", default="amber99sb-ildn")
    parser.add_argument("--ligand-ff", default="acpype_gaff2")
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--water-model", default="tip3p")
    parser.add_argument("--solvent-model", default="spc216.gro")
    # 십이면체가 기본이다. 같은 최소 거리에서 정육면체보다 부피가 ~71% 이고,
    # 물 분자가 그만큼 줄어 계산도 그만큼 준다. 과학적으로 잃는 것은 없다.
    #
    # 실측으로 이것이 실행 가능성을 갈랐다: Q14393(GAS6)의 BioEmu 구조가
    # 166.9 A 로 길게 펼쳐져 정육면체 상자가 22 nm 가 되고 물이 채워져 계가
    # 104만 원자가 됐다(P11086 은 4.1만). 그 크기에서 GPU 메모리가 터졌다.
    # 십이면체로는 10,614 → 7,505 nm^3 다.
    parser.add_argument("--box-type", default="dodecahedron")
    parser.add_argument("--box-distance-nm", type=float, default=1.0)
    parser.add_argument("--positive-ion", default="NA")
    parser.add_argument("--negative-ion", default="CL")
    parser.add_argument("--solvent-group", default="SOL")
    # Stage 7 의 MM-GBSA 평가는 saltcon=0.15 로 돈다. 명시적 용매 상자도 같은
    # 배경 염 농도로 중화해야 샘플링과 평가 조건이 어긋나지 않는다.
    parser.add_argument(
        "--ion-concentration-molar",
        type=float,
        default=SALT_CONCENTRATION_MOLAR,
        help=(
            "Background salt concentration added by genion "
            f"(default: {SALT_CONCENTRATION_MOLAR:g}); matches the MM-GBSA saltcon."
        ),
    )
    parser.add_argument("--maxwarn", type=int, default=1)
    parser.add_argument(
        "--diagnostic-empty-status",
        action="store_true",
        help="Write an explicit diagnostic_empty manifest instead of running prep.",
    )
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help="Deprecated compatibility flag; use --diagnostic-empty-status.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.allow_partial_output and not args.diagnostic_empty_status:
        remove_stale_manifest(args.out_manifest)
        raise fail_closed(
            "--allow-partial-output is no longer supported; use "
            "--diagnostic-empty-status"
        )
    if args.diagnostic_empty_status:
        write_diagnostic_empty_manifest(args.out_manifest)
        LOG.info("Wrote diagnostic empty manifest %s", args.out_manifest)
        return
    if args.top_n < 1:
        remove_stale_manifest(args.out_manifest)
        raise fail_closed(f"--top-n must be >= 1: {args.top_n}")
    if args.duration_ns <= 0:
        remove_stale_manifest(args.out_manifest)
        raise fail_closed(f"--duration-ns must be >= 1: {args.duration_ns}")
    if args.maxwarn < 0:
        remove_stale_manifest(args.out_manifest)
        raise fail_closed(f"--maxwarn must be >= 0: {args.maxwarn}")
    if not math.isfinite(args.ion_concentration_molar) or args.ion_concentration_molar < 0:
        remove_stale_manifest(args.out_manifest)
        raise fail_closed(
            "--ion-concentration-molar must be a finite value >= 0: "
            f"{args.ion_concentration_molar}"
        )

    # Validate ordinary upstream inputs first so malformed current inputs remove
    # any stale manifest. A missing complex pose is a separate external gate;
    # preserve an existing diagnostic file until that gate is supplied.
    try:
        consensus = read_ensemble_consensus(args.ensemble_consensus)
        require_nonempty(
            args.ligand_sdf,
            "Ligand SDF is missing or empty for GROMACS prep",
        )
        # 상류 입력 검증과 같은 자리다. 더 앞에 두면 인자가 잘못된 경우까지
        # 이 검사가 가로채, 실제 문제(빈 합의 파일 등)를 말하지 못한다.
        require_explicit_hydrogens(args.ligand_sdf)
    except SystemExit:
        remove_stale_manifest(args.out_manifest)
        raise

    if args.complex_pose is not None and args.complex_pose_manifest is not None:
        remove_stale_manifest(args.out_manifest)
        raise fail_closed("Use only one of --complex-pose and --complex-pose-manifest")

    try:
        # 합의가 표적당 어느 구조를 골랐는지 먼저 읽는다. 앙상블이므로 수용체
        # 매니페스트에는 표적당 여러 행이 있는 것이 정상이다.
        consensus_choice = read_ensemble_consensus(args.ensemble_consensus)
        chosen = (
            {
                str(row["target_id"]): str(row["best_medoid_pdb"]).strip()
                for _, row in consensus_choice.iterrows()
                if str(row.get("best_medoid_pdb") or "").strip()
            }
            if "best_medoid_pdb" in consensus_choice.columns
            else {}
        )
        lineage_columns = ENSEMBLE_LINEAGE_COLUMNS & set(consensus_choice.columns)
        lineage = (
            verified_ensemble_lineage(consensus_choice)
            if lineage_columns
            else None
        )
        if lineage is None and args.complex_pose is None and args.complex_pose_manifest is None:
            raise fail_closed(
                "--complex-pose or --complex-pose-manifest is required for legacy "
                "consensus without Stage 6 docked-pose lineage; re-run Stage 6"
            )
        if lineage is not None and (
            args.complex_pose is not None or args.complex_pose_manifest is not None
        ):
            raise fail_closed(
                "do not override verified Stage 6 docking lineage with --complex-pose options"
            )
        receptor_paths = read_receptor_manifest(args.receptor_manifest, chosen)
        pose_manifest = (
            read_complex_pose_manifest(args.complex_pose_manifest)
            if args.complex_pose_manifest is not None
            else None
        )
    except SystemExit:
        remove_stale_manifest(args.out_manifest)
        raise
    try:
        single_pose = (
            require_nonempty(args.complex_pose, "Protein-ligand complex pose")
            if args.complex_pose is not None
            else None
        )
    except SystemExit:
        remove_stale_manifest(args.out_manifest)
        raise
    remove_stale_manifest(args.out_manifest)
    if single_pose is not None and len(consensus.head(args.top_n)) != 1:
        raise fail_closed(
            "A single --complex-pose may only be used with --top-n 1; provide "
            "--complex-pose-manifest for top-N preparation"
        )
    gmx_cmd = require_tool(args.gmx_command, "GROMACS")
    acpype_cmd = (
        require_tool(args.acpype_command, "ACPYPE")
        if args.ligand_topology is None
        else None
    )

    selected = consensus.head(args.top_n)["target_id"].astype(str).tolist()
    rows: list[dict[str, str]] = []
    for target_id in selected:
        if target_id not in receptor_paths:
            raise SystemExit(
                "Receptor manifest has no medoid PDB for selected target_id: "
                f"{target_id}"
            )
        lineage_record = lineage.get(target_id) if lineage is not None else None
        if lineage_record is not None:
            selected_medoid = Path(str(lineage_record["best_medoid_pdb"]))
            if receptor_paths[target_id] != selected_medoid:
                raise SystemExit(
                    f"Selected medoid lineage differs from receptor manifest for {target_id}"
                )
            complex_pose = Path(str(lineage_record["best_complex_pose"]))
        else:
            complex_pose = pose_manifest.get(target_id) if pose_manifest is not None else single_pose
        if complex_pose is None:
            raise SystemExit(
                f"Complex pose manifest has no pose for selected target_id: {target_id}"
            )
        rows.append(
            prepare_target(
                target_id,
                args,
                gmx_cmd,
                acpype_cmd,
                complex_pose,
                args.ligand_topology,
                lineage_record,
            )
        )
    write_manifest(args.out_manifest, rows)
    LOG.info("Wrote %s", args.out_manifest)


if __name__ == "__main__":
    main()
