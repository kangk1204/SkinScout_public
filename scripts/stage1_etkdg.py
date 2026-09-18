#!/usr/bin/env python3
"""stage1_etkdg.py — ETKDGv3 conformer generation + MMFF94 ranking + RMSD prune."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

LOG = logging.getLogger("stage1.etkdg")


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _write_sdf_atomic(
    mol: Chem.Mol,
    conformers: list[tuple[float, int, int]],
    path: Path,
    force_field: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer = Chem.SDWriter(str(tmp))
    for energy, status, cid in conformers:
        mol.SetProp("ff_kind", force_field)
        mol.SetProp("ff_energy", f"{energy:.3f}")
        mol.SetProp("ff_status", "converged" if status == 0 else "not_converged")
        if force_field == "MMFF":
            mol.SetProp("mmff_energy", f"{energy:.3f}")
            mol.SetProp("mmff_status", "converged" if status == 0 else "not_converged")
        writer.write(mol, confId=cid)
    writer.close()
    tmp.replace(path)


def _embed(mol: Chem.Mol, base_params, n_conf: int) -> tuple[list[int], str]:
    """ETKDGv3 기본 → 실패 시 랜덤 좌표 → 거대고리 토션 순으로 시도한다.

    기본 파라미터는 대부분의 분자를 잘 임베딩하지만, 큰 고리·펩타이드성
    화합물에서 "failed to embed any conformer"로 죽는다(실측: DKK-1 후보).
    실패를 그대로 fail-closed로 두면 안전성·표적 분석 전체가 막히므로,
    카이랄리티는 그대로 두고 좌표 생성 방식만 바꿔 재시도한다.
    """
    attempts: list[tuple[str, object, int]] = [("etkdgv3", base_params, n_conf)]

    fallback_confs = max(1, min(n_conf, 10))

    random_params = AllChem.ETKDGv3()
    random_params.randomSeed = base_params.randomSeed
    random_params.pruneRmsThresh = base_params.pruneRmsThresh
    random_params.numThreads = base_params.numThreads
    random_params.useRandomCoords = True
    random_params.timeout = 30
    attempts.append(("etkdgv3_random_coords", random_params, fallback_confs))

    macro_params = AllChem.ETKDGv3()
    macro_params.randomSeed = base_params.randomSeed
    macro_params.pruneRmsThresh = base_params.pruneRmsThresh
    macro_params.numThreads = base_params.numThreads
    macro_params.useRandomCoords = True
    macro_params.timeout = 30
    if hasattr(macro_params, "useMacrocycleTorsions"):
        macro_params.useMacrocycleTorsions = True
    if hasattr(macro_params, "useMacrocycle14config"):
        macro_params.useMacrocycle14config = True
    attempts.append(("etkdgv3_macrocycle_random", macro_params, fallback_confs))

    for label, params, confs in attempts:
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=confs, params=params))
        if cids:
            return cids, label
    return [], ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-sdf", required=True, type=Path)
    parser.add_argument("--out-sdf", required=True, type=Path)
    parser.add_argument("--n-conf", type=int, default=50)
    parser.add_argument("--rmsd-prune", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _remove_outputs(args.out_sdf)
    supplier = Chem.SDMolSupplier(str(args.in_sdf), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise SystemExit(f"No parseable mol in {args.in_sdf}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC05A
    params.pruneRmsThresh = args.rmsd_prune
    params.numThreads = 0
    # 기본 시도에도 상한을 둔다. timeout=0(무제한)이면 느린 임베딩이 끝나지 않아
    # bounded fallback에 도달조차 못 한다.
    params.timeout = 60
    cids, method = _embed(mol, params, args.n_conf)
    if not cids:
        raise SystemExit("ETKDG failed to embed any conformer")
    if method != "etkdgv3":
        LOG.warning("ETKDG fallback used: %s", method)

    def _collect(force_field: str) -> tuple[list[tuple[float, int, int]], int, int]:
        energies: list[tuple[float, int, int]] = []
        nonconverged = 0
        unusable = 0
        for cid in cids:
            # timeout으로 잘린 컨포머는 좌표가 불완전해 힘장이 "Bad Conformer Id"로
            # 거부할 수 있다. 유효한 것만 남기고 센다.
            try:
                if force_field == "MMFF":
                    res = AllChem.MMFFOptimizeMolecule(mol, maxIters=args.max_iters, confId=cid)
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid) if props is not None else None
                else:
                    res = AllChem.UFFOptimizeMolecule(mol, maxIters=args.max_iters, confId=cid)
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            except (ValueError, RuntimeError):
                unusable += 1
                continue
            if ff is None:
                unusable += 1
                continue
            if res != 0:
                nonconverged += 1
            energies.append((ff.CalcEnergy(), res, cid))
        return energies, nonconverged, unusable

    energies, n_nonconverged, n_unusable = _collect("MMFF")
    force_field = "MMFF"
    if not energies:
        # MMFF94는 붕소 등 일부 원소의 파라미터가 없어 조용히 전부 실패한다
        # (실측: 벤즈옥사보롤 KLK5 후보 3종). UFF는 해당 원소를 지원한다.
        LOG.warning("MMFF unusable for every conformer; retrying with UFF")
        energies, n_nonconverged, n_unusable = _collect("UFF")
        force_field = "UFF"
    if not energies:
        raise SystemExit("MMFF and UFF failed to optimize every embedded conformer")
    # 수렴한 컨포머를 먼저, 그 다음 에너지 순으로 정렬한다. 비수렴 좌표가 낮은
    # 에너지로 best가 되어 xTB 입력이 되는 것을 막는다.
    energies.sort(key=lambda item: (item[1] != 0, item[0], item[2]))
    _write_sdf_atomic(mol, energies, args.out_sdf, force_field)
    LOG.info("Wrote %d conformers (%s; %d not converged, %d unusable; best E=%.3f) → %s",
             len(energies), force_field, n_nonconverged, n_unusable,
             energies[0][0] if energies else float("nan"), args.out_sdf)


if __name__ == "__main__":
    main()
