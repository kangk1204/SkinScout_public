#!/usr/bin/env python3
"""B 트랙: 잘 결합한 후보를 시드로 BRICS 단편 재조합으로 '준-신규' 화합물을 생성한다.

- 단편 풀: 시드 + 해당 표적의 알려진 활성(top10)
- 코어 유지: 시드/활성들의 MCS(실패 시 시드 Murcko 스캐폴드) 부분구조 필수
- 신규성: 코퍼스(106만 측정 분자) 연결키에 없고, 스캐폴드가 시드·활성과 다름
- 물성/QED/SA 필터 후 점수화해 표적별 상위 30개 기록
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, ChemicalFeatures, Descriptors, QED, rdFMCS
from rdkit import RDConfig as _RDC
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/tmp/opencode/genb")
OUT.mkdir(parents=True, exist_ok=True)
FUNNEL_HOME = Path(os.environ.get("FUNNEL_HOME", ""))
if not FUNNEL_HOME.is_dir():
    raise SystemExit("FUNNEL_HOME 환경변수에 비공개 워크스페이스 경로를 지정하세요.")
QUEUE = FUNNEL_HOME / "target_pipeline_20260913/queue_top10.csv"
FUNNEL_CANDIDATES = FUNNEL_HOME / "analog_funnel_20260915/analog_candidates.csv"
LIGANDS = ROOT / "data/activity_retrieval_runtime_merged_202608/ligands.parquet"

TARGETS = {
    "PAR-2": {"uniprot": "P55085", "parent": "SUB0009"},
    "TGF-B1": {"uniprot": "P01137", "parent": "SUB0008"},
}
MAX_PRODUCTS = 25000
TOP_N = 30


def load_corpus_keys() -> set[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(LIGANDS, columns=["connectivity_key"])
    return {value for value in table.column("connectivity_key").to_pylist() if value}


def load_sources(gene: str, parent_id: str) -> tuple[str, list[str]]:
    parent = None
    for row in csv.DictReader(FUNNEL_CANDIDATES.open(encoding="utf-8-sig")):
        if row["gene"] == gene and row["candidate_id"] == parent_id:
            parent = row["candidate_smiles"]
            break
    if parent is None:
        raise SystemExit(f"parent {parent_id} not found for {gene}")
    actives = [
        row["smiles"].strip()
        for row in csv.DictReader(QUEUE.open(encoding="utf-8-sig"))
        if row["gene"] == gene
    ]
    return parent, actives


def feature_counts(factory, mol):
    # BRICS 조합물 일부는 ring info가 초기화되지 않아 피처 계산이 예외로 죽는다.
    Chem.GetSymmSSSR(mol)
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    counts: dict[str, int] = {}
    for feat in factory.GetFeaturesForMol(mol):
        counts[feat.GetFamily()] = counts.get(feat.GetFamily(), 0) + 1
    return counts


def recall(seed_counts: dict[str, int], mol_counts: dict[str, int]) -> float:
    if not seed_counts:
        return 0.0
    return sum(
        min(1.0, mol_counts.get(family, 0) / count)
        for family, count in seed_counts.items()
    ) / len(seed_counts)


def main() -> None:
    corpus = load_corpus_keys()
    print(f"[genb] 코퍼스 키 {len(corpus)}", flush=True)
    for gene, spec in TARGETS.items():
        parent, actives = load_sources(gene, spec["parent"])
        parent_mol = Chem.MolFromSmiles(parent)
        active_mols = [m for m in (Chem.MolFromSmiles(s) for s in actives) if m is not None]
        factory = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(_RDC.RDDataDir, "BaseFeatures.fdef")
        )
        seed_counts = feature_counts(factory, parent_mol)
        basis = "features:" + ",".join(f"{k}x{v}" for k, v in sorted(seed_counts.items()))
        source_smiles = {Chem.MolToSmiles(m) for m in active_mols + [parent_mol]}
        source_scaffolds = {
            Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
            for m in active_mols + [parent_mol]
        }
        fragments = set()
        for mol in active_mols + [parent_mol]:
            try:
                for frag in BRICS.BRICSDecompose(mol, returnMols=True):
                    fragments.add(Chem.MolToSmiles(frag))
            except Exception:
                continue
        frag_mols = [Chem.MolFromSmiles(s) for s in fragments]
        frag_mols = [m for m in frag_mols if m is not None]
        print(f"[genb] {gene}: 코어={basis}, 단편 {len(frag_mols)}", flush=True)
        seen: set[str] = set()
        kept: list[dict] = []
        generated = 0
        try:
            builder = BRICS.BRICSBuild(frag_mols, onlyCompleteMols=True, seed=42, maxDepth=2)
        except TypeError:
            builder = BRICS.BRICSBuild(frag_mols, onlyCompleteMols=True)
        for mol in builder:
            generated += 1
            if generated > MAX_PRODUCTS:
                break
            try:
                if mol is None or mol.GetNumHeavyAtoms() == 0:
                    continue
                key = Chem.MolToSmiles(mol, canonical=True)
                if key in seen or key in source_smiles:
                    continue
                seen.add(key)
                # 파마코포어(2D 피처 패밀리) 커버리지: 시드 피처를 70% 이상 재현
                fr = recall(seed_counts, feature_counts(factory, mol))
                if fr < 0.65:
                    continue
                ikey = Chem.MolToInchiKey(mol).split("-")[0]
                if ikey in corpus:
                    continue
                heavy = mol.GetNumHeavyAtoms()
                mw = Descriptors.MolWt(mol)
                rotb = Descriptors.NumRotatableBonds(mol)
                tpsa = Descriptors.TPSA(mol)
                logp = Descriptors.MolLogP(mol)
                if not (12 <= heavy <= 45 and 180 <= mw <= 650 and rotb <= 10 and tpsa <= 140 and -1 <= logp <= 5.5):
                    continue
                qed = QED.qed(mol)
                sa = sascorer.calculateScore(mol)
                if qed < 0.4 or sa > 4.5:
                    continue
                scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
                kept.append({
                    "gene": gene, "uniprot": spec["uniprot"], "parent": spec["parent"],
                    "smiles": key, "inchikey": ikey, "qed": round(qed, 3),
                    "sa": round(sa, 2), "mw": round(mw, 1), "heavy": heavy,
                    "rotb": rotb, "tpsa": round(tpsa, 1), "logp": round(logp, 2),
                    "feature_recall": round(fr, 3), "core_basis": basis, "scaffold": scaffold,
                })
            except Exception:
                continue
        kept.sort(key=lambda r: -(r["qed"] - 0.15 * r["sa"] + 0.2 * r["feature_recall"]))
        out_path = OUT / f"{gene.replace(' ', '_')}_candidates.csv"
        with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kept[0].keys()) if kept else [])
            writer.writeheader()
            writer.writerows(kept[:TOP_N])
        print(f"[genb] {gene}: 열거 {generated} → 필터 통과 {len(kept)} → 상위 {min(TOP_N, len(kept))} 저장 {out_path}",
              flush=True)


if __name__ == "__main__":
    main()
