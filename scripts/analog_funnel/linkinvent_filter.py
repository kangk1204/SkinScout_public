#!/usr/bin/env python3
"""de novo 후보 재선별: 파마코포어·물성 필터 + 구조경보 + admet_ai Skin_Reaction 컷."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from rdkit import Chem, RDLogger, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors, FilterCatalog, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
import pyarrow.parquet as pq

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
P = Path("/tmp/opencode/reinvent_pilot")
SEED = os.environ.get("FUNNEL_SEED_SMILES", "")
SKIN_MAX = 0.45
TOP = 10


def main() -> None:
    if not SEED:
        raise SystemExit("FUNNEL_SEED_SMILES 환경변수에 기준 화합물 SMILES를 지정하세요.")
    seed_mol = Chem.MolFromSmiles(SEED)
    fac = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )

    def counts(mol):
        Chem.GetSymmSSSR(mol)
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        c = {}
        for f in fac.GetFeaturesForMol(mol):
            c[f.GetFamily()] = c.get(f.GetFamily(), 0) + 1
        return c

    sc = counts(seed_mol)
    def recall(mc):
        return sum(min(1.0, mc.get(k, 0) / v) for k, v in sc.items()) / len(sc)

    seed_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(seed_mol))
    corpus = {
        v
        for v in pq.read_table(
            ROOT / "data/activity_retrieval_runtime_merged_202608/ligands.parquet",
            columns=["connectivity_key"],
        ).column("connectivity_key").to_pylist()
        if v
    }
    params = FilterCatalog.FilterCatalogParams()
    for name in (
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    ):
        params.AddCatalog(name)
    catalog = FilterCatalog.FilterCatalog(params)

    rows = []
    seen = set()
    for line in (P / "linked.smi").read_text(encoding="utf-8").splitlines()[1:]:
        smi = line.split(",")[0].strip().strip('"')
        if not smi or smi in seen:
            continue
        seen.add(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rc = recall(counts(mol))
        if rc < 0.65:
            continue
        if Chem.MolToInchiKey(mol).split("-")[0] in corpus:
            continue
        h = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        if not (12 <= h <= 40 and 180 <= mw <= 520 and Descriptors.NumRotatableBonds(mol) <= 8
                and Descriptors.TPSA(mol) <= 130 and -1 <= Descriptors.MolLogP(mol) <= 5):
            continue
        q = QED.qed(mol)
        sa = sascorer.calculateScore(mol)
        if q < 0.5 or sa > 4.0:
            continue
        if Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) == seed_scaffold:
            continue
        if catalog.HasMatch(mol):
            continue
        rows.append({"smiles": smi, "qed": round(q, 3), "sa": round(sa, 2),
                     "mw": round(mw, 1), "recall": round(rc, 3)})
    print(f"[preselect] {len(rows)}", flush=True)
    with (P / "filtered_all.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import admet_ai

    model = admet_ai.ADMETModel()
    rows.sort(key=lambda r: -(r["qed"] + 0.3 * r["recall"]))
    probe = rows[:600]
    skin = {}
    for idx, r in enumerate(probe):
        try:
            pred = model.predict(smiles=r["smiles"])
            if isinstance(pred, dict):
                skin[r["smiles"]] = float(pred.get("Skin_Reaction", 1.0))
        except Exception:
            skin[r["smiles"]] = 1.0
        if (idx + 1) % 100 == 0:
            print(f"[admet] {idx + 1}/{len(probe)}", flush=True)
    rows = [r for r in rows if r["smiles"] in skin]
    kept = [dict(r, skin_reaction=round(skin[r["smiles"]], 3))
            for r in rows if skin[r["smiles"]] < SKIN_MAX]
    kept.sort(key=lambda r: -(r["qed"] + 0.3 * r["recall"]))
    print(f"[select] skin<{SKIN_MAX}: {len(kept)} → 상위 {min(TOP, len(kept))}", flush=True)
    with (P / "TGF-B1_candidates.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["gene", "uniprot", "smiles", "qed", "sa", "mw", "feature_recall", "skin_reaction"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in kept[:TOP]:
            writer.writerow({"gene": "TGF-B1", "uniprot": "P01137", "smiles": r["smiles"],
                             "qed": r["qed"], "sa": r["sa"], "mw": r["mw"],
                             "feature_recall": r["recall"], "skin_reaction": r["skin_reaction"]})
    for r in kept[:5]:
        print("  ", r)


if __name__ == "__main__":
    main()
