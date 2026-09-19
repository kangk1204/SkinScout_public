#!/usr/bin/env python3
"""de novo 후보 재선별: 파마코포어·물성 필터 + 구조경보 + admet_ai Skin_Reaction 컷."""

from __future__ import annotations

import csv
import json
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
# rows[0]에서 fieldnames를 얻지 않는다: 후보 0개여도 같은 schema로 header를 쓴다.
FILTERED_FIELDS = ("smiles", "qed", "sa", "mw", "recall")
CANDIDATE_FIELDS = ("gene", "uniprot", "smiles", "qed", "sa", "mw",
                    "feature_recall", "skin_reaction")
FILTERED_NAME = "filtered_all.csv"
CANDIDATE_NAME = "TGF-B1_candidates.csv"
MANIFEST_NAME = "linkinvent_filter_manifest.json"


def write_rows(path: Path, fields, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(counts: dict, *, status: str, reason: str) -> None:
    payload = {
        "schema": "skinscout.analog-linkinvent-filter.v1",
        "status": status,
        "reason": reason,
        "input": counts["input"],
        "parsed": counts["parsed"],
        "filtered": counts["filtered"],
        "kept": counts["kept"],
    }
    (P / MANIFEST_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    if not SEED:
        raise SystemExit("FUNNEL_SEED_SMILES 환경변수에 기준 화합물 SMILES를 지정하세요.")
    seed_mol = Chem.MolFromSmiles(SEED)
    if seed_mol is None:
        raise SystemExit(f"FUNNEL_SEED_SMILES를 해석할 수 없습니다: {SEED!r}")
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
    linked = P / "linked.smi"
    if not linked.is_file():
        raise SystemExit(f"linked.smi가 없습니다: {linked}")
    lines = [line for line in linked.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    if not sc:
        # 시드 feature가 0이면 recall 분모가 0이라 어떤 후보도 평가할 수 없다.
        # header-only 출력과 사유를 남기고 설명 가능한 상태로 끝낸다.
        write_rows(P / FILTERED_NAME, FILTERED_FIELDS, [])
        write_rows(P / CANDIDATE_NAME, CANDIDATE_FIELDS, [])
        write_manifest({"input": len(lines), "parsed": 0, "filtered": 0, "kept": 0},
                       status="seed_feature_zero",
                       reason=f"seed has no pharmacophore features: {SEED}")
        print("[preselect] 0 (seed_feature_zero)", flush=True)
        return 2
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
    parsed = 0
    for line in lines:
        smi = line.split(",")[0].strip().strip('"')
        if not smi or smi in seen:
            continue
        seen.add(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        parsed += 1
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
    if not rows:
        status = "empty_input" if not lines else "all_filtered"
        reason = ("linked.smi has no candidate rows" if status == "empty_input"
                  else f"all {len(lines)} input row(s) removed by preselect filters")
        write_rows(P / FILTERED_NAME, FILTERED_FIELDS, [])
        write_rows(P / CANDIDATE_NAME, CANDIDATE_FIELDS, [])
        write_manifest({"input": len(lines), "parsed": parsed, "filtered": 0, "kept": 0},
                       status=status, reason=reason)
        print(f"[select] {status}: kept 0", flush=True)
        return 0
    write_rows(P / FILTERED_NAME, FILTERED_FIELDS, rows)
    filtered = len(rows)

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
    write_rows(P / CANDIDATE_NAME, CANDIDATE_FIELDS, [
        {"gene": "TGF-B1", "uniprot": "P01137", "smiles": r["smiles"],
         "qed": r["qed"], "sa": r["sa"], "mw": r["mw"],
         "feature_recall": r["recall"], "skin_reaction": r["skin_reaction"]}
        for r in kept[:TOP]
    ])
    write_manifest({"input": len(lines), "parsed": parsed, "filtered": filtered,
                    "kept": len(kept)}, status="ok", reason="")
    for r in kept[:5]:
        print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
