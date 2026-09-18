#!/usr/bin/env python3
"""유사체 선별: 게이트(근거 유지·경보 0·물성) + MMR 다양성으로 시드별 대표 후보를 고른다."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENROOT = Path("/tmp/opencode/analog_funnel/gen")
OUT = Path("/tmp/opencode/analog_funnel/shortlist.csv")
PER_SEED = 5
MMR_LAMBDA = 0.6

sys.path.insert(0, str(ROOT / "scripts"))
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator, DataStructs  # noqa: E402


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def gate(record: dict) -> tuple[bool, str]:
    reasons: list[str] = []
    if not (as_bool(record.get("binding_retained")) or record.get("evidence_tier") == "direct_retained"):
        reasons.append("evidence_retention")
    if as_float(record.get("structural_alert_count"), 99) > 0:
        reasons.append("structural_alert")
    if as_float(record.get("property_suitability_score")) < 0.5:
        reasons.append("property")
    return (not reasons), ",".join(reasons)


def score(record: dict) -> float:
    return (
        0.50 * as_float(record.get("pharmacophore_preservation_score"))
        + 0.30 * as_float(record.get("binding_support_score"))
        + 0.20 * as_float(record.get("safety_triage_score"))
    )


FIELDNAMES = [
    "seed_dir", "mmr_rank", "gate_passed", "gate_reason", "funnel_score",
    "candidate_id", "name", "smiles", "inchikey", "evidence_tier", "priority_label",
    "cosing_reference", "pharmacophore_preservation_score",
    "strict_3d_pharmacophore_gate_passed", "corpus_novelty_proxy",
    "structural_alert_count", "structural_similarity_to_parent",
    "binding_status", "binding_pactivity_delta",
]


def main() -> None:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows_out: list[dict] = []
    rejected_out: list[dict] = []
    for seed_dir in sorted(p for p in GENROOT.iterdir() if p.is_dir()):
        csv_path = seed_dir / "substitute_candidates.csv"
        if not csv_path.is_file():
            continue
        records = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
        pool: list[dict] = []
        seen: set[str] = set()
        for record in records:
            key = record.get("inchikey") or record.get("smiles")
            if key in seen:
                continue
            seen.add(key)
            mol = Chem.MolFromSmiles(record.get("smiles", ""))
            if mol is None:
                continue
            passed, why = gate(record)
            pool.append({"record": record, "mol": mol, "gate": passed, "gate_reason": why,
                         "score": score(record), "fp": generator.GetFingerprint(mol)})
        chosen: list[dict] = []
        # C01: 게이트는 정렬 힌트가 아니라 MMR 입력의 제약이다. 탈락 후보는
        # shortlist에 들어갈 수 없고, 별도 파일로 남겨 추적만 가능하게 한다.
        eligible = [item for item in pool if item["gate"]]
        rejected = [item for item in pool if not item["gate"]]
        for item in rejected:
            record = item["record"]
            rejected_out.append({
                "seed_dir": seed_dir.name,
                "candidate_id": record.get("candidate_id", ""),
                "smiles": record.get("smiles", ""),
                "gate_reason": item["gate_reason"],
                "funnel_score": round(item["score"], 4),
            })
        candidates = sorted(eligible, key=lambda item: -item["score"])
        while candidates and len(chosen) < PER_SEED:
            best = None
            best_value = None
            for item in candidates:
                if chosen:
                    sim = max(
                        DataStructs.TanimotoSimilarity(item["fp"], pick["fp"]) for pick in chosen
                    )
                else:
                    sim = 0.0
                value = item["score"] - MMR_LAMBDA * sim
                if best_value is None or value > best_value:
                    best, best_value = item, value
            candidates.remove(best)
            chosen.append(best)
        for rank, item in enumerate(chosen, 1):
            record = item["record"]
            rows_out.append({
                "seed_dir": seed_dir.name,
                "mmr_rank": rank,
                "gate_passed": item["gate"],
                "gate_reason": item["gate_reason"],
                "funnel_score": round(item["score"], 4),
                "candidate_id": record.get("candidate_id", ""),
                "name": record.get("name", ""),
                "smiles": record.get("smiles", ""),
                "inchikey": record.get("inchikey", ""),
                "evidence_tier": record.get("evidence_tier", ""),
                "priority_label": record.get("priority_label", ""),
                "cosing_reference": record.get("cosing_reference", ""),
                "pharmacophore_preservation_score": record.get("pharmacophore_preservation_score", ""),
                "strict_3d_pharmacophore_gate_passed": record.get("strict_3d_pharmacophore_gate_passed", ""),
                "corpus_novelty_proxy": record.get("corpus_novelty_proxy", ""),
                "structural_alert_count": record.get("structural_alert_count", ""),
                "structural_similarity_to_parent": record.get("structural_similarity_to_parent", ""),
                "binding_status": record.get("binding_status", ""),
                "binding_pactivity_delta": record.get("binding_pactivity_delta", ""),
            })
        print(f"{seed_dir.name}: 풀 {len(pool)} → 게이트 통과 {len(eligible)} "
              f"(탈락 {len(rejected)}) → 선별 {len(chosen)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)
    rejected_path = OUT.with_name("shortlist_rejected.csv")
    with rejected_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["seed_dir", "candidate_id", "smiles", "gate_reason", "funnel_score"]
        )
        writer.writeheader()
        writer.writerows(rejected_out)
    print(f"wrote {OUT} rows={len(rows_out)} (탈락 기록 {len(rejected_out)} → {rejected_path})")


if __name__ == "__main__":
    main()
