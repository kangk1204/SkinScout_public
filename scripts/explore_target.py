#!/usr/bin/env python3
"""단백질 이름으로 표적을 골라, 그 표적에 대해 측정 활성이 알려진 화합물을
검색 인덱스에서 꺼내 보여주고 파이프라인 입력으로 넘긴다.

SkinScout 기본 파이프라인은 "성분 → 표적 순위"다. 초보자가 표적부터 시작하고
싶을 때 이 도구가 그 다리를 놓는다: UniProt 계정번호나 유전자·단백질 이름을
넣으면 ChEMBL/BindingDB 측정으로 연결돼 있는 화합물을 인덱스에서 찾아 정렬해
보여 주고, 그 화합물을 `run_skinscout.py`로 보내는 명령을 안내한다.

예:
  python scripts/explore_target.py P14679
  python scripts/explore_target.py tyrosinase        # TYR
  python scripts/explore_target.py --search melan     # 이름 후보부터

읽는 것은 로컬 검색 인덱스(Stage 0 빌드 산출물)뿐이라, 공개 저장소 원본처럼
`data/`에 인덱스가 없으면 안내 멘트와 함께 멈춘다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_ROLE = "production"
POSITIVE_THRESHOLD = 6.0
NEGATIVE_THRESHOLD = 5.0

DEFAULT_TOP = 20
MAX_RUN_CMDS = 10

# 잘 알려진 피부 관련 단백질의 흔한 이름 → 유전자 기호. 인덱스는 UniProt
# 계정번호로 연결되므로 이름을 계정번호로 바꿔 주는 작은 사전이다. 렉시콘에서
# 잡히지 않는 흔한 이름만 넣는다.
COMMON_PROTEIN_ALIASES: dict[str, str] = {
    "tyrosinase": "TYR",
    "tyrosinase-related protein 1": "TYRP1",
    "matrix metalloproteinase-2": "MMP2",
    "gelatinase a": "MMP2",
    "matrix metalloproteinase-9": "MMP9",
    "gelatinase b": "MMP9",
    "neutrophil elastase": "ELANE",
    "elastase": "ELANE",
    "retinoic acid receptor alpha": "RARA",
    "retinoic acid receptor beta": "RARB",
    "retinoic acid receptor gamma": "RARG",
    "poly [adp-ribose] polymerase 1": "PARP1",
    "sirtuin 1": "SIRT1",
    "nicotinamide n-methyltransferase": "NNMT",
    "transient receptor potential ankyrin 1": "TRPA1",
}


# --------------------------------------------------------------------------- target lookup


@dataclass(frozen=True)
class LexiconEntry:
    uniprot: str
    gene: str
    description: str


def _csv_entries() -> list[LexiconEntry]:
    """인덱스 검색어를 만드는 저장소 내부 큐레이션 표를 모은다.

    각 표는 어차피 로컬 데이터라, 계정번호와 유전자 기호(+부분 검색용
    description)만 보관한다.
    """
    entries: dict[str, LexiconEntry] = {}

    def add(uniprot: str, gene: str, description: str = "") -> None:
        uniprot = uniprot.strip()
        gene = gene.strip()
        if not uniprot or not gene or ";" in uniprot or ";" in gene:
            return
        entries.setdefault(uniprot, LexiconEntry(uniprot, gene, description.strip()))

    worklist = ROOT / "data" / "curation" / "skin_target_worklist.csv"
    if worklist.is_file():
        for row in csv.DictReader(worklist.open(encoding="utf-8-sig")):
            add(row.get("uniprot") or "", row.get("gene") or "", row.get("description") or "")

    failures = ROOT / "data" / "validation" / "known_failure_modes.csv"
    if failures.is_file():
        for row in csv.DictReader(failures.open(encoding="utf-8-sig")):
            add(row.get("uniprot") or "", row.get("gene") or "")

    # rerank 진실값·대조 표는 표적 라벨(RAR-alpha 같은)을 담고 있다. 둘 다
    # 세미콜론으로 여러 개를 연결할 수 있어, 순서쌍으로 zip해 넣는다.
    for panel in (
        ROOT / "data" / "validation" / "rerank_band_20260828" / "known_targets.csv",
        ROOT / "data" / "validation" / "skin_known_target_panel.csv",
    ):
        if not panel.is_file():
            continue
        for row in csv.DictReader(panel.open(encoding="utf-8-sig")):
            uniprots = (row.get("target_id") or row.get("known_targets") or "").split(";")
            labels = (row.get("target_label") or row.get("known_target_labels") or "").split(";")
            for uniprot, label in zip(uniprots, labels):
                if not uniprot.strip() or not label.strip():
                    continue
                gene = label.strip().replace("-alpha", "A").replace("-beta", "B").replace("-gamma", "G")
                gene = re.sub(r"[-_ .].*$", "", gene)
                add(uniprot, gene or uniprot)

    return list(entries.values())


def _build_lexicon(entries: list[LexiconEntry]) -> dict[str, str]:
    """유전자·이름 → UniProt 계정번호. 키는 소문자."""
    mapping: dict[str, str] = {}
    for entry in entries:
        mapping.setdefault(entry.gene.lower(), entry.uniprot)
        mapping.setdefault(entry.uniprot.lower(), entry.uniprot)
        for token in re.split(r"\s+", entry.description.lower()):
            if len(token) >= 4 and token.isalpha():
                mapping.setdefault(token, entry.uniprot)
    for alias, gene in COMMON_PROTEIN_ALIASES.items():
        target = mapping.get(gene.lower())
        if target:
            # 설명 문구의 토큰(예: P17643의 "Tyrosinase related protein 1")이
            # "tyrosinase" 키를 먼저 차지할 수 있으니, 명시적 별칭이 이긴다.
            mapping[alias.lower()] = target
    return mapping


def resolve_target(query: str) -> tuple[LexiconEntry | None, list[str]]:
    """질의를 표적으로 해석한다. (결과, 부분 일치 후보)를 돌려준다."""
    entries = _csv_entries()
    lexicon = _build_lexicon(entries)
    q = query.strip()
    if not q:
        return None, []
    key = q.lower()
    by_uniprot = {e.uniprot: e for e in entries}
    if key in by_uniprot:
        return by_uniprot[key], []
    exact = lexicon.get(key)
    if exact:
        return by_uniprot.get(exact) or LexiconEntry(exact, "", ""), []
    candidates: list[str] = []
    for entry in entries:
        hay = " ".join((entry.uniprot, entry.gene, entry.description)).lower()
        if len(q) >= 3 and key in hay and entry.uniprot not in candidates:
            candidates.append(entry.uniprot)
    return None, candidates


# --------------------------------------------------------------------------- index read


def _configured_index_dir() -> Path | None:
    """파이프라인과 같은 인덱스를 쓴다.

    `workflow/config.yaml`의 `daina_recipe_index_dir`가 실행 경로가 실제로 읽는
    인덱스다. 여기서 놀고 다른 판(예: GtoPdb 포함)을 보여주면 초보자가 파이프라인
    결과와 다른 목록을 보게 된다.
    """
    config = ROOT / "workflow" / "config.yaml"
    if not config.is_file():
        return None
    match = re.search(
        r'^\s*daina_recipe_index_dir:\s*["\']?([^"\'\s]+)', config.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        return None
    value = match.group(1)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate if (candidate / "manifest.json").is_file() else None


def _discover_index_dir() -> Path | None:
    configured = _configured_index_dir()
    if configured is not None:
        return configured
    if not (ROOT / "data").is_dir():
        return None
    for child in sorted((ROOT / "data").iterdir(), key=lambda p: p.name, reverse=True):
        if not child.is_dir() or not child.name.startswith("activity_retrieval"):
            continue
        manifest = child / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("index_role") == PRODUCTION_ROLE:
            return child
    return None


def load_index(index_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = index_dir / "manifest.json"
    for required in (index_dir / "edges.parquet", index_dir / "ligands.parquet", manifest_path):
        if not required.exists():
            raise SystemExit(f"검색 인덱스가 불완전합니다: {required}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"manifest.json을 읽을 수 없습니다: {manifest_path}: {exc}") from exc
    if manifest.get("index_role") != PRODUCTION_ROLE:
        raise SystemExit(
            f"--index-dir는 production 인덱스여야 합니다 (지정값: "
            f"{manifest.get('index_role', '알 수 없음')!r}). 평가용 인덱스는 훈련 "
            "분할만 담아 실제보다 협소하게 보입니다."
        )
    edges = pd.read_parquet(index_dir / "edges.parquet")
    ligands = pd.read_parquet(
        index_dir / "ligands.parquet",
        columns=["ligand_index", "ligand_key", "standard_inchikey", "canonical_smiles"],
    )
    if not {"ligand_index", "uniprot"} <= set(edges.columns):
        raise SystemExit("인덱스 edges.parquet 열이 기대와 다릅니다")
    if not {"ligand_index", "canonical_smiles"} <= set(ligands.columns):
        raise SystemExit("인덱스 ligands.parquet 열이 기대와 다릅니다")
    return edges, ligands


def rank_target(
    edges: pd.DataFrame,
    ligands: pd.DataFrame,
    uniprot: str,
    top: int,
    mode: str,
) -> pd.DataFrame:
    target = edges[edges["uniprot"] == uniprot]
    if target.empty:
        return target
    merged = target.merge(ligands, on="ligand_index", how="inner")
    if merged.empty:
        return merged
    columns = [
        col for col in (
            "ligand_key",
            "standard_inchikey",
            "canonical_smiles",
            "source_db",
            "max_pactivity",
            "median_pactivity",
            "positive_measurement_count",
            "publication_count",
            "measurement_count",
        ) if col in merged.columns
    ]
    merged = merged[columns]
    if mode == "potency":
        merged = merged.sort_values(
            ["max_pactivity", "publication_count", "measurement_count"],
            ascending=False,
        )
    elif mode == "evidence":
        merged = merged.sort_values(
            ["publication_count", "measurement_count", "max_pactivity"],
            ascending=False,
        )
    else:
        merged = merged.sort_values(["max_pactivity", "publication_count"], ascending=False)
    return merged.head(top)


def _threshold_note(max_pactivity: float) -> str:
    if max_pactivity >= POSITIVE_THRESHOLD:
        return ""
    if max_pactivity >= NEGATIVE_THRESHOLD:
        return "  (경계, 5 이하는 음성 문턱)"
    return "  (문턱 아래)"
# --------------------------------------------------------------------------- reporting


def format_target_table(frame: pd.DataFrame, uniprot: str, entry: LexiconEntry | None) -> list[str]:
    """표적 정보와 순위표를 화면 줄 목록으로 만든다."""
    header = f"표적 {uniprot}"
    if entry:
        header += f"  ({entry.gene})"
        if entry.description:
            header += f" — {entry.description}"
    lines = [header, ""]
    if frame.empty:
        lines.append("이 표적에 대해 이 인덱스에 측정 활성이 없습니다.")
        lines.append("(인덱스는 ChEMBL 37 + BindingDB 측정, 표적 4,873개로 구성됩니다.)")
        return lines
    show = frame.copy()
    show["활성_문턱"] = show["max_pactivity"].map(_threshold_note)
    show["max_pactivity"] = show["max_pactivity"].map(lambda v: f"{v:.2f}")
    show["canonical_smiles"] = show["canonical_smiles"].map(
        lambda s: s if len(s) <= 40 else s[:37] + "..."
    )
    lines.append(
        show[["canonical_smiles", "max_pactivity", "활성_문턱",
              "positive_measurement_count", "publication_count", "source_db"]]
        .to_string(index=False)
    )
    return lines


def pipeline_commands(frame: pd.DataFrame, mode: str, limit: int) -> list[str]:
    """선택된 화합물을 `run_skinscout.py`로 보내는 명령 목록."""
    cmds: list[str] = []
    for _, row in frame.head(limit).iterrows():
        smiles = row["canonical_smiles"]
        cmds.append(f"python scripts/run_skinscout.py --smiles '{smiles}' --mode {mode}")
    return cmds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="표적 단백질(UniProt·유전자·이름)로 시작해 알려진 결합 화합물을 찾는다",
    )
    parser.add_argument("target", nargs="?", help="UniProt 계정번호 또는 유전자·단백질 이름")
    parser.add_argument("--search", help="이름·유전자 부분 문자열로 표적 후보를 나열한다")
    parser.add_argument("--mode", choices=("balanced", "potency", "evidence"), default="balanced")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--run-mode", choices=("fast", "comprehensive"), default="fast")
    parser.add_argument("--index-dir", type=Path, help="production 검색 인덱스 디렉터리")
    parser.add_argument("--smiles-out", type=Path, help="정규화 SMILES 목록을 파일로 저장")
    parser.add_argument("--print-run-cmds", action="store_true", help="파이프라인 명령 전체 출력")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top 은 1 이상이어야 합니다")

    if args.search:
        hits = [
            e for e in _csv_entries()
            if args.search.lower()
            in " ".join((e.uniprot, e.gene, e.description)).lower()
        ]
        for entry in sorted(hits, key=lambda e: e.uniprot):
            print(f"{entry.uniprot}\t{entry.gene}\t{entry.description}")
        print(f"\n후보 {len(hits)}개. `python scripts/explore_target.py <UniProt>`로 조회하세요.")
        return 0

    entry, candidates = resolve_target(args.target or "")
    if entry is None:
        if candidates:
            print(f"'{args.target}'와 정확히 일치하는 표적이 없습니다. 후보:")
            for uniprot in candidates:
                hit = next(e for e in _csv_entries() if e.uniprot == uniprot)
                print(f"  {uniprot}\t{hit.gene}\t{hit.description}")
            return 2
        raise SystemExit(
            f"'{args.target}'를 표적으로 해석할 수 없습니다. UniProt 계정번호 (예: P14679)나 "
            "유전자 이름 (예: TYR, MMP2)을 쓰거나, `--search`로 후보를 찾아보세요."
        )

    index_dir = args.index_dir or _discover_index_dir()
    if index_dir is None:
        raise SystemExit(
            "이 컴퓨터에 검색 인덱스가 없습니다. README의 Stage 0 데이터 빌드를 먼저 "
            "돌려주세요 (`data/activity_retrieval_*` production 인덱스)."
        )
    edges, ligands = load_index(index_dir)
    frame = rank_target(edges, ligands, entry.uniprot, args.top, args.mode)
    for line in format_target_table(frame, entry.uniprot, entry):
        print(line)

    if frame.empty:
        return 0

    if args.smiles_out is not None:
        args.smiles_out.parent.mkdir(parents=True, exist_ok=True)
        args.smiles_out.write_text(
            "\n".join(frame["canonical_smiles"]) + "\n", encoding="utf-8"
        )
        print(f"\nSMILES 목록 저장: {args.smiles_out} (파이프라인 입력용)")

    if args.print_run_cmds:
        print("\n파이프라인 실행 명령 (위에서 선택한 화합물, 한 줄 = 하나의 run):")
        for cmd in pipeline_commands(frame, args.run_mode, MAX_RUN_CMDS):
            print(f"  {cmd}")
    else:
        top_row = frame.iloc[0]
        top_smiles = top_row["canonical_smiles"]
        print(
            f"\n파이프라인으로 보내려면 (1위 예시, fast 경로):\n"
            f"  python scripts/run_skinscout.py --smiles '{top_smiles}' --mode {args.run_mode}"
        )
        print("전체 명령이 필요하면 --print-run-cmds, SMILES 파일은 --smiles-out을 쓰세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())