#!/usr/bin/env python3
"""stage0_skin_kg.py — Skin-efficacy knowledge graph from PubTator 3.0.

Schema (NetworkX):
    Gene  -[ASSOCIATED_WITH {pmid_list, n_papers}]->  EfficacyCategory
    Compound -[USED_FOR]-> EfficacyCategory
    Gene  -[KNOWN_TARGET_OF]-> Compound

Writes <OUTDIR>/skin_efficacy.graphml.

Use --dry-run on hosts without external network — emits a small seeded graph
covering the 11 efficacy categories in §3.4 plus a handful of canonical
gene/efficacy edges (TYR→whitening, MMP1→wrinkle, FLG→hydration, …).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import networkx as nx
import requests

LOG = logging.getLogger("stage0.skin_kg")

PUBTATOR_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
GENE_TAG_RE = re.compile(r"@GENE_([A-Za-z0-9_.-]+)")
GENE_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{1,24}$")

EFFICACY_KEYWORDS = {
    "whitening": ["tyrosinase inhibitor", "melanogenesis", "skin lightening",
                  "depigmenting", "melanin synthesis inhibitor"],
    "anti_aging": ["anti-wrinkle", "photoaging", "collagen synthesis",
                   "elastin", "MMP inhibitor"],
    "hydration": ["moisturization", "aquaporin AQP3",
                  "hyaluronic acid synthase", "filaggrin"],
    "acne": ["acne sebocyte", "sebum production",
             "Cutibacterium acnes inhibitor"],
    "atopic_dermatitis": ["atopic dermatitis barrier",
                          "filaggrin atopic", "Th2 IL-13 skin"],
    "anti_inflammatory": ["NF-kB skin", "COX-2 skin",
                          "inflammasome NLRP3 skin"],
    "antioxidant": ["NRF2 KEAP1 skin", "oxidative stress skin"],
    "photoaging": ["UV-induced damage skin", "AhR photoaging"],
    "dandruff": ["seborrheic dermatitis Malassezia"],
    "hair_growth": ["hair follicle Wnt", "DKK1 hair", "androgen receptor hair"],
    "retinoid": ["retinoid RAR RXR", "ALDH1A2"],
}

# Curated seeds for the dry-run / fallback graph.
SEED_GENE_EFFICACY = [
    ("P14679", "TYR",   "whitening",         562),
    ("O75030", "MITF",  "whitening",          89),
    ("P03956", "MMP1",  "anti_aging",        421),
    ("P14780", "MMP9",  "anti_aging",        290),
    ("P20930", "FLG",   "hydration",         174),
    ("Q5D862", "FLG2",  "hydration",          22),
    ("Q92482", "AQP3",  "hydration",         118),
    ("P10275", "AR",    "acne",              140),
    ("P10275", "AR",    "hair_growth",       250),
    ("P18405", "SRD5A1","acne",               46),
    ("P10276", "RARA",  "retinoid",          640),
    ("P10826", "RARB",  "retinoid",          188),
    ("P13631", "RARG",  "retinoid",          155),
    ("P19838", "NFKB1", "anti_inflammatory", 612),
    ("Q16236", "NFE2L2","antioxidant",       380),
    ("P35354", "PTGS2", "anti_inflammatory", 442),
]
SEED_SYMBOL_TO_UNIPROT = {
    symbol: uniprot for uniprot, symbol, _category, _n_papers in SEED_GENE_EFFICACY
}


def _remove_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_graphml_atomic(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    nx.write_graphml(graph, tmp)
    tmp.replace(path)


def query_pubtator(keyword: str, max_pmids: int = 200) -> list[dict]:
    try:
        r = requests.get(
            f"{PUBTATOR_BASE}/search/",
            params={
                "text": keyword,
                "format": "json",
                "page_size": str(max(1, min(max_pmids, 200))),
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"PubTator query failed for {keyword!r}: {exc}") from exc
    if r.status_code != 200:
        detail = getattr(r, "text", "")[:200]
        raise SystemExit(
            f"PubTator query failed for {keyword!r}: HTTP {r.status_code} {detail}"
        )
    try:
        payload = r.json()
    except ValueError as exc:
        raise SystemExit(
            f"PubTator query failed for {keyword!r}: invalid JSON response"
        ) from exc
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise SystemExit(
            f"PubTator query failed for {keyword!r}: JSON field 'results' is not a list"
        )
    return [x for x in results[:max_pmids] if isinstance(x, dict)]


def _hit_pmid(hit: dict) -> str | None:
    pmid = str(hit.get("pmid") or "").strip()
    return pmid or None


def _gene_symbols_from_hit(hit: dict) -> set[str]:
    text = " ".join(
        str(hit.get(key, ""))
        for key in ("text_hl", "title", "abstract")
        if hit.get(key) is not None
    )
    symbols: set[str] = set()
    for token in GENE_TAG_RE.findall(text):
        symbol = token.strip("_.-").upper()
        if symbol.isdigit() or not GENE_SYMBOL_RE.match(symbol):
            continue
        symbols.add(symbol)
    return symbols


def _is_reviewed_uniprot(entry: dict) -> bool:
    return str(entry.get("entryType", "")).lower().startswith("uniprotkb reviewed")


def _load_uniprot_cache(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key).upper(): (str(value) if value else None)
        for key, value in payload.items()
    }


def _write_uniprot_cache(path: Path, cache: dict[str, str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def map_gene_symbol_to_uniprot(
    symbol: str,
    *,
    session: requests.Session,
    cache: dict[str, str | None],
    sleep_s: float,
) -> str | None:
    symbol = symbol.strip().upper()
    if not symbol:
        return None
    if symbol in SEED_SYMBOL_TO_UNIPROT:
        cache.setdefault(symbol, SEED_SYMBOL_TO_UNIPROT[symbol])
        return SEED_SYMBOL_TO_UNIPROT[symbol]
    if symbol in cache:
        return cache[symbol]
    try:
        response = session.get(
            UNIPROT_BASE,
            params={
                "query": f"gene_exact:{symbol} AND organism_id:9606",
                "fields": "accession,gene_primary,gene_synonym",
                "format": "json",
                "size": "10",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"UniProt mapping failed for gene {symbol!r}: {exc}") from exc
    if response.status_code != 200:
        detail = getattr(response, "text", "")[:200]
        raise SystemExit(
            f"UniProt mapping failed for gene {symbol!r}: "
            f"HTTP {response.status_code} {detail}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise SystemExit(f"UniProt mapping failed for gene {symbol!r}: invalid JSON") from exc
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise SystemExit(
            f"UniProt mapping failed for gene {symbol!r}: JSON field 'results' is not a list"
        )
    results = [entry for entry in results if isinstance(entry, dict)]
    if not results:
        cache[symbol] = None
        return None
    results.sort(
        key=lambda entry: (
            0 if _is_reviewed_uniprot(entry) else 1,
            str(entry.get("primaryAccession", "")),
        )
    )
    accession = str(results[0].get("primaryAccession", "")).strip()
    cache[symbol] = accession or None
    if sleep_s:
        time.sleep(sleep_s)
    return cache[symbol]


def build_seed_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for category in EFFICACY_KEYWORDS:
        g.add_node(f"category:{category}",
                   type="EfficacyCategory", name=category)
    for uniprot, symbol, category, n_papers in SEED_GENE_EFFICACY:
        gene_node = f"gene:{uniprot}"
        g.add_node(gene_node, type="Gene", uniprot=uniprot, gene_symbol=symbol)
        g.add_edge(gene_node, f"category:{category}",
                   relation="ASSOCIATED_WITH", n_papers=0,
                   n_papers_seed=n_papers, n_papers_counted=0,
                   n_papers_basis="curated_seed")
    return g


def _append_semicolon_values(existing: object, values: set[str]) -> str:
    seen = {
        item.strip()
        for item in str(existing or "").split(";")
        if item.strip()
    }
    seen.update(str(value).strip() for value in values if str(value).strip())
    return ";".join(sorted(seen))


def _upsert_gene_category_edge(
    g: nx.MultiDiGraph,
    *,
    uniprot: str,
    symbol: str,
    category: str,
    pmids: set[str],
) -> bool:
    gene_node = f"gene:{uniprot}"
    cat_node = f"category:{category}"
    if gene_node not in g:
        g.add_node(
            gene_node,
            type="Gene",
            uniprot=uniprot,
            gene_symbol=symbol,
            gene_symbols=symbol,
        )
    else:
        attrs = g.nodes[gene_node]
        attrs.setdefault("type", "Gene")
        attrs.setdefault("uniprot", uniprot)
        if not attrs.get("gene_symbol"):
            attrs["gene_symbol"] = symbol
        attrs["gene_symbols"] = _append_semicolon_values(
            attrs.get("gene_symbols") or attrs.get("gene_symbol"),
            {symbol},
        )

    edge_data = g.get_edge_data(gene_node, cat_node, default={})
    for _key, attrs in edge_data.items():
        if attrs.get("relation") != "ASSOCIATED_WITH":
            continue
        existing_pmids = {
            item.strip()
            for item in str(attrs.get("sample_pmids", "")).split(";")
            if item.strip()
        }
        existing_pmids.update(pmids)
        had_seed = int(attrs.get("n_papers_seed", 0) or 0) > 0
        # Only observed, unique publication identifiers are paper counts.  The
        # curated seed weight remains available as a prior but must never be
        # displayed or ranked as a literature count.
        attrs["n_papers"] = len(existing_pmids)
        attrs["n_papers_counted"] = len(existing_pmids)
        attrs["n_papers_seed"] = int(attrs.get("n_papers_seed", 0) or 0)
        attrs["n_papers_basis"] = (
            "curated_seed_exceeds_counted"
            if attrs["n_papers_seed"] > len(existing_pmids)
            else "pubtator_counted"
        )
        attrs["sample_pmids"] = ";".join(sorted(existing_pmids)[:50])
        attrs["evidence_sources"] = _append_semicolon_values(
            attrs.get("evidence_sources"),
            {"seed", "pubtator"} if had_seed else {"pubtator"},
        )
        return False

    g.add_edge(
        gene_node,
        cat_node,
        relation="ASSOCIATED_WITH",
        n_papers=max(len(pmids), 1),
        n_papers_counted=len(pmids),
        n_papers_seed=0,
        n_papers_basis="pubtator_counted",
        sample_pmids=";".join(sorted(pmids)[:50]),
        evidence_sources="pubtator",
    )
    return True


def enrich_with_pubtator(
    g: nx.MultiDiGraph,
    *,
    uniprot_cache: dict[str, str | None],
    request_sleep_s: float,
) -> nx.MultiDiGraph:
    session = requests.Session()
    new_gene_edges = 0
    mapped_symbols: set[str] = set()
    unmapped_symbols: set[str] = set()
    for category, keywords in EFFICACY_KEYWORDS.items():
        for kw in keywords:
            hits = query_pubtator(kw)
            pmids = {pmid for hit in hits if (pmid := _hit_pmid(hit))}
            if not pmids:
                continue
            cat_node = f"category:{category}"
            existing = g.nodes[cat_node].get("pmid_count", 0)
            g.nodes[cat_node]["pmid_count"] = existing + len(pmids)
            g.nodes[cat_node]["sample_pmids"] = _append_semicolon_values(
                g.nodes[cat_node].get("sample_pmids"),
                set(sorted(pmids)[:20]),
            )
            symbol_pmids: dict[str, set[str]] = {}
            for hit in hits:
                pmid = _hit_pmid(hit)
                if not pmid:
                    continue
                for symbol in _gene_symbols_from_hit(hit):
                    symbol_pmids.setdefault(symbol, set()).add(pmid)
            for symbol, evidence_pmids in sorted(symbol_pmids.items()):
                uniprot = map_gene_symbol_to_uniprot(
                    symbol,
                    session=session,
                    cache=uniprot_cache,
                    sleep_s=request_sleep_s,
                )
                if not uniprot:
                    unmapped_symbols.add(symbol)
                    continue
                mapped_symbols.add(symbol)
                if _upsert_gene_category_edge(
                    g,
                    uniprot=uniprot,
                    symbol=symbol,
                    category=category,
                    pmids=evidence_pmids,
                ):
                    new_gene_edges += 1
            if request_sleep_s:
                time.sleep(request_sleep_s)
    g.graph["pubtator_mapped_symbols"] = len(mapped_symbols)
    g.graph["pubtator_unmapped_symbols"] = len(unmapped_symbols)
    g.graph["pubtator_new_gene_edges"] = new_gene_edges
    LOG.info(
        "PubTator enrichment mapped_symbols=%d unmapped_symbols=%d new_gene_edges=%d",
        len(mapped_symbols),
        len(unmapped_symbols),
        new_gene_edges,
    )
    return g


def gene_edge_count(g: nx.MultiDiGraph) -> int:
    return sum(1 for src, _dst, _attrs in g.edges(data=True) if str(src).startswith("gene:"))


def pubtator_backed_category_count(g: nx.MultiDiGraph) -> int:
    return sum(
        1
        for _node, attrs in g.nodes(data=True)
        if attrs.get("type") == "EfficacyCategory" and int(attrs.get("pmid_count", 0) or 0) > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-pubtator", action="store_true",
                        help="Build the seed graph only (no API calls).")
    parser.add_argument("--min-gene-edges", type=int, default=50)
    parser.add_argument("--request-sleep-s", type=float, default=0.1)
    parser.add_argument("--uniprot-cache", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "skin_efficacy.graphml"
    _remove_outputs(out)
    if args.min_gene_edges < 1:
        raise SystemExit(f"--min-gene-edges must be positive: {args.min_gene_edges}")
    if args.request_sleep_s < 0:
        raise SystemExit(f"--request-sleep-s must be non-negative: {args.request_sleep_s}")

    g = build_seed_graph()
    if not (args.dry_run or args.skip_pubtator):
        cache_path = args.uniprot_cache or args.out_dir / ".uniprot_gene_cache.json"
        cache = _load_uniprot_cache(cache_path)
        g = enrich_with_pubtator(
            g,
            uniprot_cache=cache,
            request_sleep_s=args.request_sleep_s,
        )
        _write_uniprot_cache(cache_path, cache)
        if pubtator_backed_category_count(g) == 0:
            raise SystemExit(
                "PubTator enrichment is required for full skin-efficacy KG ingest "
                "and returned no PMID-backed categories. Use --dry-run or "
                "--skip-pubtator only for explicit seed-graph diagnostics."
            )
        n_gene_edges = gene_edge_count(g)
        if n_gene_edges < args.min_gene_edges:
            raise SystemExit(
                "PubTator enrichment did not meet skin-efficacy KG quality gate: "
                f"{n_gene_edges} gene-category edges; required at least "
                f"{args.min_gene_edges}. Use --dry-run or --skip-pubtator only "
                "for explicit seed-graph diagnostics."
            )

    _write_graphml_atomic(g, out)
    LOG.info(
        "Wrote %s (nodes=%d, edges=%d, gene_edges=%d)",
        out,
        g.number_of_nodes(),
        g.number_of_edges(),
        gene_edge_count(g),
    )


if __name__ == "__main__":
    main()
