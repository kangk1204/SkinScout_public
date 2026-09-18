#!/usr/bin/env python3
"""stage11_data_availability.py — Emit a journal-ready data availability MD.

Captures the public sources used by every stage of the pipeline + their
licenses, exactly as INSTRUCTIONS.md §17.2 requires.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

LOG = logging.getLogger("stage11.data_availability")

DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)

TEMPLATE = """\
# Data Availability

All training data sources used by this pipeline are publicly available and
listed below with their license. Pipeline source code is released under
Apache-2.0 (see LICENSE).

## Structural data

- **AlphaFold Protein Structure Database — human proteome v4**
  https://alphafold.ebi.ac.uk
  License: CC-BY-4.0 (attribution required)
  Citation: Varadi et al., *Nucleic Acids Res.* (2024).

- **Human Protein Atlas v25**
  https://www.proteinatlas.org
  License: CC-BY-SA-3.0
  Citation: Uhlén et al. (2024 update).

- **Skin proteome atlas** (Dyring-Andersen 2020)
  *Nat. Commun.* 11:5587. Open access.

- **GTEx v10** — bulk skin RNA-seq.
  https://gtexportal.org
  License: open (subject to GTEx data use policy).

## Chemical / Drug data

- **ChEMBL37** - human bioactivity annotations with source and assay provenance.
  License: CC-BY-SA-3.0.
- **BindingDB** — affinity / PDB-linked complex annotations.
- **Approved-drug reference** - ChEMBL37 / FDA Orange Book, optionally enriched
  with DrugBank open subset when licensed access is available.
- **FDA Orange Book** — public domain.
- **CosIng (EU Cosmetic Ingredient Database)** — EU public data.
- **ZINC-22** — commercial-availability tranche (Tingle et al. 2023).
- **PoseBusters Benchmark v2** — open, used as time-split eval (post-2023-10).
- **PLINDER-PL50** — open, ML-ready evaluation split.

## Knowledge graph

- **PubTator 3.0** — NLM, free API.
  Citation: Wei et al., *Nucleic Acids Res.* 52(W1):W540–W546 (2024).
- The derived `skin_efficacy.graphml` is released at
  `Zenodo (DOI: {skin_efficacy_kg_doi})` under CC-BY-4.0.

## Software

- All open-source dependencies, with pinned versions, are listed in
  Supplementary Table S1 and recorded under
  `results/runs/<run_id>/publication/reproducibility/tool_versions.lock`.
- Pipeline source: {pipeline_source_url} (Apache-2.0).
- Random seeds and config SHA-256 hash for full reproducibility are recorded
  in the same reproducibility directory.

## Exclusions

Where required by license, the following weights / databases are **not**
redistributed by this pipeline: AlphaFold3 weights, Chai-1 weights, and
licensed DrugBank tiers. Users must source these directly from the upstream
providers under their own license terms.
"""


def _validate_doi(doi: str, draft_ok: bool) -> None:
    if DOI_RE.fullmatch(doi) and "XXXX" not in doi.upper():
        return
    if draft_ok and doi == "pending":
        return
    raise SystemExit(
        "Data availability requires a real skin-efficacy KG DOI for "
        "claim-quality publication output; use --draft-doi-ok only for "
        "explicit drafts"
    )


def _validate_pipeline_source_url(url: str) -> None:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and len(path_parts) >= 2
    ):
        return
    raise SystemExit(
        "Data availability requires a public GitHub source URL: "
        f"{url}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument(
        "--skin-efficacy-kg-doi",
        default="pending",
        help="Zenodo DOI for the released skin_efficacy.graphml artifact",
    )
    parser.add_argument(
        "--pipeline-source-url",
        default="https://github.com/kangk1204/SkinScout_public",
        help="public source repository URL for the pipeline",
    )
    parser.add_argument(
        "--draft-doi-ok",
        action="store_true",
        help="allow a pending DOI only for explicit draft diagnostics",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_md.unlink(missing_ok=True)
    _validate_doi(args.skin_efficacy_kg_doi, args.draft_doi_ok)
    _validate_pipeline_source_url(args.pipeline_source_url)
    text = TEMPLATE.format(
        skin_efficacy_kg_doi=args.skin_efficacy_kg_doi,
        pipeline_source_url=args.pipeline_source_url,
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_md.with_suffix(args.out_md.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(args.out_md)
    LOG.info("Wrote %s (%d bytes)", args.out_md, len(text))


if __name__ == "__main__":
    main()
