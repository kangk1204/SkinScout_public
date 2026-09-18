#!/usr/bin/env bash
# stage0_mirror_drugbank.sh
# Mirror the DrugBank XML release. Requires a registered account → the script
# expects the XML to be placed at <OUTDIR>/drugbank_full_database.xml by the
# operator before invocation (license terms forbid automated download).
# Usage: stage0_mirror_drugbank.sh <OUTDIR> [--allow-placeholder|--allow-missing-optional]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=python_runtime.sh
source "${SCRIPT_DIR}/python_runtime.sh"

OUTDIR="${1:?outdir required}"
ALLOW_PLACEHOLDER=0
ALLOW_MISSING_OPTIONAL=0
for arg in "${@:2}"; do
    case "${arg}" in
        --allow-placeholder)
            ALLOW_PLACEHOLDER=1
            ;;
        --allow-missing-optional)
            ALLOW_MISSING_OPTIONAL=1
            ;;
        "")
            ;;
        *)
            echo "[stage0.drugbank][FATAL] unknown option: ${arg}" >&2
            exit 2
            ;;
    esac
done
mkdir -p "${OUTDIR}"

SRC="${OUTDIR}/drugbank_full_database.xml"
if [ ! -f "${SRC}" ]; then
    if [ "${ALLOW_PLACEHOLDER}" -ne 1 ] && [ "${ALLOW_MISSING_OPTIONAL}" -ne 1 ]; then
        cat >&2 <<EOF
[stage0.drugbank][FATAL] ${SRC} not found.
DrugBank requires a registered account (license forbids automated download).
Place the unzipped XML at:
    ${SRC}
or pass --allow-missing-optional when ChEMBL/Orange Book approved-drug
avoidance is sufficient for this run.
EOF
        exit 4
    fi
    skinscout_resolve_python "import pandas, pyarrow"
    if [ "${ALLOW_PLACEHOLDER}" -eq 1 ]; then
        mode="placeholder diagnostic"
    else
        mode="optional licensed enrichment absent"
    fi
    cat >&2 <<EOF
[stage0.drugbank][WARN] ${SRC} not found — emitting empty DrugBank slice (${mode}).
DrugBank requires a registered account (license forbids automated download).
To enable the full polypharmacology table, place the unzipped XML at:
    ${SRC}
then rerun:  snakemake --forcerun mirror_drugbank
See https://go.drugbank.com/releases for the latest download.
EOF
    export OUTDIR
    "${SKINSCOUT_PYTHON_CMD[@]}" - <<'PY'
import os
import pandas as pd
from pathlib import Path
outdir = Path(os.environ["OUTDIR"])
out = outdir / "drugbank_polypharm.parquet"
pd.DataFrame(columns=["drugbank_id", "drug_name", "uniprot"]).to_parquet(out, index=False)
approved = outdir / "drugbank_approved.parquet"
pd.DataFrame(columns=["drug_id", "name", "smiles"]).to_parquet(approved, index=False)
print(f"[stage0.drugbank] empty slice written → {out} (0 edges)")
print(f"[stage0.drugbank] empty slice written → {approved} (0 drugs)")
PY
    echo "[stage0.drugbank] done (${mode})."
    exit 0
fi

# Slices:
#   - polypharmacology table: one row per (drug, target) edge
#   - approved-drug table: one row per approved drug with a DrugBank SMILES
skinscout_resolve_python "import pandas, pyarrow"
export OUTDIR
"${SKINSCOUT_PYTHON_CMD[@]}" - <<'PY'
import os, xml.etree.ElementTree as ET, pandas as pd
from pathlib import Path
outdir = Path(os.environ["OUTDIR"])
src = outdir / "drugbank_full_database.xml"
print(f"[stage0.drugbank] parsing {src} (XML, ~1-3 GB) …", flush=True)
ns = {"db": "http://www.drugbank.ca"}
rows = []
approved_rows = []
ctx = ET.iterparse(src, events=("end",))
for ev, elem in ctx:
    if elem.tag.endswith("}drug"):
        did = elem.findtext("db:drugbank-id[@primary='true']", namespaces=ns)
        name = elem.findtext("db:name", namespaces=ns)
        groups = {
            str(group.text or "").strip().lower()
            for group in elem.findall("db:groups/db:group", ns)
        }
        if "approved" in groups:
            smiles = None
            for prop in elem.findall("db:calculated-properties/db:property", ns):
                kind = prop.findtext("db:kind", namespaces=ns)
                if kind == "SMILES":
                    smiles = prop.findtext("db:value", namespaces=ns)
                    break
            if did and name and smiles:
                approved_rows.append({
                    "drug_id": did,
                    "name": name,
                    "smiles": smiles,
                })
        for tgt in elem.findall("db:targets/db:target", ns):
            uid = None
            for ext in tgt.findall("db:polypeptide/db:external-identifiers/db:external-identifier", ns):
                if ext.findtext("db:resource", namespaces=ns) == "UniProtKB":
                    uid = ext.findtext("db:identifier", namespaces=ns)
                    break
            if uid:
                rows.append({"drugbank_id": did, "drug_name": name, "uniprot": uid})
        elem.clear()
df = pd.DataFrame(rows, columns=["drugbank_id", "drug_name", "uniprot"]).drop_duplicates()
out = outdir / "drugbank_polypharm.parquet"
df.to_parquet(out, index=False)
print(f"[stage0.drugbank] wrote {out} edges={len(df):,}")
approved = pd.DataFrame(
    approved_rows,
    columns=["drug_id", "name", "smiles"],
).drop_duplicates()
approved_out = outdir / "drugbank_approved.parquet"
approved.to_parquet(approved_out, index=False)
print(f"[stage0.drugbank] wrote {approved_out} drugs={len(approved):,}")
PY

echo "[stage0.drugbank] done."
