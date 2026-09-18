#!/usr/bin/env bash
# stage0_mirror_bindingdb.sh
# Snapshot the latest BindingDB TSV release plus a post-cutoff slice for
# evaluation (PDB-linked complexes). A pinned release is required so the raw
# archive, extracted TSV, and source manifest remain reproducible.
# Usage: stage0_mirror_bindingdb.sh <OUTDIR> --release YYYY-MM [--allow-placeholder]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=python_runtime.sh
source "${SCRIPT_DIR}/python_runtime.sh"

OUTDIR="${1:?outdir required}"
ALLOW_PLACEHOLDER=0
RELEASE=""
shift || true
while [ "$#" -gt 0 ]; do
    case "$1" in
        --allow-placeholder)
            ALLOW_PLACEHOLDER=1
            shift
            ;;
        --release)
            RELEASE="${2:?--release requires YYYY-MM}"
            shift 2
            ;;
        *)
            echo "[stage0.bindingdb][FATAL] unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "${RELEASE}" ]; then
    echo "[stage0.bindingdb][FATAL] --release YYYY-MM is required for a pinned source manifest" >&2
    exit 2
fi
if ! [[ "${RELEASE}" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
    echo "[stage0.bindingdb][FATAL] --release must be YYYY-MM" >&2
    exit 2
fi

skinscout_resolve_python "import pandas, pyarrow"
mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

BASE="https://www.bindingdb.org/rwd/bind/downloads"
TSV="BindingDB_All.tsv"
MANIFEST="bindingdb_source_manifest.json"

cleanup_claim_artifacts() {
    rm -f "${TSV}" "${TSV}.tmp" BindingDB_All.tsv.*.tmp "bindingdb_with_pdb.parquet" "${MANIFEST}" "${MANIFEST}.tmp" bindingdb_source_manifest.json.*.tmp
}

validate_and_extract_release_zip() {
    local zip_path="$1"
    local release="$2"
    local url="$3"
    "${SKINSCOUT_PYTHON_CMD[@]}" - "$zip_path" "$release" "$url" "$MANIFEST" "$TSV" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
release = sys.argv[2]
url = sys.argv[3]
manifest_path = Path(sys.argv[4])
tsv_path = Path(sys.argv[5])

def sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

try:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        matches = [name for name in names if Path(name).name == "BindingDB_All.tsv"]
        if not matches:
            raise SystemExit("[stage0.bindingdb][FATAL] release archive lacks BindingDB_All.tsv")
        if len(matches) != 1:
            raise SystemExit("[stage0.bindingdb][FATAL] release archive has multiple BindingDB_All.tsv members")
        member = matches[0]
        info = archive.getinfo(member)
        if info.file_size <= 0:
            raise SystemExit("[stage0.bindingdb][FATAL] BindingDB_All.tsv in release archive is empty")
        fd, tmp_name = tempfile.mkstemp(prefix="BindingDB_All.tsv.", suffix=".tmp", dir=".")
        try:
            with os.fdopen(fd, "wb") as out, archive.open(member) as src:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    out.write(chunk)
            tmp_path = Path(tmp_name)
            extracted_sha = sha256_bytes(tmp_path)
            os.replace(tmp_path, tsv_path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
except zipfile.BadZipFile as exc:
    raise SystemExit(f"[stage0.bindingdb][FATAL] release archive is not a valid ZIP: {zip_path}") from exc

manifest = {
    "schema_version": 1,
    "source": {
        "name": "BindingDB",
        "official_url": "https://www.bindingdb.org/",
        "downloads_url": "https://www.bindingdb.org/rwd/bind/downloads",
        "license": "CC BY 3.0",
        "license_url": "https://www.bindingdb.org/rwd/bind/info.jsp",
        "citation_url": "https://www.bindingdb.org/rwd/bind/info.jsp",
        "redistribution": "allowed",
        "redistribution_requirements": "attribution",
    },
    "release": release,
    "release_date": f"{release}-01",
    "url": url,
    "archive": {
        "path": zip_path.name,
        "sha256": sha256_bytes(zip_path),
        "bytes": zip_path.stat().st_size,
    },
    "extracted": {
        "path": tsv_path.name,
        "member": member,
        "sha256": extracted_sha,
        "bytes": tsv_path.stat().st_size,
    },
}
fd, tmp_name = tempfile.mkstemp(prefix=f"{manifest_path.name}.", suffix=".tmp", dir=".")
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp_name, manifest_path)
print(f"[stage0.bindingdb] manifest written -> {manifest_path}")
PY
}

download_ok=0
ym="${RELEASE/-/}"
ZIP="BindingDB_All_${ym}_tsv.zip"
url="${BASE}/${ZIP}"
if [ ! -f "${ZIP}" ]; then
    if wget -c -nv -O "${ZIP}.tmp" "${url}"; then
        mv "${ZIP}.tmp" "${ZIP}"
        download_ok=1
    else
        rm -f "${ZIP}.tmp"
    fi
else
    download_ok=1
    echo "[stage0.bindingdb] reusing ${ZIP}"
fi

if [ "${download_ok}" -ne 1 ]; then
    cleanup_claim_artifacts
    if [ "${ALLOW_PLACEHOLDER}" -ne 1 ]; then
        echo "[stage0.bindingdb][FATAL] no reachable BindingDB release. Use --allow-placeholder only for explicit diagnostics." >&2
        exit 4
    fi
    echo "[stage0.bindingdb][WARN] no reachable BindingDB release — emitting explicit placeholder." >&2
    "${SKINSCOUT_PYTHON_CMD[@]}" - <<'PY'
import pandas as pd
pd.DataFrame(columns=["pdb_id", "uniprot", "smiles"]).to_parquet("bindingdb_with_pdb.parquet", index=False)
print("[stage0.bindingdb] placeholder written → bindingdb_with_pdb.parquet (0 rows)")
PY
    echo "[stage0.bindingdb] done (placeholder)."
    exit 0
fi

if ! validate_and_extract_release_zip "${ZIP}" "${RELEASE}" "${url}"; then
    cleanup_claim_artifacts
    rm -f "${ZIP}"
    exit 5
fi

# Eval slice: rows with PDB-derived target.
export ALLOW_PLACEHOLDER
if ! "${SKINSCOUT_PYTHON_CMD[@]}" - <<'PY'
import os
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
print("[stage0.bindingdb] slicing PDB-linked evaluation set …", flush=True)
source = "BindingDB_All.tsv"
output = "bindingdb_with_pdb.parquet"
tmp_output = output + ".tmp"
pdb_col = "PDB ID(s) for Ligand-Target Complex"
header = list(pd.read_csv(source, sep="\t", nrows=0).columns)
if pdb_col not in header:
    has_pdb_column = False
else:
    has_pdb_column = True

writer = None
rows_written = 0
try:
    if has_pdb_column:
        reader = pacsv.open_csv(
            source,
            read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
            parse_options=pacsv.ParseOptions(delimiter="\t"),
            convert_options=pacsv.ConvertOptions(
                column_types={column: pa.string() for column in header},
                strings_can_be_null=True,
            ),
        )
        for batch in reader:
            table = pa.Table.from_batches([batch])
            pdb_values = pc.fill_null(table[pdb_col], "")
            mask = pc.not_equal(pc.utf8_trim_whitespace(pdb_values), "")
            subset = table.filter(mask)
            if subset.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(tmp_output, subset.schema, compression="snappy")
            writer.write_table(subset)
            rows_written += subset.num_rows
finally:
    if writer is not None:
        writer.close()

if rows_written:
    os.replace(tmp_output, output)
    print(f"[stage0.bindingdb] rows-with-pdb: {rows_written:,}")
else:
    try:
        os.unlink(tmp_output)
    except FileNotFoundError:
        pass
    if os.environ.get("ALLOW_PLACEHOLDER") != "1":
        raise SystemExit("[stage0.bindingdb][FATAL] no PDB-linked BindingDB rows; use --allow-placeholder only for diagnostics")
    pd.DataFrame(columns=["pdb_id", "uniprot", "smiles"]).to_parquet("bindingdb_with_pdb.parquet", index=False)
    print("[stage0.bindingdb] no PDB column — placeholder written")
PY
then
    cleanup_claim_artifacts
    exit 6
fi

echo "[stage0.bindingdb] done."
