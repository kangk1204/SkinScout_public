"""Minimal format checks for operator-supplied Stage 0 source files."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


STAGE0_SOURCE_KEYS_BY_LABEL = {
    "Stage 0 source: CosIng CSV": "cosing_csv",
    "Stage 0 source: DrugBank full database XML": "drugbank_xml",
    "Stage 0 source: skin proteome LFQ TSV": "skin_proteome_tsv",
    "Stage 0 source: GTEx gene TPM GCT": "gtex_gct",
}


@dataclass(frozen=True)
class Stage0SourceValidation:
    ok: bool
    detail: str


def _local_xml_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _validate_cosing_csv(path: Path) -> Stage0SourceValidation:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(handle, dialect)
            header = next(reader, None)
            if not header:
                return Stage0SourceValidation(False, "CosIng CSV has no header row")
            normalized = [col.strip().lower() for col in header]
            name_col = None
            for candidate in ("inci name", "inci_name", "name"):
                if candidate in normalized:
                    name_col = normalized.index(candidate)
                    break
            if name_col is None:
                return Stage0SourceValidation(
                    False,
                    "CosIng CSV must include an INCI name column "
                    "('inci name', 'inci_name', or 'name')",
                )
            rows = 0
            nonblank_names = 0
            for rows, row in enumerate(reader, start=1):
                if len(row) > name_col and row[name_col].strip():
                    nonblank_names += 1
                if rows >= 100:
                    break
            if rows == 0:
                return Stage0SourceValidation(False, "CosIng CSV contains no data rows")
            if nonblank_names == 0:
                return Stage0SourceValidation(
                    False,
                    "CosIng CSV sample contains no nonblank INCI names",
                )
            return Stage0SourceValidation(
                True,
                f"CosIng CSV header ok; sampled_rows={rows} "
                f"nonblank_inci_names={nonblank_names}",
            )
    except UnicodeDecodeError as exc:
        return Stage0SourceValidation(False, f"CosIng CSV is not UTF-8 text: {exc}")
    except OSError as exc:
        return Stage0SourceValidation(False, f"CosIng CSV could not be read: {exc}")


def _validate_skin_proteome_tsv(path: Path) -> Stage0SourceValidation:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            header_line = handle.readline()
            if not header_line:
                return Stage0SourceValidation(
                    False,
                    "skin proteome LFQ TSV has no header row",
                )
            header = header_line.rstrip("\n").split("\t")
            normalized = [col.strip().lower() for col in header]
            uid_col = None
            for candidate in ("uniprot", "accession"):
                if candidate in normalized:
                    uid_col = normalized.index(candidate)
                    break
            value_cols = [
                idx
                for idx, col in enumerate(normalized)
                if idx != uid_col and col.startswith(("lfq", "intensity"))
            ]
            if uid_col is None or not value_cols:
                return Stage0SourceValidation(
                    False,
                    "skin proteome LFQ TSV must include UniProt/accession and "
                    "LFQ/intensity columns",
                )
            rows = 0
            usable_rows = 0
            for rows, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                has_uid = len(fields) > uid_col and fields[uid_col].strip()
                has_value = any(
                    len(fields) > idx and fields[idx].strip()
                    for idx in value_cols
                )
                if has_uid and has_value:
                    usable_rows += 1
                if rows >= 100:
                    break
            if rows == 0:
                return Stage0SourceValidation(
                    False,
                    "skin proteome LFQ TSV contains no data rows",
                )
            if usable_rows == 0:
                return Stage0SourceValidation(
                    False,
                    "skin proteome LFQ TSV sample contains no usable LFQ rows",
                )
            return Stage0SourceValidation(
                True,
                f"skin proteome LFQ TSV header ok; sampled_rows={rows} "
                f"usable_rows={usable_rows}",
            )
    except UnicodeDecodeError as exc:
        return Stage0SourceValidation(
            False,
            f"skin proteome LFQ TSV is not UTF-8 text: {exc}",
        )
    except OSError as exc:
        return Stage0SourceValidation(
            False,
            f"skin proteome LFQ TSV could not be read: {exc}",
        )


def _validate_gtex_gct(path: Path) -> Stage0SourceValidation:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            version = handle.readline().strip()
            dims = handle.readline().strip()
            header_line = handle.readline()
            if not version or not dims or not header_line:
                return Stage0SourceValidation(
                    False,
                    "GTEx GCT must include version, dimension, and header rows",
                )
            if not version.startswith("#"):
                return Stage0SourceValidation(
                    False,
                    "GTEx GCT first line must be a GCT version marker",
                )
            header = header_line.rstrip("\n").split("\t")
            if "Name" not in header:
                return Stage0SourceValidation(
                    False,
                    "GTEx GCT header must include a Name column",
                )
            skin_cols = [col for col in header if "Skin" in col]
            if not skin_cols:
                return Stage0SourceValidation(
                    False,
                    "GTEx GCT header contains no skin sample columns",
                )
            first_data = handle.readline()
            if not first_data:
                return Stage0SourceValidation(False, "GTEx GCT contains no data rows")
            return Stage0SourceValidation(
                True,
                f"GTEx GCT header ok; skin_columns={len(skin_cols)}",
            )
    except UnicodeDecodeError as exc:
        return Stage0SourceValidation(False, f"GTEx GCT is not UTF-8 text: {exc}")
    except OSError as exc:
        return Stage0SourceValidation(False, f"GTEx GCT could not be read: {exc}")


def _validate_drugbank_xml(path: Path) -> Stage0SourceValidation:
    try:
        with path.open("rb") as handle:
            events = ET.iterparse(handle, events=("start",))
            root_seen = False
            for idx, (_event, elem) in enumerate(events):
                tag = _local_xml_tag(elem.tag)
                if not root_seen:
                    root_seen = True
                    if tag != "drugbank":
                        return Stage0SourceValidation(
                            False,
                            f"DrugBank XML root must be drugbank, got {tag!r}",
                        )
                elif tag == "drug":
                    return Stage0SourceValidation(
                        True,
                        f"DrugBank XML root ok; first drug element at event={idx}",
                    )
                if idx >= 1000:
                    break
            return Stage0SourceValidation(
                False,
                "DrugBank XML sample contains no drug elements",
            )
    except ET.ParseError as exc:
        return Stage0SourceValidation(False, f"DrugBank XML failed to parse: {exc}")
    except OSError as exc:
        return Stage0SourceValidation(False, f"DrugBank XML could not be read: {exc}")


def stage0_source_key(label: str) -> str | None:
    return STAGE0_SOURCE_KEYS_BY_LABEL.get(label)


def validate_stage0_source(label: str, path: Path) -> Stage0SourceValidation:
    key = stage0_source_key(label)
    if key == "cosing_csv":
        return _validate_cosing_csv(path)
    if key == "drugbank_xml":
        return _validate_drugbank_xml(path)
    if key == "skin_proteome_tsv":
        return _validate_skin_proteome_tsv(path)
    if key == "gtex_gct":
        return _validate_gtex_gct(path)
    return Stage0SourceValidation(True, "no Stage 0 source validator registered")
