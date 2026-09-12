#!/usr/bin/env python3
"""stage5_5_plip.py — Run PLIP on every Boltz-2 complex; aggregate XML."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("stage5_5.plip")


def kept_complexes(report_path: Path) -> list[tuple[str, Path]]:
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise SystemExit(f"Boltz report is required and must be non-empty: {report_path}")
    report = pd.read_csv(report_path, sep="\t", skip_blank_lines=False)
    required = {"target_id", "complex_pdb", "kept"}
    missing = sorted(required - set(report.columns))
    if missing:
        raise SystemExit(f"Boltz report missing required columns: {missing}")
    if report.empty:
        raise SystemExit("Boltz report contains no rows")
    if not set(report["kept"].astype(str)).issubset({"yes", "no"}):
        raise SystemExit("Boltz report kept column must contain only yes/no")
    rows: list[tuple[str, Path]] = []
    for _, row in report[report["kept"] == "yes"].iterrows():
        target_id = str(row["target_id"]).strip()
        complex_pdb = Path(str(row["complex_pdb"]).strip())
        if not target_id or not complex_pdb.exists() or complex_pdb.stat().st_size == 0:
            raise SystemExit(f"Kept Boltz complex is missing or empty for {target_id}: {complex_pdb}")
        rows.append((target_id, complex_pdb))
    if not rows:
        raise SystemExit("Boltz report contains no kept complex poses")
    return rows


def run_plip(pdb: Path, out_dir: Path) -> Path | None:
    if not shutil.which("plip"):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.xml"
    report.unlink(missing_ok=True)
    cmd = ["plip", "-f", str(pdb), "-o", str(out_dir), "-x"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    return report if report.exists() and report.stat().st_size > 0 else None


def ligand_serial_to_index(pdb: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for line in pdb.read_text(errors="replace").splitlines():
        if not line.startswith("HETATM"):
            continue
        try:
            serial = int(line[6:11])
        except ValueError as exc:
            raise SystemExit(f"Boltz complex has invalid HETATM serial: {pdb}") from exc
        if serial in mapping:
            raise SystemExit(f"Boltz complex has duplicate HETATM serial {serial}: {pdb}")
        mapping[serial] = len(mapping)
    if not mapping:
        raise SystemExit(f"Boltz complex has no ligand HETATM records: {pdb}")
    return mapping


def _interaction_atom_serial(element: ET.Element, tag: str, xml: Path) -> int:
    text = element.findtext(tag)
    try:
        serial = int((text or "").strip())
    except ValueError as exc:
        raise SystemExit(f"PLIP {tag} must be an integer: {xml}") from exc
    if serial < 1:
        raise SystemExit(f"PLIP {tag} must be a positive PDB atom serial: {xml}")
    return serial


def _protein_is_donor(element: ET.Element, xml: Path) -> bool:
    value = (element.findtext("protisdon") or "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise SystemExit(f"PLIP protisdon must be True or False: {xml}")


def plip_ligand_serials(root: ET.Element, xml: Path) -> set[int]:
    serials: set[int] = set()
    scalar_paths = (
        (".//hydrophobic_interaction", "ligcarbonidx"),
        (".//halogen_bond", "don_idx"),
    )
    for path, tag in scalar_paths:
        for interaction in root.findall(path):
            serials.add(_interaction_atom_serial(interaction, tag, xml))
    for interaction in root.findall(".//hydrogen_bond"):
        tag = "acceptoridx" if _protein_is_donor(interaction, xml) else "donoridx"
        serials.add(_interaction_atom_serial(interaction, tag, xml))
    for interaction in root.findall(".//water_bridge"):
        tag = "acceptor_idx" if _protein_is_donor(interaction, xml) else "donor_idx"
        serials.add(_interaction_atom_serial(interaction, tag, xml))
    for path in (
        ".//salt_bridge",
        ".//pi_stack",
        ".//pi_cation_interaction",
    ):
        for interaction in root.findall(path):
            atom_elements = interaction.findall("./lig_idx_list/idx")
            if not atom_elements:
                raise SystemExit(f"PLIP interaction missing lig_idx_list/idx: {xml}")
            for atom_element in atom_elements:
                try:
                    serial = int((atom_element.text or "").strip())
                except ValueError as exc:
                    raise SystemExit(f"PLIP ligand idx must be an integer: {xml}") from exc
                if serial < 1:
                    raise SystemExit(f"PLIP ligand idx must be a positive PDB atom serial: {xml}")
                serials.add(serial)
    for interaction in root.findall(".//metal_complex"):
        if (interaction.findtext("location") or "").strip().lower() == "ligand":
            serials.add(_interaction_atom_serial(interaction, "target_idx", xml))
    return serials


def normalized_ligand_indices(xml: Path, complex_pdb: Path) -> set[int]:
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError as exc:
        raise SystemExit(f"PLIP XML is malformed: {xml}") from exc
    serial_to_index = ligand_serial_to_index(complex_pdb)
    indices: set[int] = set()
    for serial in plip_ligand_serials(root, xml):
        if serial not in serial_to_index:
            raise SystemExit(
                f"PLIP ligand atom serial {serial} is absent from Boltz HETATM records: {complex_pdb}"
            )
        indices.add(serial_to_index[serial])
    return indices


def merge_xmls(items: list[tuple[str, Path, Path]], out_xml: Path) -> None:
    root = ET.Element("plip_report")
    for tid, xml, complex_pdb in items:
        target = ET.SubElement(root, "target", attrib={"id": tid})
        for atom_idx in sorted(normalized_ligand_indices(xml, complex_pdb)):
            ET.SubElement(target, "ligand_atom", attrib={"idx": str(atom_idx)})
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tmp_xml = out_xml.with_suffix(out_xml.suffix + ".tmp")
    tmp_xml.unlink(missing_ok=True)
    ET.ElementTree(root).write(tmp_xml, encoding="utf-8", xml_declaration=True)
    tmp_xml.replace(out_xml)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boltz-dir", required=True, type=Path)
    parser.add_argument("--boltz-report", type=Path)
    parser.add_argument("--out-xml", required=True, type=Path)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Emit an explicit empty degraded XML when PLIP evidence is unavailable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_xml.unlink(missing_ok=True)
    if not args.boltz_dir.exists():
        raise SystemExit(f"Boltz output directory does not exist: {args.boltz_dir}")
    if not shutil.which("plip") and not args.allow_empty:
        raise SystemExit(
            "PLIP is required for Stage 5.5 pharmacophore evidence; "
            "use --allow-empty only for explicit degraded diagnostics"
        )

    items: list[tuple[str, Path, Path]] = []
    if not shutil.which("plip") and args.allow_empty:
        merge_xmls(items, args.out_xml)
        return
    if args.boltz_report is None:
        raise SystemExit("--boltz-report is required to select quality-approved complexes")
    complexes = kept_complexes(args.boltz_report)
    for target_id, complex_pdb in complexes:
        out = run_plip(complex_pdb, args.boltz_dir / target_id / "plip_out")
        if out is not None:
            items.append((target_id, out, complex_pdb))
    if not items and not args.allow_empty:
        raise SystemExit("No PLIP interaction reports were produced")
    merge_xmls(items, args.out_xml)
    LOG.info("PLIP report merged → %s (n=%d)", args.out_xml, len(items))


if __name__ == "__main__":
    main()
