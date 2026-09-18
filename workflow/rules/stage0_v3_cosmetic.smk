# =============================================================================
# Stage 0 v3 extension — CosIng + Drug avoidance + Skin-efficacy KG
# =============================================================================

COSING_DIR  = Path(config["paths_v3"]["cosing"])
DRUG_DIR    = Path(config["paths_v3"]["drug_avoidance"])
KG_DIR      = Path(config["paths_v3"]["skin_kg"])
DRUG_CFG    = config.get("drug_avoidance", {})
COSING_CFG  = config.get("cosing", {})
DRUG_CHEMBL_DIR = Path(DRUG_CFG.get("chembl_dir", config["paths"]["chembl"]))
DRUG_ORANGE_BOOK = Path(DRUG_CFG.get("orange_book", DRUG_DIR / "orange_book.csv"))
DRUGBANK_APPROVED = Path(
    DRUG_CFG.get(
        "drugbank_approved_parquet",
        Path(config["paths"]["drugbank"]) / "drugbank_approved.parquet",
    )
)

# placeholder mode is an explicit CI/demo path. Default runs require real
# CosIng + approved-drug references plus live PubTator enrichment.
COSMETIC_DB_MODE = config.get("cosmetic_db_mode", "full")
if COSMETIC_DB_MODE not in {"full", "placeholder"}:
    raise ValueError(
        "cosmetic_db_mode must be 'full' or 'placeholder', "
        f"got {COSMETIC_DB_MODE!r}"
    )
_PLACEHOLDER = COSMETIC_DB_MODE == "placeholder"
COSING_AUTO_MIRROR = config_bool(COSING_CFG.get("auto_mirror_public_api"), True)
COSING_SOURCE_PATH = COSING_DIR / "cosing.csv"
COSING_SOURCE_INPUT = [] if _PLACEHOLDER else [COSING_SOURCE_PATH]
COSING_FLAG = "--dry-run" if _PLACEHOLDER else ""
DRUG_FLAG   = "--dry-run" if _PLACEHOLDER else ""
KG_FLAG     = "--skip-pubtator" if _PLACEHOLDER else ""


if COSING_AUTO_MIRROR:

    rule mirror_cosing_public_api:
        output:
            csv = COSING_SOURCE_PATH,
            manifest = COSING_DIR / "cosing_api_manifest.json"
        log:
            "results/logs/stage0_cosing_mirror.log"
        conda:
            "../../envs/base.yml"
        params:
            outdir = str(COSING_DIR)
        shell:
            r"""
            mkdir -p {params.outdir:q}
            python scripts/stage0_mirror_cosing.py \
                --out-dir {params.outdir:q} \
                --manifest {output.manifest:q} \
                --sleep-s 0.01 \
                --progress-every-pages 5 > {log:q} 2>&1
            """


rule cosing_ingest:
    input:
        COSING_SOURCE_INPUT
    output:
        parquet = COSING_DIR / "cosing.parquet"
    log:
        "results/logs/stage0_cosing.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(COSING_DIR),
        flag = COSING_FLAG
    shell:
        r"""
        mkdir -p {params.outdir:q}
        python scripts/stage0_cosing.py --out-dir {params.outdir:q} {params.flag} > {log:q} 2>&1
        """


rule drug_avoidance_ingest:
    input:
        chembl = rules.mirror_chembl.output.flag,
        drugbank = rules.mirror_drugbank.output.approved
    output:
        parquet  = DRUG_DIR / "drugs.parquet",
        scaffolds = DRUG_DIR / "scaffolds.parquet"
    log:
        "results/logs/stage0_drug_avoidance.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(DRUG_DIR),
        chembl_dir = str(DRUG_CHEMBL_DIR),
        orange_book = str(DRUG_ORANGE_BOOK),
        drugbank_parquet = str(DRUGBANK_APPROVED),
        flag = DRUG_FLAG
    shell:
        r"""
        mkdir -p {params.outdir:q}
        python scripts/stage0_drug_avoidance.py \
            --out-dir {params.outdir:q} \
            --chembl-dir {params.chembl_dir:q} \
            --orange-book {params.orange_book:q} \
            --drugbank-parquet {params.drugbank_parquet:q} \
            {params.flag} > {log:q} 2>&1
        """


rule skin_efficacy_kg:
    output:
        graphml = KG_DIR / "skin_efficacy.graphml"
    log:
        "results/logs/stage0_skin_kg.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(KG_DIR),
        flag = KG_FLAG
    shell:
        r"""
        mkdir -p {params.outdir:q}
        python scripts/stage0_skin_kg.py --out-dir {params.outdir:q} {params.flag} > {log:q} 2>&1
        """


rule stage0_v3_complete:
    input:
        rules.cosing_ingest.output.parquet,
        rules.drug_avoidance_ingest.output.parquet,
        rules.skin_efficacy_kg.output.graphml,
        rules.compute_skin_score.output.tsv,
    output:
        flag = str(MANIFESTS / "stage0_v3_complete.flag")
    shell:
        "touch {output.flag:q}"
