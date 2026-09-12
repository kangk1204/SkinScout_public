# =============================================================================
# Stage 0 v3 extension — Skin expression database
# =============================================================================

HPA_DIR             = Path(config["paths_v3"]["hpa"])
SKIN_PROTEOME_DIR   = Path(config["paths_v3"]["skin_proteome"])
GTEX_DIR            = Path(config["paths_v3"]["gtex"])
SC_DIR              = Path(config["paths_v3"]["scrnaseq"])
SKIN_EXPR_DIR       = Path(config["paths_v3"]["skin_expression"])
SKIN_EXPRESSION_CONFIG = config.get("skin_expression", {})
ALLOW_EMPTY_SKIN_SOURCES = config_bool(
    SKIN_EXPRESSION_CONFIG.get("allow_empty_sources"),
    False,
)
REQUIRE_SKIN_PROTEOME_SOURCE = config_bool(
    SKIN_EXPRESSION_CONFIG.get("require_proteome_source"),
    False,
)
REQUIRE_GTEX_SOURCE = config_bool(
    SKIN_EXPRESSION_CONFIG.get("require_gtex_source"),
    False,
)
SKIN_PROTEOME_SOURCE_PATH = SKIN_PROTEOME_DIR / "raw_lfq.tsv"
GTEX_SOURCE_PATH = GTEX_DIR / "gtex_v10_gene_tpm.gct"


def skin_source_input(source_path, required):
    if required and not ALLOW_EMPTY_SKIN_SOURCES:
        return [source_path]
    if Path(source_path).exists():
        return [source_path]
    return []


def allow_empty_skin_source(source_path, required):
    if ALLOW_EMPTY_SKIN_SOURCES:
        return "--allow-empty-placeholder"
    if not required and not Path(source_path).exists():
        return "--allow-empty-placeholder"
    return ""


SKIN_PROTEOME_SOURCE_INPUT = skin_source_input(
    SKIN_PROTEOME_SOURCE_PATH,
    REQUIRE_SKIN_PROTEOME_SOURCE,
)
GTEX_SOURCE_INPUT = skin_source_input(
    GTEX_SOURCE_PATH,
    REQUIRE_GTEX_SOURCE,
)


rule download_hpa:
    output:
        flag = HPA_DIR / ".download_complete",
        atlas = HPA_DIR / "proteinatlas.tsv",
        tissue = HPA_DIR / "rna_tissue_consensus.tsv",
        cell = HPA_DIR / "rna_single_cell_type.tsv"
    log:
        "results/logs/stage0_hpa.log"
    params:
        outdir = str(HPA_DIR)
    shell:
        r"""
        bash scripts/stage0_download_hpa.sh {params.outdir:q} > {log:q} 2>&1
        touch {output.flag:q}
        """


rule ingest_skin_proteome:
    input:
        SKIN_PROTEOME_SOURCE_INPUT
    output:
        flag = SKIN_PROTEOME_DIR / ".ingest_complete",
        tsv = SKIN_PROTEOME_DIR / "skin_proteome.tsv"
    log:
        "results/logs/stage0_skin_proteome.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(SKIN_PROTEOME_DIR),
        allow_empty = allow_empty_skin_source(
            SKIN_PROTEOME_SOURCE_PATH,
            REQUIRE_SKIN_PROTEOME_SOURCE,
        )
    shell:
        r"""
        python scripts/stage0_skin_proteome.py \
            --out-dir {params.outdir:q} \
            {params.allow_empty} > {log:q} 2>&1
        touch {output.flag:q}
        """


rule ingest_gtex_skin:
    input:
        GTEX_SOURCE_INPUT
    output:
        flag = GTEX_DIR / ".ingest_complete",
        tsv = GTEX_DIR / "skin_tpm.tsv"
    log:
        "results/logs/stage0_gtex.log"
    conda:
        "../../envs/base.yml"
    params:
        outdir = str(GTEX_DIR),
        allow_empty = allow_empty_skin_source(
            GTEX_SOURCE_PATH,
            REQUIRE_GTEX_SOURCE,
        )
    shell:
        r"""
        python scripts/stage0_gtex_skin.py \
            --out-dir {params.outdir:q} \
            {params.allow_empty} > {log:q} 2>&1
        touch {output.flag:q}
        """


rule compute_skin_score:
    input:
        hpa_tissue = rules.download_hpa.output.tissue,
        hpa_cell = rules.download_hpa.output.cell,
        proteome = rules.ingest_skin_proteome.output.tsv,
        gtex = rules.ingest_gtex_skin.output.tsv
    output:
        tsv = SKIN_EXPR_DIR / "skin_score.tsv"
    log:
        "results/logs/stage0_skin_score.log"
    conda:
        "../../envs/base.yml"
    params:
        hpa_dir       = str(HPA_DIR),
        proteome_dir  = str(SKIN_PROTEOME_DIR),
        gtex_dir      = str(GTEX_DIR),
        sc_dir        = str(SC_DIR),
        out_dir       = str(SKIN_EXPR_DIR),
        allow_empty = "--allow-empty-output"
            if ALLOW_EMPTY_SKIN_SOURCES
            else ""
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage0_skin_score.py \
            --hpa-dir {params.hpa_dir:q} \
            --proteome-dir {params.proteome_dir:q} \
            --gtex-dir {params.gtex_dir:q} \
            --sc-dir {params.sc_dir:q} \
            --out-tsv {output.tsv:q} \
            {params.allow_empty} > {log:q} 2>&1
        """
