# =============================================================================
# Stage 4 — Target structure preparation for the top 50 candidates.
# =============================================================================
# Input: 03_targets/ranked_targets_v3_with_efficacy.csv + Stage 0 cleaned PDBs
# Output: 04_target/{uniprot}/{uniprot}_input.pdb, pocket_box.json
# =============================================================================

S4_DIR = RUN_DIR / "04_target"


def stage4_top_csv() -> str:
    return str(rules.kg_efficacy_label.output.csv)


rule prepare_top50_structures:
    input:
        top = stage4_top_csv()
    output:
        manifest = S4_DIR / "manifest.tsv"
    log:
        "results/logs/{run_id}/stage4_prep.log".format(run_id=RUN_ID)
    conda:
        # boltz2 를 선언하고 있었다. 그 환경은 torch, boltz, cuequivariance, dgl 을
        # 담은 수 GB짜리 GPU 환경인데, 이 스테이지는 RCSB 조회와 파일 준비만 한다.
        # base 로 충분하다 (pandas, requests, biopython).
        "../../envs/base.yml"
    params:
        clean_dir = str(AF_CLEAN),
        pocket_dir = str(POCKETS),
        cutoff_date = config["evaluation"]["common_cutoff_date"],
        out_dir = str(S4_DIR)
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage4_prepare_structures.py \
            --top-csv {input.top:q} \
            --clean-dir {params.clean_dir:q} \
            --pocket-dir {params.pocket_dir:q} \
            --cutoff-date {params.cutoff_date:q} \
            --out-dir {params.out_dir:q} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """
