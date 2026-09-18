# =============================================================================
# Additive performance-v2 branch
# =============================================================================
# This branch trains and scores the frozen-encoder P1 candidate. It never
# replaces the E0/Stage 3 ranking; predictions are attached to a separate CSV
# until the preregistered P0-P3 selection gate promotes a candidate.
# =============================================================================

PV2_ROOT = Path(PERFORMANCE_V2["output_dir"])
PV2_INPUTS = PV2_ROOT / "inputs"
PV2_EMBEDDINGS = PV2_ROOT / "embeddings"
PV2_MODEL = PV2_ROOT / "model"
PV2_RUN = RUN_DIR / "03_targets" / "performance_v2"
PV2_FIXTURE = PERFORMANCE_V2["mode"] == "fixture"
PV2_LIGAND_EMBEDDINGS = (
    PV2_EMBEDDINGS / "ligands.csv"
    if PV2_FIXTURE
    else PV2_EMBEDDINGS / "ligands"
)
PV2_TARGET_EMBEDDINGS = (
    PV2_EMBEDDINGS / "targets.csv"
    if PV2_FIXTURE
    else PV2_EMBEDDINGS / "targets"
)
PV2_MODEL_ARTIFACT = PV2_MODEL / ("model.npz" if PV2_FIXTURE else "model.pt")


rule performance_v2_prepare_inputs:
    input:
        benchmark = PERFORMANCE_V2["train_benchmark"],
        ligands = PERFORMANCE_V2["retrieval_ligands"],
        fasta = PERFORMANCE_V2["target_fasta"],
        clusters = PERFORMANCE_V2["target_clusters"]
    output:
        train = PV2_INPUTS / "train.csv",
        ligands = PV2_INPUTS / "ligands.csv",
        targets = PV2_INPUTS / "targets.csv",
        manifest = PV2_INPUTS / "manifest.json"
    log:
        "results/logs/performance_v2_prepare_inputs.log"
    conda:
        "../../envs/base.yml"
    shell:
        r"""
        python scripts/prepare_performance_v2_inputs.py \
            --benchmark-train {input.benchmark:q} \
            --retrieval-ligands {input.ligands:q} \
            --target-fasta {input.fasta:q} \
            --target-clusters {input.clusters:q} \
            --out-train {output.train:q} \
            --out-ligands {output.ligands:q} \
            --out-targets {output.targets:q} \
            --out-manifest {output.manifest:q} \
            > {log:q} 2>&1
        """


rule performance_v2_embeddings:
    input:
        ligands = rules.performance_v2_prepare_inputs.output.ligands,
        targets = rules.performance_v2_prepare_inputs.output.targets,
        inputs_manifest = rules.performance_v2_prepare_inputs.output.manifest
    output:
        ligands = (
            PV2_LIGAND_EMBEDDINGS
            if PV2_FIXTURE
            else directory(PV2_LIGAND_EMBEDDINGS)
        ),
        targets = (
            PV2_TARGET_EMBEDDINGS
            if PV2_FIXTURE
            else directory(PV2_TARGET_EMBEDDINGS)
        ),
        manifest = PV2_EMBEDDINGS / "manifest.json"
    log:
        "results/logs/performance_v2_embeddings.log"
    conda:
        "../../envs/performance_v2.yml"
    resources:
        gpu = 1,
        mem_mb = 24576
    params:
        fixture = "--fixture" if PV2_FIXTURE else "",
        ligand_batch = PERFORMANCE_V2["ligand_batch_size"],
        protein_batch = PERFORMANCE_V2["protein_batch_size"]
    shell:
        r"""
        python scripts/build_performance_v2_embeddings.py \
            --ligands {input.ligands:q} \
            --targets {input.targets:q} \
            --input-manifest {input.inputs_manifest:q} \
            --out-ligands {output.ligands:q} \
            --out-targets {output.targets:q} \
            --out-manifest {output.manifest:q} \
            --batch-size {params.ligand_batch} \
            --protein-batch-size {params.protein_batch} \
            {params.fixture} \
            > {log:q} 2>&1
        """


rule performance_v2_train_p1:
    input:
        train = rules.performance_v2_prepare_inputs.output.train,
        inputs_manifest = rules.performance_v2_prepare_inputs.output.manifest,
        embeddings = rules.performance_v2_embeddings.output.manifest
    output:
        model = PV2_MODEL_ARTIFACT,
        manifest = PV2_MODEL / "manifest.json",
        budget = PV2_MODEL / "budget.json"
    log:
        "results/logs/performance_v2_train.log"
    conda:
        "../../envs/performance_v2.yml"
    resources:
        gpu = 1,
        mem_mb = 24576
    params:
        fixture = "--fixture" if PV2_FIXTURE else "",
        seeds = ",".join(str(seed) for seed in PERFORMANCE_V2["seeds"]),
        epochs = PERFORMANCE_V2["epochs"],
        projection_dim = PERFORMANCE_V2["hidden_dim"],
        learning_rate = PERFORMANCE_V2["learning_rate"],
        batch_size = PERFORMANCE_V2["training_batch_size"]
    shell:
        r"""
        python scripts/train_performance_v2.py \
            --train {input.train:q} \
            --embedding-manifest {input.embeddings:q} \
            --out-model {output.model:q} \
            --out-manifest {output.manifest:q} \
            --out-budget {output.budget:q} \
            --seeds {params.seeds:q} \
            --epochs {params.epochs} \
            --projection-dim {params.projection_dim} \
            --learning-rate {params.learning_rate} \
            --batch-size {params.batch_size} \
            {params.fixture} \
            > {log:q} 2>&1
        """


rule performance_v2_score_query:
    input:
        compound = S1_DIR / "compound_canonical.json",
        model = rules.performance_v2_train_p1.output.model,
        model_manifest = rules.performance_v2_train_p1.output.manifest,
        budget = rules.performance_v2_train_p1.output.budget
    output:
        ranking = PV2_RUN / "ranking.csv",
        manifest = PV2_RUN / "ranking.manifest.json"
    log:
        "results/logs/{run_id}/performance_v2_score.log".format(run_id=RUN_ID)
    conda:
        "../../envs/performance_v2.yml"
    resources:
        gpu = 1,
        mem_mb = 16384
    shell:
        r"""
        python scripts/score_performance_v2.py \
            --model-manifest {input.model_manifest:q} \
            --compound-json {input.compound:q} \
            --out-ranking {output.ranking:q} \
            --out-manifest {output.manifest:q} \
            > {log:q} 2>&1
        """


rule performance_v2_attach_stage3:
    input:
        ranked = rules.kg_efficacy_label.output.csv,
        ranking = rules.performance_v2_score_query.output.ranking,
        ranking_manifest = rules.performance_v2_score_query.output.manifest
    output:
        csv = PV2_RUN / "ranked_targets_v3_with_efficacy_and_performance_v2.csv",
        manifest = PV2_RUN / "attachment.manifest.json"
    log:
        "results/logs/{run_id}/performance_v2_attach.log".format(run_id=RUN_ID)
    conda:
        "../../envs/performance_v2.yml"
    shell:
        r"""
        python scripts/attach_performance_v2_predictions.py \
            --ranked-csv {input.ranked:q} \
            --ranking-csv {input.ranking:q} \
            --ranking-manifest {input.ranking_manifest:q} \
            --candidate-id P1 \
            --out-csv {output.csv:q} \
            --out-manifest {output.manifest:q} \
            > {log:q} 2>&1
        """
