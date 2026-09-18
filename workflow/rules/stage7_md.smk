# =============================================================================
# Stage 7 — GROMACS MD refinement + gmx_MMPBSA (top 3-5)
# =============================================================================

import shlex

S7_DIR = RUN_DIR / "07_md"


def _optional_flag(flag: str, value: str) -> str:
    value = str(value or "").strip()
    return f"{flag} {shlex.quote(value)}" if value else ""


rule gromacs_prep:
    input:
        ensemble = rules.ensemble_dock.output.consensus,
        receptor_manifest = rules.bioemu_ensemble.output.cluster_manifest,
        ligand_sdf = rules.xtb_optimize.output.sdf,
        boltz_poses = rules.boltz2_cofold.output.report
    output:
        manifest = S7_DIR / "md_manifest.tsv"
    log:
        "results/logs/{run_id}/stage7_prep.log".format(run_id=RUN_ID)
    conda:
        "../../envs/md.yml"
    params:
        out_dir = str(S7_DIR),
        top_n   = config["md"]["top_n_for_md"],
        duration_ns = config["md"]["duration_ns"],
        forcefield = config["md"]["forcefield"],
        gmx_command = config["md"].get("gmx_command", "gmx"),
        acpype_command = config["md"].get("acpype_command", "acpype"),
        complex_pose = lambda wildcards, input: _optional_flag(
            "--complex-pose", config["md"].get("complex_pose", "")
        ),
        complex_pose_manifest = lambda wildcards, input: _optional_flag(
            "--complex-pose-manifest",
            config["md"].get("complex_pose_manifest", "") or input.boltz_poses,
        ),
        ligand_topology = lambda wildcards, input: _optional_flag(
            "--ligand-topology", config["md"].get("ligand_topology", "")
        )
    shell:
        r"""
        mkdir -p {params.out_dir:q}
        python scripts/stage7_gromacs_prep.py \
            --ensemble-consensus {input.ensemble:q} \
            --receptor-manifest {input.receptor_manifest:q} \
            --ligand-sdf {input.ligand_sdf:q} \
            --out-dir {params.out_dir:q} \
            --top-n {params.top_n:q} \
            --duration-ns {params.duration_ns:q} \
            --forcefield {params.forcefield:q} \
            --gmx-command {params.gmx_command:q} \
            --acpype-command {params.acpype_command:q} \
            {params.complex_pose} \
            {params.complex_pose_manifest} \
            {params.ligand_topology} \
            --out-manifest {output.manifest:q} > {log:q} 2>&1
        """


rule gromacs_production:
    input:
        manifest = rules.gromacs_prep.output.manifest
    output:
        traj_index = S7_DIR / "trajectory_index.tsv"
    log:
        "results/logs/{run_id}/stage7_production.log".format(run_id=RUN_ID)
    conda:
        "../../envs/md.yml"
    resources:
        gpu = 1
    params:
        out_dir = str(S7_DIR),
        duration_ns = config["md"]["duration_ns"],
        replicas    = config["md"]["num_replicas"],
        seed        = config["md"].get("seed", 17391),
        gmx_command = config["md"].get("gmx_command", "gmx")
    shell:
        r"""
        python scripts/stage7_gromacs_run.py \
            --manifest {input.manifest:q} \
            --out-dir {params.out_dir:q} \
            --duration-ns {params.duration_ns:q} \
            --replicas {params.replicas:q} \
            --seed {params.seed:q} \
            --gmx-command {params.gmx_command:q} \
            --out-index {output.traj_index:q} > {log:q} 2>&1
        """


rule mmgbsa:
    input:
        traj_index = rules.gromacs_production.output.traj_index
    output:
        report = S7_DIR / "mmgbsa.tsv"
    log:
        "results/logs/{run_id}/stage7_mmgbsa.log".format(run_id=RUN_ID)
    conda:
        "../../envs/md.yml"
    shell:
        r"""
        python scripts/stage7_mmgbsa.py \
            --trajectory-index {input.traj_index:q} \
            --out-report {output.report:q} > {log:q} 2>&1
        """
