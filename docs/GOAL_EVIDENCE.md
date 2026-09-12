# Goal Evidence

Date: 2026-08-12

This document records the verified preimplementation baseline for SkinScout.
The machine-readable freeze is
[`BASELINE_FREEZE_20260812.json`](./BASELINE_FREEZE_20260812.json).

## Verified Scope

- `origin/main` history was merged into `codex/harden-workflow-gates` at
  `f5a871524634f03b49eef129f90667a299012c1d`. The merge changed history but
  not the working tree inherited from `1404451`.
- The existing test suite passes in the supported micromamba environment:
  `1891 passed`.
- Stage 0 strict claim-quality verification passes `18/18` checks.
- `target-id` fast data readiness and aggregate pipeline readiness return
  `status: ok` for ethanol (`CCO`).
- The current activity-retrieval scientific gate does **not** pass. The frozen
  evaluation records `passes_frozen_test_gate: false`.
- No completed target run currently passes the fresh verifier. The archived
  ethanol run predates required AutoDock provenance columns and is therefore
  stale and non-claimable.

## Authoritative Commands

Full regression baseline:

```bash
micromamba run -p /home/keunsoo/.local/share/micromamba/envs/cosmax-base \
  python -m pytest -q scripts/tests
```

Observed: `1891 passed in 526.83s`.

Stage 0 claim-quality baseline:

```bash
micromamba run -p /home/keunsoo/.local/share/micromamba/envs/cosmax-base \
  python scripts/stage0_verify.py --strict --claim-quality
```

Observed: `18/18 passed`.

Target-fast readiness:

```bash
micromamba run -p /home/keunsoo/.local/share/micromamba/envs/cosmax-base \
  python scripts/pipeline_readiness.py \
  --smiles CCO \
  --run-id ethanol_target_fast_full \
  --preset target-id \
  --mode fast \
  --json
```

Observed: `status: ok`, all data, Stage 0, safety, and required model groups
reported ready. This is not yet sufficient for Release 1 qualification because
the current readiness contract does not fail on the host NVML mismatch.

Fresh verification of the archived run:

```bash
micromamba run -p /home/keunsoo/.local/share/micromamba/envs/cosmax-base \
  python scripts/run_skinscout.py \
  --smiles CCO \
  --run-id ethanol_target_fast_full \
  --preset target-id \
  --mode fast \
  --no-use-conda \
  --verify-existing-run
```

Observed: failed closed because
`03_targets/mode_fast/autodock_top5k.tsv` lacks the current `engine`,
`degraded`, and map-coverage provenance columns. The corresponding goal
contract also fails because the underlying run verifier status is `failed`.

## Scientific Baseline

The selected frozen retrieval recipe is `union_p6_max`.

| Panel | Top-10 | Top-30 | MRR | Status |
| --- | ---: | ---: | ---: | --- |
| Temporal test 2025 | 0.7475 | 0.7967 | 0.6036 | temporal/calibration gate passes |
| Known skin panel, 15 cases | 0.2500 | 0.5000 | 0.1453 | required gate fails |
| Dual cold | 0.0000 | 0.0000 | 0.00028 | sparse diagnostic only |

Selected calibration is Brier `0.06834`, log loss `0.25637`, and ECE10
`0.02698`. These values do not support a SOTA claim.

## Host Qualification Boundary

The observed host is Ubuntu 26.04, Intel i7-13700K, 128 GiB RAM, and RTX 3080
Ti. `nvidia-smi` fails because the loaded NVIDIA kernel module is `595.71.05`
while the NVML user library is `595.84`; Torch CUDA still reports one usable
device. This exposes a real false-ready gap that Release 1 readiness must close.

This host is not the approved SLA reference machine. Ubuntu 24.04 plus RTX
4090 qualification remains required before a 60-minute release claim.

## Current Conclusion

The code and Stage 0 assets have a clean, reproducible preimplementation
baseline. The scientific model gate and fresh completed-run contract do not
pass, so the prior model and all non-claimable labels remain in force while the
approved Release 1 plan is implemented.
