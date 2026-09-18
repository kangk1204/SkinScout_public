# Coverage routes — measured results, 2026-08-24

The retrieval model scores a target only if that target carries an activity
edge. On the committed 2026-08 index that is **4,510 of 20,204** screenable
targets (22.3%), and **0 of the 98** dual-cold truth targets. The dual-cold
`top10 = 0` is therefore a coverage result, not a ranking one.

Four routes were implemented and measured. All numbers below come from the
committed artifacts in this repository.

## D — separate "could not score" from "scored and missed"

Ranking rows now carry `in_scorable_universe`, and every ranking summary carries
a `coverage` block. Published metric definitions and values are unchanged; the
accounting is additive.

| | value |
|---|---|
| screenable universe | 20,204 |
| scorable (has an activity edge) | 4,510 (22.3%) |
| dual-cold truth targets inside that universe | 0 / 98 |

## A — sequence-cluster transfer (MMseqs)

Cluster maps for all 20,204 targets already existed, so this cost no new
compute. Discount 0.5.

| level | coverage | newly scorable | donors | dual-cold reached |
|---|---|---:|---:|---:|
| `target_cluster_30` | 0.223 → 0.426 | +4,087 | 827 | **0 / 98** |
| `target_cluster_50` | 0.223 → 0.327 | +2,090 | 677 | **0 / 98** |

The zero is definitional, not a weakness. The dual-cold view is *"test rows
whose pair, publication, scaffold, and both 30% and 50% target clusters are
absent from train/dev"*. Sequence-cluster transfer is excluded by construction
and has to be measured on the `ligand_scaffold_cold` view instead, which holds
out pair, publication and scaffold but not the target cluster.

## B — pocket-cluster transfer (Foldseek)

The pocket-fragment universe had been built against
`evidence_target_clusters_2026_02.csv` (4,595 targets), so it covered almost
exactly the population that already had ligand evidence: rebuilding the
transfer on it added **36** scorable targets. Re-running the same builder
against `screenable_target_clusters_2026_02.csv` raised the fragment universe
from **4,071 to 15,421** and the tm50 cluster count from 2,840 to 10,890.

Coverage, discount 0.5:

| level | coverage | newly scorable | dual-cold reached |
|---|---|---:|---:|
| `pocket_cluster_tm40` | 0.223 → 0.326 | +2,082 | **16 / 98** |
| `pocket_cluster_tm50` | 0.223 → 0.318 | +1,917 | 11 / 98 |
| `pocket_cluster_tm60` | 0.223 → 0.309 | +1,726 | 10 / 98 |

Pocket transfer reaches targets sequence transfer cannot, which is the expected
behaviour: Foldseek connects structurally similar sites across sequences too
distant for MMseqs.

End-to-end ranking on the 115 dual-cold queries / 163 truth pairs, recipe
`union_p6_consensus`, exclusion 0.85:

| config | top10 | top30 | MRR | median rank | scorable truth pairs |
|---|---:|---:|---:|---:|---:|
| none | 0 | 0 | 0.000084 | 11,885 | 0 / 163 |
| tm40, d 0.8 | 0 | 0 | 0.000378 | 12,840 | **30 / 163** |
| tm50, d 0.8 | 0 | 0 | 0.000354 | 12,763 | 18 / 163 |
| tm60, d 0.8 | 0 | 0 | 0.000298 | 12,666 | 11 / 163 |

**Coverage improves and ranking does not.** MRR rises 4.5×, but no truth target
enters the top 30, and the median rank gets *worse*: the same pass lifts ~1,900
other previously-zero targets, which demotes the 133 truth pairs transfer still
cannot reach.

Within the tier it does cover (tm40, median tier size 1,911), the 30 reachable
truth pairs sit at the **26.7th percentile** against 50% for random — real
signal, far too weak to use. Best rank inside the tier is 154; none reach the
top 30.

Pocket transfer is a coverage primitive, not a ranking solution. Merging it into
the primary ranking costs more than it returns; it belongs in a separate
hypothesis tier.

## C — per-receptor docking background

Ranking one compound against thousands of receptors by raw dG partly ranks
receptors. Measured on the 22 committed `autodock_top5k.tsv` runs, over the 224
receptors scored by at least 8 compounds:

- Leave-one-compound-out, a receptor's rank for a compound it has never seen is
  predicted at Spearman **rho 0.607** (median, range 0.184–0.792) by how it
  scored the other compounds. About **37%** of the receptor ordering for a new
  compound is receptor-intrinsic.
- The effect tracks ligand size, as the surface-area mechanism predicts:
  tretinoin 0.792, adapalene 0.729, ethanol 0.184.
- Scoring each query as its displacement below that receptor's own background —
  background built only from the other compounds, so nothing leaks — drops the
  correlation with receptor-intrinsic bias from **0.626 to 0.182**, removing
  **71%** of the bias.

Backgrounds are query-independent: computed once per receptor, reused for every
future compound.

### Storage was never the binding constraint

An earlier draft of this document treated the ~639 GB proteome-wide AutoGrid
cache as a blocker. It is not:

- This workstation has **17 TB free**; 639 GB is 3.7% of it.
- The production fast path never needs a proteome-wide cache. `config.yaml`
  sets `fast_mode_autodock_max_receptors: 256`, so only the selected Daina set
  is gridded — about **10.9 GB** at 30 Å.
- Vina computes its grids internally and needs no map cache at all.

The 300 GB figure is the README's *end-user install* requirement, which is a
distribution constraint, not a research one. Box size should therefore be chosen
on geometry, and on that basis 30 Å is the better default: a 20 Å box holds 90%
of pocket atoms for only 1.6% of targets against 82.8% at 30 Å, and buys just
14% in docking time. The 20 Å boxes and `scripts/resize_docking_boxes.py` are
kept for a low-spec distribution profile.

### The box was never derived from the pocket

`stage0_meeko_prep.write_box` sized every box as
`min(2 * (max(radius, 8) + 4), 30)`, and every `*.pockets.json` stores the
constant fallback `radius = 12.0` — the value is 12.0 in all 2,985 non-empty
pocket files checked. The expression collapsed to the 30 Å cap for all 15,038
receptors, so box size carried no geometric information at all.

P2Rank does publish the geometry: `*_predictions.csv` lists each pocket's
surface atom serials, and those resolve against the cleaned PDB with a 100% hit
rate. `scripts/pocket_box.py` sizes the box per axis to reach the outermost
rank-1 pocket surface atom plus 4 Å of ligand headroom, keeping P2Rank's centre
because that is where a ligand is predicted to sit.

Applied across the proteome (`scripts/derive_docking_boxes.py`):

| | value |
|---|---|
| boxes written | 15,456 |
| excluded (no rank-1 pocket) | 4,715 |
| median edge | 22.6 Å |
| median volume vs the 30 Å cube | **0.428×** |
| smaller than the 30 Å cube | 88.4% |
| clamped to the 16 Å floor / 40 Å ceiling | 1,451 / 633 |

The constant was wrong in both directions: too large for 88% of receptors,
wasting search volume on surface poses, and too small for the rest, truncating
the pocket. Measured over 500 targets, the fraction holding **every** rank-1
pocket surface atom:

| box | targets holding the whole pocket |
|---|---:|
| derived | **98.8%** |
| uniform 30 Å cube | 93.0% |
| uniform 20 Å cube | 66.8% |

The derived box is strictly better on both axes — 57% less volume in median and
higher pocket containment — which also cuts docking and AutoGrid cost roughly in
proportion to volume.

Stage 0 now derives the box the same way and refuses to guess: a target with no
P2Rank prediction is skipped rather than given a default cube, and the old
constant cube survives only as an explicit `legacy_cube_edge` opt-in.
`workflow/config.yaml` points `paths.docking_boxes` at the derived set; all
15,038 receptors that have a PDBQT also have a derived box, so nothing lost the
ability to dock.

#### Verified by a head-to-head run

Six known cosmetic ingredients were docked against a fixed 150-receptor panel
(11 literature targets plus 139 shared decoys) with both box sets, 900 dockings
each:

| | old 30 Å cube | derived |
|---|---:|---:|
| completed dockings | 900/900 | 900/900 |
| known targets in top 10 | 1/12 | 1/12 |
| known targets in top 30 | 5/12 | **6/12** |
| median known-target rank | 36 | **34** |
| MRR | 0.1131 | 0.1107 |

Recovery is unchanged within the noise of a 12-pair panel, which is the intended
result: the box got smaller without losing the pocket.

One pose out of 900 came back positive (+13.46 kcal/mol, kojic acid on `Q8N434`)
and was traced rather than dismissed. It is a sampling artefact, not a geometry
defect — that receptor's derived box is 38.3 × 29.0 × 36.7 Å, and Vina itself
warns above 27,000 Å³ that default exhaustiveness is insufficient:

| box | exhaustiveness | ΔG |
|---|---:|---:|
| derived | 8 (seed 42) | **+13.46** |
| derived | 8 (seed 7) | −4.99 |
| derived | 32 | **−5.24** |
| 30 Å cube | 8 or 32 | −4.53 … −5.25 |

At adequate sampling the derived box beats the cube. 1,799 of the 15,456 derived
boxes (11.6%) sit above that advisory volume; the manifest reports the count, and
the former uniform cube sat exactly at the threshold for every receptor.

#### Search effort now scales with the box

Since boxes are no longer a uniform cube, one configured effort no longer means
one sampling density. Calibrated on four derived boxes at 1.5–2.1× the advisory,
three ligands and six seeds each (360 dockings):

| effort | clashing poses | seed spread (sd) | best ΔG |
|---:|---:|---:|---:|
| 4 | 1/72 | 0.633 | −6.64 |
| 8 | 1/72 | 0.627 | −6.70 |
| 16 | 1/72 | 0.611 | −6.71 |
| **32** | **0/72** | **0.030** | −6.71 |
| 64 | 0/72 | 0.011 | −6.71 |

Convergence is a cliff at 32, not a slope: 4, 8 and 16 are equally bad, each
leaving a clash and roughly 0.6 kcal/mol of seed-to-seed spread — enough to
reorder targets separated by less than that. At 32 the clash disappears and the
spread falls 20-fold. A plain multiple of the configured effort would not reach
it from the fast path's base of 4, so the floor is absolute:

```
effort = base                                    if volume <= advisory
effort = max(ceil(base * volume / advisory), 32) otherwise
```

Exposed as `docking.search_volume_advisory` and `docking.high_volume_min_runs`,
forwarded by both docking rules. Cost across the real box set: **1.81×** for the
fast path (base 4) and **1.08×** for comprehensive (base 20), since only 11.6% of
boxes are raised and the comprehensive base already sits near the floor.

AutoDock-GPU encodes the box in its precomputed maps and never read the box file,
so a missing one leaves that receptor's effort unscaled and logged rather than
failing a run that could otherwise proceed. The calibration is Vina's; applying
it to AutoDock-GPU's `-nrun` is by analogy, not measurement.

### Engine choice: measured, not assumed

`gnina` v1.3.2 and an RTX 3080 Ti are available, so GPU docking was benchmarked
against CPU Vina on the same receptor and box:

| engine | small ligand | large ligand |
|---|---:|---:|
| gnina GPU, `--cnn_scoring none` | 0.9 s | 8.1 s |
| gnina GPU, `--cnn_scoring rescore` | 2.0 s | 9.2 s |
| Vina, 1 CPU core | ~2.5 s mean | — |

GNINA's conformational search is still CPU-bound — the GPU accelerates CNN
rescoring — so one serial GPU docking does not beat 15 parallel CPU Vina
processes on a 16-core host, and gnina additionally core-dumps on exit here.
GPU is the right tool for CNN *rescoring* of a shortlist, not for the screen
itself.

The real cost driver is ligand flexibility, not the engine: on one receptor,
docking time correlated with rotatable-bond count, and the dual-cold panel is
dominated by large flexible ligands (median 35 heavy atoms, up to 42 rotatable
bonds). Only **31 of the 115** dual-cold queries fall in a dockable size regime
(10–30 heavy atoms, ≤8 rotatable bonds) — the panel's chemistry is far from
SkinScout's own domain of small cosmetic ingredients, which is a scope limit on
any docking result measured against it.

### Measured cost, and why the screen was not run here

Per-docking cost was measured directly (AutoDock Vina python bindings,
exhaustiveness 8, 20 A box, one core, uncontended, receptor `Q9NV35`):

| ligand | time |
|---|---:|
| 1 | 54.7 s |
| 2 | 17.5 s |
| 3 | 14.0 s |
| 4 | 10.2 s |
| 5 | 5.0 s |

Mean 20.3 s, median 14.0 s. An earlier estimate of ~2 s came from timing
caffeine, which is far smaller and more rigid than the panel's drug-like
ligands; docking cost tracks rotatable bonds, so that probe understated the
real workload by an order of magnitude.

At 20.3 s per docking on this 16-core host:

| workload | dockings | core-hours | wall clock |
|---|---:|---:|---:|
| fast path, one compound (Daina top-256) | 256 | 1 | **~6 min** |
| proteome screen, one compound | 15,038 | 85 | ~5.3 h |
| proteome background panel (one-time) | 541,368 | 3,047 | **~7.9 days** |

This is the reason the end-to-end recovery experiment is not reported. Several
panel sizes were attempted (1,200, 600, 220 and 60 receptors) and none produced
a completed receptor within the session; the per-receptor cost is roughly
56 ligands x 20 s = ~19 minutes even at 20 A on a small receptor, before CPU
contention across 15 workers.

The shape of that table is the useful result. The production fast path is
already tractable — six minutes per compound — precisely because it docks only
the Daina-selected 256. What is expensive is the proteome-wide screen, and that
is exactly what closing the coverage hole would require: a one-time background
build of about eight days, then roughly five hours per compound. Route C is a
batch-scale commitment, not an interactive one.

Halving exhaustiveness to 4 roughly halves both figures and is defensible for a
ranking comparison, since docking noise applies equally to the raw and
normalised scores being compared.

## What the measurements change

1. Dual-cold `top10 = 0` must be read as coverage, not ranking. Route D makes
   that visible in the metric itself.
2. Sequence transfer cannot be evaluated on dual-cold at all. Use
   `ligand_scaffold_cold`.
3. Pocket transfer is the only route that reaches dual-cold targets, and it
   still does not rank them usably. Report it as a separate tier.
4. ~~Receptor-background normalisation is the remaining untested lever.~~
   **Tested 2026-08-29 and retired — see `docs/COVERAGE_ROUTE_C_RETIRED_20260829.md`.**
   The bias is real and normalisation removes it, but over 19 known pairs with
   leave-one-compound-out backgrounds it moves 9 pairs up and 10 down
   (Wilcoxon p = 0.83). De-biasing a signal that is already near chance for
   recovery leaves a signal that is near chance. The eight-day background build
   is not justified on this evidence. Route B (pocket-cluster transfer) is the
   only measured route that reaches dual-cold targets, and its prerequisite is
   an adequately sized pocket-cold panel, which does not exist yet.

## Evaluation caveat

Both transfer routes create a leakage surface: borrowing from cluster mates and
then evaluating on a panel whose clusters overlap train is circular. New
channels must be measured on pocket-cold splits. `eval/audit_pocket_leakage.py`
and the tm40/50/60 cold definitions already exist, but the current dev
pocket-cold rate is only 2.1%, so an adequately sized pocket-cold panel has to
be built deliberately.

## Floor

Nine of the 98 dual-cold truth targets (insulin `P01308`, ApoA1 `P02647`, ApoE
`P02649`, …) have no P2Rank pocket at all — secreted or disordered proteins. No
structural route reaches them, so they belong outside the denominator of any
coverage target.
