> **Archived unrelated reference:** This NanoForge bootstrap is not part of
> SkinScout and must not be extracted or executed in this repository.
> `INSTRUCTIONS.md` is the authoritative SkinScout specification.

# NanoForge v3 — Phase 0 Bootstrap Package

**Purpose**: Starter files for Codex CLI to initialize the project. Each section below contains a complete file with a header indicating the target path. Codex CLI should extract each file to the path specified.

**Companion to**: `NANOBODY_PIPELINE_SPEC_v3.md` (the authoritative spec).

**Usage with Codex CLI**:
```bash
codex "Read NANOBODY_PIPELINE_SPEC_v3.md as the authoritative spec.
Read NANOBODY_PIPELINE_v3_BOOTSTRAP.md and extract each file marked with
'### FILE: <path>' into the project at <path>.
Initialize the project per Phase 0 of the spec."
```

---

## Section A — Project root files

### FILE: `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nanoforge"
version = "0.1.0"
description = "Automated nanobody design & refinement pipeline (commercial-clean)"
readme = "README.md"
requires-python = ">=3.11,<3.13"
license = { file = "LICENSE" }
authors = [
    { name = "Keunsoo Kang Lab" },
]
keywords = ["nanobody", "VHH", "antibody-design", "bioinformatics", "structural-biology"]

dependencies = [
    # Core scientific
    "numpy>=1.26,<2.0",
    "scipy>=1.11",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "scikit-learn>=1.4",
    "biopython>=1.83",

    # Deep learning
    "torch>=2.4",
    "transformers>=4.40",

    # Workflow & infra
    "pydantic>=2.6",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "psycopg[binary]>=3.1",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "streamlit>=1.32",
    "click>=8.1",
    "pyyaml>=6.0",
    "loguru>=0.7",
    "rich>=13.7",

    # Bioinformatics dedicated
    "anarci",
    "freesasa>=2.2",
    "propka>=3.5",
    "mdtraj>=1.10",
    "openmm>=8.1",

    # Tracking
    "wandb>=0.17",
    "mlflow>=2.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "black>=24.4",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "pip-licenses>=4.4",
    "scancode-toolkit>=32.0",
]
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
]

[project.scripts]
nanoforge = "nanoforge.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["nanoforge"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "B", "UP", "RUF"]
ignore = ["E501"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=nanoforge --cov-report=term-missing"
```

### FILE: `LICENSE`

```
MIT License

Copyright (c) 2026 Keunsoo Kang Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### FILE: `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.venv/
.eggs/

# Project
runs/
data/patents/
data/uniref30/
models/*.pt
models/*.ckpt
containers/*.sif
.env

# Logs / cache
*.log
.cache/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Wandb / MLflow
wandb/
mlruns/
```

### FILE: `nextflow.config`

```groovy
// nextflow.config — NanoForge pipeline

manifest {
    name        = 'nanoforge'
    description = 'Automated nanobody design & refinement pipeline'
    version     = '0.1.0'
    mainScript  = 'main.nf'
    nextflowVersion = '>=24.04'
}

params {
    // Defaults; override via --param at CLI or -params-file YAML
    output_dir          = './runs'
    target_config       = null
    application_profile = 'general'
    fast_mode           = false
    gpu                 = 0
}

process {
    // Default resources per process; overridden per-process below
    cpus    = 4
    memory  = '32 GB'
    time    = '24h'

    // Containers
    container = 'containers/base_cuda.sif'

    // Resource labels
    withLabel: 'gpu' {
        clusterOptions = '--gres=gpu:1'
        memory = '64 GB'
    }
    withLabel: 'gpu_large' {
        clusterOptions = '--gres=gpu:1'
        memory = '128 GB'
        time = '48h'
    }
    withLabel: 'cpu_only' {
        cpus = 8
        memory = '16 GB'
    }

    // Per-module containers
    withName: 'boltz2.*'      { container = 'containers/boltz2.sif'      }
    withName: 'chai1.*'       { container = 'containers/chai1.sif'       }
    withName: 'rfantibody.*'  { container = 'containers/rfantibody.sif'  }
    withName: 'alphaflow.*'   { container = 'containers/alphaflow.sif'   }
    withName: 'mace.*'        { container = 'containers/mace.sif'        }
    withName: 'ani.*'         { container = 'containers/ani.sif'         }
    withName: 'aimnet2.*'     { container = 'containers/aimnet2.sif'     }
    withName: 'xtb.*'         { container = 'containers/xtb_pyscf.sif'   }
    withName: 'pyscf.*'       { container = 'containers/xtb_pyscf.sif'   }
    withName: 'gromacs.*'     { container = 'containers/gromacs_pmx.sif' }
    withName: 'ablang2.*'     { container = 'containers/ablang2.sif'     }
    withName: 'paragraph.*'   { container = 'containers/paragraph.sif'   }
    withName: 'biophi.*'      { container = 'containers/biophi.sif'      }
    withName: 'immunebuilder.*' { container = 'containers/immunebuilder.sif' }
}

executor {
    name = 'local'
    queueSize = 4
    cpus   = 16
    memory = '256 GB'
}

apptainer {
    enabled = true
    autoMounts = true
    cacheDir = "$PWD/containers/cache"
    runOptions = '--nv'  // GPU access
}

report {
    enabled = true
    file    = "${params.output_dir}/reports/report.html"
}

trace {
    enabled = true
    file    = "${params.output_dir}/reports/trace.txt"
}

timeline {
    enabled = true
    file    = "${params.output_dir}/reports/timeline.html"
}

dag {
    enabled = true
    file    = "${params.output_dir}/reports/dag.svg"
}
```

### FILE: `main.nf`

```groovy
#!/usr/bin/env nextflow

nextflow.enable.dsl=2

include { TARGET_PREP }       from './workflows/target_prep.nf'
include { ENSEMBLE }          from './workflows/ensemble.nf'
include { FTO_PRE }           from './workflows/fto_pre.nf'
include { LIBRARY_ROUTE }     from './workflows/library_route.nf'
include { DENOVO_ROUTE }      from './workflows/denovo_route.nf'
include { VALIDATE }          from './workflows/validate.nf'
include { PARATOPE_SANITY }   from './workflows/paratope_sanity.nf'
include { REFINE }            from './workflows/refine.nf'
include { MATURE }            from './workflows/mature.nf'
include { CROSSREACT }        from './workflows/crossreact.nf'
include { FILTER_RANK }       from './workflows/filter_rank.nf'
include { FTO_POST }          from './workflows/fto_post.nf'
include { BRIEF }             from './workflows/brief.nf'

workflow {
    target_ch = Channel.fromPath(params.target_config)

    FTO_PRE(target_ch)
    target_prep_out = TARGET_PREP(target_ch)
    ensemble_out    = ENSEMBLE(target_prep_out.target_info)

    lib_hits    = LIBRARY_ROUTE(ensemble_out.conformers)
    denovo_hits = DENOVO_ROUTE(ensemble_out.conformers)

    all_candidates = lib_hits.mix(denovo_hits)
    validated      = VALIDATE(all_candidates, ensemble_out.conformers)
    sanity_passed  = PARATOPE_SANITY(validated)
    refined        = REFINE(sanity_passed)
    matured        = MATURE(refined)
    crossreact_out = CROSSREACT(matured, target_prep_out.target_info)

    ranked = FILTER_RANK(crossreact_out)
    fto    = FTO_POST(ranked)
    BRIEF(fto)
}
```

### FILE: `pre-commit-config.yaml`

```yaml
# Save as .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict

  - repo: local
    hooks:
      - id: license-audit
        name: license-audit
        entry: python scripts/license_audit.py
        language: system
        pass_filenames: false
        always_run: true
```

---

## Section B — Installation script

### FILE: `scripts/install_models.sh`

```bash
#!/bin/bash
# install_models.sh — Download all model weights for NanoForge
# Idempotent: skips models already present with correct SHA256
# Run from project root: bash scripts/install_models.sh
#
# Estimated total size: ~80 GB
# Estimated time: 1-3 hours depending on bandwidth

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
mkdir -p "${MODELS_DIR}"

# Utility
download_if_missing() {
    local url="$1"
    local dest="$2"
    local expected_sha256="${3:-}"

    if [[ -f "${dest}" ]]; then
        if [[ -n "${expected_sha256}" ]]; then
            actual=$(sha256sum "${dest}" | awk '{print $1}')
            if [[ "${actual}" == "${expected_sha256}" ]]; then
                echo "✓ ${dest} (cached, SHA matches)"
                return 0
            else
                echo "⚠ ${dest} SHA mismatch; re-downloading"
                rm "${dest}"
            fi
        else
            echo "✓ ${dest} (cached, no SHA pin)"
            return 0
        fi
    fi

    echo "→ downloading: ${url}"
    mkdir -p "$(dirname "${dest}")"
    wget --show-progress -O "${dest}" "${url}"

    if [[ -n "${expected_sha256}" ]]; then
        actual=$(sha256sum "${dest}" | awk '{print $1}')
        if [[ "${actual}" != "${expected_sha256}" ]]; then
            echo "❌ SHA mismatch after download: ${dest}"
            echo "  expected: ${expected_sha256}"
            echo "  actual:   ${actual}"
            exit 1
        fi
    fi
}

echo "================================================"
echo "NanoForge model weights download"
echo "================================================"
echo ""
echo "WARNING: SHA256 hashes below are placeholders."
echo "On first run, hashes will be recorded for pinning."
echo "Subsequent runs verify against pinned hashes."
echo ""

#─────────────────────────────────────────────────────
# Boltz-2 (MIT)
#─────────────────────────────────────────────────────
echo "── Boltz-2 (MIT) ──"
mkdir -p "${MODELS_DIR}/boltz2"
download_if_missing \
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_model.ckpt" \
    "${MODELS_DIR}/boltz2/boltz2_model.ckpt"

#─────────────────────────────────────────────────────
# Chai-1 (Apache 2.0)
#─────────────────────────────────────────────────────
echo "── Chai-1 (Apache 2.0) ──"
mkdir -p "${MODELS_DIR}/chai1"
# Chai-1 weights are downloaded on first inference via chai_lab library; nothing to do here
echo "  (downloaded on first chai_lab inference)"

#─────────────────────────────────────────────────────
# ESM-2 / ESMFold (MIT)
#─────────────────────────────────────────────────────
echo "── ESM-2 / ESMFold (MIT) ──"
mkdir -p "${MODELS_DIR}/esm"
download_if_missing \
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt" \
    "${MODELS_DIR}/esm/esm2_t33_650M_UR50D.pt"
download_if_missing \
    "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt" \
    "${MODELS_DIR}/esm/esmfold_3B_v1.pt"

#─────────────────────────────────────────────────────
# RFantibody (MIT)
#─────────────────────────────────────────────────────
echo "── RFantibody (MIT) ──"
mkdir -p "${MODELS_DIR}/rfantibody"
# RFantibody official download script
if [[ ! -f "${MODELS_DIR}/rfantibody/.installed" ]]; then
    git clone --depth 1 https://github.com/RosettaCommons/RFantibody.git "${MODELS_DIR}/rfantibody/repo"
    bash "${MODELS_DIR}/rfantibody/repo/scripts/download_weights.sh"
    touch "${MODELS_DIR}/rfantibody/.installed"
else
    echo "✓ RFantibody already installed"
fi

#─────────────────────────────────────────────────────
# AlphaFlow / ESMFlow (MIT)
#─────────────────────────────────────────────────────
echo "── AlphaFlow / ESMFlow (MIT) ──"
mkdir -p "${MODELS_DIR}/alphaflow"
# Default: ESMFlow-MD+Templates 12l-distilled (fastest, no MSA needed)
download_if_missing \
    "https://storage.googleapis.com/alphaflow/params/esmflow_md_templates_distilled_202402.pt" \
    "${MODELS_DIR}/alphaflow/esmflow_md_templates_distilled.pt"

# Also fetch AlphaFlow-MD+Templates 12l-distilled (alt, MSA required)
download_if_missing \
    "https://storage.googleapis.com/alphaflow/params/alphaflow_12l_md_templates_distilled_202406.pt" \
    "${MODELS_DIR}/alphaflow/alphaflow_12l_md_templates_distilled.pt"

# AlphaFlow uses base AF2 weights internally for the AF variant
download_if_missing \
    "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
    "${MODELS_DIR}/alphaflow/alphafold_params_2022-12-06.tar"

#─────────────────────────────────────────────────────
# AbLang2 (BSD-3) — base for VHH-nativeness head
#─────────────────────────────────────────────────────
echo "── AbLang2 (BSD-3) ──"
mkdir -p "${MODELS_DIR}/ablang2"
# Downloaded via HuggingFace on first use
echo "  (downloaded on first transformers.from_pretrained call)"
echo "  ⚠ One-time fine-tuning on OAS camelid required (Phase 2)"
echo "  See: scripts/train_vhh_nativeness.sh"

#─────────────────────────────────────────────────────
# Paragraph (BSD-3)
#─────────────────────────────────────────────────────
echo "── Paragraph (BSD-3) ──"
# Bundled with pip install Paragraph
echo "  (installed via pip; weights bundled in package)"

#─────────────────────────────────────────────────────
# NanoBodyBuilder2 / ImmuneBuilder (BSD-3)
#─────────────────────────────────────────────────────
echo "── ImmuneBuilder / NanoBodyBuilder2 (BSD-3) ──"
mkdir -p "${MODELS_DIR}/immunebuilder"
echo "  (downloaded on first ImmuneBuilder.predict call)"

#─────────────────────────────────────────────────────
# MACE-OFF (MIT)
#─────────────────────────────────────────────────────
echo "── MACE-OFF (MIT) ──"
mkdir -p "${MODELS_DIR}/mace"
download_if_missing \
    "https://github.com/ACEsuit/mace-off/releases/download/v23.07.0/MACE-OFF23_medium.model" \
    "${MODELS_DIR}/mace/MACE-OFF23_medium.model"

# Also MACE-MP-0 for universal element coverage
download_if_missing \
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE-MP-0_medium.model" \
    "${MODELS_DIR}/mace/MACE-MP-0_medium.model"

#─────────────────────────────────────────────────────
# AIMNet2 (MIT)
#─────────────────────────────────────────────────────
echo "── AIMNet2 (MIT) ──"
mkdir -p "${MODELS_DIR}/aimnet2"
download_if_missing \
    "https://github.com/zubatyuk/aimnet2-asf/releases/download/v1.0/aimnet2_b973c_d3_ens.jpt" \
    "${MODELS_DIR}/aimnet2/aimnet2_b973c_d3_ens.jpt"

#─────────────────────────────────────────────────────
# TorchANI / ANI-2x (MIT)
#─────────────────────────────────────────────────────
echo "── TorchANI / ANI-2x (MIT) ──"
# Bundled with pip install torchani
echo "  (installed via pip)"

#─────────────────────────────────────────────────────
# MHCflurry (Apache 2.0)
#─────────────────────────────────────────────────────
echo "── MHCflurry (Apache 2.0) ──"
if [[ ! -f "${MODELS_DIR}/mhcflurry/.installed" ]]; then
    mkdir -p "${MODELS_DIR}/mhcflurry"
    mhcflurry-downloads fetch
    touch "${MODELS_DIR}/mhcflurry/.installed"
fi

#─────────────────────────────────────────────────────
# MMseqs2 / ColabFold-search (MIT)
#─────────────────────────────────────────────────────
echo "── MMseqs2 (MIT) ──"
# Installed via container; database download is separate step
echo "  Container: containers/base_cuda.sif (bundled)"
echo "  Note: UniRef30 database (~500 GB) — run scripts/download_uniref30.sh separately if local MSA needed"

#─────────────────────────────────────────────────────
# Summary
#─────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "Model downloads complete"
echo "================================================"
echo ""
echo "Total disk usage:"
du -sh "${MODELS_DIR}" || true
echo ""
echo "Next steps:"
echo "  1. Build Apptainer containers: bash scripts/build_containers.sh"
echo "  2. Initialize database: bash scripts/init_db.sh"
echo "  3. Build VHH library:  bash scripts/build_vhh_library.sh"
echo "  4. (Phase 2) Fine-tune AbLang2 on OAS camelid: bash scripts/train_vhh_nativeness.sh"
```

### FILE: `scripts/build_containers.sh`

```bash
#!/bin/bash
# Build all Apptainer containers from .def files
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFS_DIR="${PROJECT_ROOT}/containers"
SIF_DIR="${PROJECT_ROOT}/containers"

# Build base first; others may depend
DEFS=(
    "base_cuda.def"
    "boltz2.def"
    "chai1.def"
    "rfantibody.def"
    "alphaflow.def"
    "mace.def"
    "ani.def"
    "aimnet2.def"
    "xtb_pyscf.def"
    "gromacs_pmx.def"
    "ablang2.def"
    "paragraph.def"
    "biophi.def"
    "immunebuilder.def"
    "freebindcraft.def"
)

for def in "${DEFS[@]}"; do
    sif="${def%.def}.sif"
    sif_path="${SIF_DIR}/${sif}"
    def_path="${DEFS_DIR}/${def}"

    if [[ ! -f "${def_path}" ]]; then
        echo "⊘ ${def} (not present, skipping)"
        continue
    fi

    if [[ -f "${sif_path}" ]] && [[ "${sif_path}" -nt "${def_path}" ]]; then
        echo "✓ ${sif} (newer than def, skipping)"
        continue
    fi

    echo "→ building ${sif}..."
    apptainer build --fakeroot "${sif_path}" "${def_path}"
    echo "✓ built ${sif}"
done

echo "All containers built."
```

### FILE: `scripts/init_db.sh`

```bash
#!/bin/bash
# Initialize PostgreSQL database
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DB_NAME="${NANOFORGE_DB_NAME:-nanoforge}"
DB_USER="${NANOFORGE_DB_USER:-nanoforge}"
DB_HOST="${NANOFORGE_DB_HOST:-localhost}"
DB_PORT="${NANOFORGE_DB_PORT:-5432}"

# Create user + database if needed
psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres <<EOF
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
        CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${NANOFORGE_DB_PASSWORD:-changeme}';
    END IF;
END
\$\$;
EOF

psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres <<EOF
SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}');
\gexec
EOF

# Apply schema via Alembic
cd "${PROJECT_ROOT}"
export DATABASE_URL="postgresql+psycopg://${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
alembic upgrade head

# Initial license register seed
python -c "
from nanoforge.db.models import LicenseRegister, get_session
from nanoforge.utils.licenses import seed_license_register
with get_session() as session:
    seed_license_register(session)
    session.commit()
"

echo "Database initialized."
```

### FILE: `scripts/license_audit.py`

```python
"""License audit — runs in CI to prevent introduction of non-commercial deps."""

import json
import subprocess
import sys
from pathlib import Path

# Known commercial-OK SPDX identifiers
COMMERCIAL_OK = {
    "MIT", "MIT License",
    "BSD-3-Clause", "BSD-2-Clause", "BSD License",
    "Apache-2.0", "Apache 2.0",
    "ISC", "ISC License",
    "Python Software Foundation License",
    "PostgreSQL License",
}

# Known commercial-OK with caveats (LGPL: OK for dynamic linking only)
COMMERCIAL_WITH_CAVEAT = {
    "LGPL-2.1", "LGPL-3.0", "LGPL-2.1-or-later", "LGPL-3.0-or-later",
    "MPL-2.0",
}

# Explicit denylist
DENY = {
    "CC-BY-NC-SA-4.0", "CC-BY-NC-4.0",
    "GPL-2.0", "GPL-3.0",
    "AGPL-3.0",
    "Proprietary",
    "UNKNOWN",
}

REGISTER_PATH = Path("LICENSES/REGISTER.md")


def check_pip_licenses() -> int:
    """Run pip-licenses and validate against allowlist."""
    result = subprocess.run(
        ["pip-licenses", "--format=json", "--with-license-file"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(result.stdout)

    failures = []
    warnings = []

    for pkg in packages:
        name = pkg["Name"]
        license_ = pkg["License"]

        if license_ in DENY:
            failures.append(f"{name}: {license_} (DENY)")
        elif license_ in COMMERCIAL_OK:
            continue
        elif license_ in COMMERCIAL_WITH_CAVEAT:
            warnings.append(f"{name}: {license_} (allowed with dynamic-linking caveat)")
        else:
            warnings.append(f"{name}: {license_} (unrecognized; manual review needed)")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  ❌ {f}")
        print(f"\n{len(failures)} package(s) have disallowed licenses. CI fails.")
        return 1

    print(f"License audit passed: {len(packages)} packages checked.")
    return 0


def check_register_consistency() -> int:
    """Verify LICENSES/REGISTER.md exists and has entries for major deps."""
    if not REGISTER_PATH.exists():
        print(f"❌ {REGISTER_PATH} missing")
        return 1
    content = REGISTER_PATH.read_text()
    required_entries = [
        "Boltz-2", "RFantibody", "AlphaFlow", "MACE-OFF", "AbLang2",
        "Paragraph", "BioPhi", "OpenMM", "GROMACS",
    ]
    missing = [e for e in required_entries if e not in content]
    if missing:
        print(f"❌ REGISTER.md missing entries: {missing}")
        return 1
    print(f"✓ REGISTER.md has all required entries")
    return 0


if __name__ == "__main__":
    rc = max(check_pip_licenses(), check_register_consistency())
    sys.exit(rc)
```

---

## Section C — Apptainer container definitions

### FILE: `containers/base_cuda.def`

```
Bootstrap: docker
From: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

%labels
    org.nanoforge.role base
    org.nanoforge.cuda 12.4.1

%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        git wget curl unzip build-essential cmake pkg-config \
        libhmmer-dev hmmer ncbi-blast+ \
        libpq-dev \
        libxml2-dev libxslt-dev \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    pip install --no-cache-dir --upgrade pip setuptools wheel

    # Core scientific Python
    pip install --no-cache-dir \
        numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 \
        biopython==1.83 mdtraj==1.10.0 \
        torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

    # MMseqs2
    cd /opt && \
        wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz && \
        tar xzf mmseqs-linux-gpu.tar.gz && rm mmseqs-linux-gpu.tar.gz && \
        ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs

%environment
    export PYTHONUNBUFFERED=1
    export PATH=/opt/mmseqs/bin:$PATH

%runscript
    exec python "$@"
```

### FILE: `containers/boltz2.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%labels
    org.nanoforge.role boltz2
    org.nanoforge.license MIT

%post
    pip install --no-cache-dir boltz==2.0.0
    # Boltz weights downloaded on first inference; user can pre-stage to /models/boltz2
    mkdir -p /models/boltz2

%environment
    export BOLTZ_WEIGHTS_DIR=/models/boltz2

%runscript
    exec boltz "$@"
```

### FILE: `containers/rfantibody.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%labels
    org.nanoforge.role rfantibody
    org.nanoforge.license MIT

%post
    cd /opt && \
        git clone --depth 1 https://github.com/RosettaCommons/RFantibody.git && \
        cd RFantibody && \
        pip install --no-cache-dir -e .
    # Note: RFantibody weights downloaded via its own scripts/download_weights.sh
    # which is called by scripts/install_models.sh in NanoForge

%environment
    export RFANTIBODY_HOME=/opt/RFantibody
    export PYTHONPATH=/opt/RFantibody:$PYTHONPATH

%runscript
    cd /opt/RFantibody && python -m rfantibody "$@"
```

### FILE: `containers/alphaflow.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%labels
    org.nanoforge.role alphaflow
    org.nanoforge.license MIT

%post
    pip install --no-cache-dir \
        biopython==1.79 \
        dm-tree==0.1.6 \
        modelcif==0.7 \
        ml-collections==0.1.0 \
        pytorch-lightning==2.0.4 \
        fair-esm einops absl-py

    # OpenFold (Apache 2.0) — AlphaFlow dependency
    pip install --no-cache-dir 'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037'

    # AlphaFlow itself
    cd /opt && \
        git clone --depth 1 https://github.com/bjing2016/alphaflow.git && \
        cd alphaflow && \
        pip install --no-cache-dir -e .

%environment
    export ALPHAFLOW_HOME=/opt/alphaflow
    export PYTHONPATH=/opt/alphaflow:$PYTHONPATH

%runscript
    cd /opt/alphaflow && python predict.py "$@"
```

### FILE: `containers/mace.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%labels
    org.nanoforge.role mace
    org.nanoforge.license MIT

%post
    pip install --no-cache-dir \
        mace-torch \
        ase \
        e3nn \
        openmm-torch

%environment
    export MACE_MODELS_DIR=/models/mace

%runscript
    exec python -c "from mace.calculators import mace_off; print('MACE OK')"
```

### FILE: `containers/aimnet2.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    pip install --no-cache-dir \
        ase \
        openmm-torch

    cd /opt && \
        git clone --depth 1 https://github.com/zubatyuk/aimnet2-asf.git && \
        cd aimnet2-asf && \
        pip install --no-cache-dir -e .

%environment
    export AIMNET2_MODELS_DIR=/models/aimnet2
    export PYTHONPATH=/opt/aimnet2-asf:$PYTHONPATH
```

### FILE: `containers/xtb_pyscf.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir \
        xtb-python \
        pyscf \
        ase

%runscript
    exec python -c "from xtb.interface import Calculator; print('xtb OK'); import pyscf; print('PySCF OK')"
```

### FILE: `containers/gromacs_pmx.def`

```
Bootstrap: docker
From: gromacs/gromacs:2024

%post
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir \
        pmx \
        alchemlyb \
        biopython

%environment
    export GMXLIB=/usr/local/gromacs/share/gromacs/top

%runscript
    exec gmx "$@"
```

### FILE: `containers/ablang2.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    pip install --no-cache-dir \
        transformers==4.40.0 \
        AbLang2 \
        ANARCI

%environment
    export ABLANG2_MODELS_DIR=/models/ablang2
```

### FILE: `containers/paragraph.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    pip install --no-cache-dir \
        einops>=0.4 \
        prody==2.4 \
        Paragraph
```

### FILE: `containers/biophi.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    pip install --no-cache-dir \
        biophi \
        sapiens
```

### FILE: `containers/immunebuilder.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    pip install --no-cache-dir \
        ImmuneBuilder \
        ANARCI
```

### FILE: `containers/chai1.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    pip install --no-cache-dir chai_lab
```

### FILE: `containers/ani.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    pip install --no-cache-dir \
        torchani \
        ase \
        openmm-torch
```

### FILE: `containers/freebindcraft.def`

```
Bootstrap: localimage
From: containers/base_cuda.sif

%post
    cd /opt && \
        git clone --depth 1 https://github.com/cytokineking/FreeBindCraft.git && \
        cd FreeBindCraft && \
        pip install --no-cache-dir -e .
```

---

## Section D — Database initialization

### FILE: `nanoforge/db/migrations/versions/0001_initial_schema.py`

```python
"""Initial schema (v3)

Revision ID: 0001
Revises:
Create Date: 2026-05-24

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable required extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.create_table(
        "targets",
        sa.Column("target_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("uniprot_id", sa.Text()),
        sa.Column("sequence", sa.Text(), nullable=False),
        sa.Column("structure_path", sa.Text()),
        sa.Column("plddt_mean", sa.Float()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "target_conformers",
        sa.Column("conformer_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("targets.target_id")),
        sa.Column("structure_path", sa.Text(), nullable=False),
        sa.Column("ensemble_method", sa.Text(), nullable=False),
        sa.Column("cluster_representative", sa.Integer()),
        sa.Column("cluster_size", sa.Integer()),
        sa.Column("plddt_mean", sa.Float()),
        sa.Column("epitope_map", postgresql.JSONB()),
    )

    op.create_table(
        "runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("targets.target_id")),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("git_commit", sa.Text()),
        sa.Column("container_digests", postgresql.JSONB()),
    )

    op.create_table(
        "candidates",
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("runs.run_id")),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Text(), nullable=False),
        sa.Column("cdr1", sa.Text()),
        sa.Column("cdr2", sa.Text()),
        sa.Column("cdr3", sa.Text()),
        sa.Column("structure_path", sa.Text()),
        sa.Column("complex_path", sa.Text()),
        sa.Column("scores", postgresql.JSONB(), nullable=False),
        sa.Column("liabilities", postgresql.JSONB()),
        sa.Column("risk_tier", sa.Text()),
        sa.Column("pareto_rank", sa.Integer()),
        sa.Column("final_rank", sa.Integer()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_candidates_run", "candidates", ["run_id"])
    op.create_index("idx_candidates_rank", "candidates", ["run_id", "final_rank"])
    op.create_index("idx_candidates_source", "candidates", ["source"])

    op.create_table(
        "paratope_sanity_results",
        sa.Column("paratope_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("candidates.candidate_id")),
        sa.Column("paragraph_predicted_residues", postgresql.ARRAY(sa.Integer())),
        sa.Column("observed_residues", postgresql.ARRAY(sa.Integer())),
        sa.Column("concordance_jaccard", sa.Float()),
        sa.Column("cdr_contact_fraction", sa.Float()),
        sa.Column("framework_contact_fraction", sa.Float()),
        sa.Column("sanity_pass", sa.Boolean()),
    )

    op.create_table(
        "refinement_results",
        sa.Column("refinement_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("candidates.candidate_id")),
        sa.Column("md_engine", sa.Text()),
        sa.Column("md_ns", sa.Float()),
        sa.Column("nnp_model", sa.Text()),
        sa.Column("interface_rmsd_mean", sa.Float()),
        sa.Column("interface_rmsd_std", sa.Float()),
        sa.Column("cdr3_cluster_occupancy", sa.Float()),
        sa.Column("nnp_de_bind_kcal", sa.Float()),
        sa.Column("mmgbsa_dg_kcal", sa.Float()),
        sa.Column("polarization_sensitive", sa.Boolean()),
        sa.Column("persistent_contacts", postgresql.JSONB()),
        sa.Column("hotspot_residues", postgresql.JSONB()),
        sa.Column("qm_used", sa.Boolean(), server_default="false"),
        sa.Column("qm_method", sa.Text()),
        sa.Column("trajectory_path", sa.Text()),
    )

    op.create_table(
        "matured_variants",
        sa.Column("variant_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("parent_candidate_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("candidates.candidate_id")),
        sa.Column("mutations", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Text(), nullable=False),
        sa.Column("ddg_proteinmpnn", sa.Float()),
        sa.Column("ddg_nnp_mmgbsa", sa.Float()),
        sa.Column("ddg_classical_mmgbsa_crosscheck", sa.Float()),
        sa.Column("ddg_fep", sa.Float()),
        sa.Column("ddg_fep_error", sa.Float()),
        sa.Column("predicted_kd_fold", sa.Float()),
        sa.Column("tier", sa.Text()),
    )
    op.create_index("idx_matured_parent", "matured_variants", ["parent_candidate_id"])

    op.create_table(
        "cross_reactivity_results",
        sa.Column("crossreact_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("candidates.candidate_id")),
        sa.Column("paralog_results", postgresql.JSONB()),
        sa.Column("psr_proxy_score", sa.Float()),
        sa.Column("cross_reactivity_tier", sa.Text()),
    )

    op.create_table(
        "fto_results",
        sa.Column("fto_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("candidates.candidate_id")),
        sa.Column("cdr3_match", postgresql.JSONB()),
        sa.Column("full_seq_match", postgresql.JSONB()),
        sa.Column("risk_tier", sa.Text()),
        sa.Column("external_review_required", sa.Boolean()),
        sa.Column("checked_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "experimental_results",
        sa.Column("result_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("candidates.candidate_id")),
        sa.Column("expression_yield_mg_per_l", sa.Float()),
        sa.Column("binding_observed", sa.Boolean()),
        sa.Column("kd_nm", sa.Float()),
        sa.Column("kon", sa.Float()),
        sa.Column("koff", sa.Float()),
        sa.Column("tm_celsius", sa.Float()),
        sa.Column("notes", sa.Text()),
        sa.Column("submitted_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("submitted_by", sa.Text()),
    )

    op.create_table(
        "license_register",
        sa.Column("dep_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text()),
        sa.Column("license_spdx", sa.Text(), nullable=False),
        sa.Column("commercial_ok", sa.Boolean(), nullable=False),
        sa.Column("verification_method", sa.Text()),
        sa.Column("obligations", sa.Text()),
        sa.Column("last_audited", sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.Column("auditor", sa.Text()),
    )


def downgrade() -> None:
    for tbl in ["license_register", "experimental_results", "fto_results",
                "cross_reactivity_results", "matured_variants",
                "refinement_results", "paratope_sanity_results",
                "candidates", "runs", "target_conformers", "targets"]:
        op.drop_table(tbl)
```

### FILE: `nanoforge/db/alembic.ini`

```ini
[alembic]
script_location = nanoforge/db/migrations
sqlalchemy.url = ${DATABASE_URL}

[loggers]
keys = root, sqlalchemy, alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

---

## Section E — Initial ADRs

### FILE: `adrs/ADR-template.md`

```markdown
# ADR-NNNN: Short title

Status: proposed | accepted | superseded | deprecated
Date: YYYY-MM-DD
Deciders: <names>

## Context
What problem are we solving? What are the constraints?

## Decision
What did we decide?

## Consequences
What are the trade-offs and risks?

## Alternatives considered
What did we reject and why?
```

### FILE: `adrs/ADR-0005-nnp-primary-refinement.md`

```markdown
# ADR-0005: NNP as primary CDR refinement engine

Status: accepted
Date: 2026-05-24
Deciders: K Kang

## Context

Neural Network Potentials (MACE-OFF, AIMNet2, ANI-2x) reached production maturity in 2023-2025. On H200-class hardware they offer QM-level accuracy at 10-100× the speed of DFT and 10-100× the cost of classical force fields — a regime where they are no longer "optional refinement add-on" but the default production engine for interface dynamics.

VHH-antigen interfaces commonly involve aromatic π-stacking (CDR Trp/Tyr/Phe), salt bridges (charged target residues), and polarization effects that classical Amber/CHARMM force fields capture poorly. Published binding-energy errors for classical MM/GBSA on such interfaces are 3-5 kcal/mol; NNP cuts this roughly in half.

## Decision

NNP is the primary production-MD engine in Module 5.1 for all 50 promoted candidates. NNP-derived ΔE_bind is the primary binding energy metric in Module 5.2. Classical MD restricted to equilibration. Classical MM/GBSA retained only as sanity-check cross-reference.

NNP selection auto-determined via decision tree (Module 5.1 §B):
- Default: MACE-OFF
- Charged interface: AIMNet2
- Aromatic-dominated: MACE-OFF
- Non-standard elements: MACE-MP-0
- Speed-critical screening: ANI-2x

Module 6 Tier 2 (mutation scoring) similarly replaced classical MM/GBSA with NNP-MD ΔΔG.

## Consequences

+ More accurate ΔG and ΔΔG estimation, especially for polar/aromatic interfaces
+ Better correlation with experimental KD (gate: Spearman > 0.4 on antibody-antigen test set)
+ Per-residue energy decomposition more reliable for hotspot ID
+ NNP-MD scales to ~10,000 atoms on H200 in reasonable time

− Wallclock increases ~30% compared to classical-only baseline (offset by H200 parallelization)
− NNP element coverage limits some targets (mitigated by MACE-MP-0 fallback)
− Reproducibility requires strict checkpoint pinning
− More complex configuration surface

## Alternatives considered

- Classical MD as primary: rejected — known accuracy limits on polar/aromatic
- Pure QM/QM-MM: rejected — infeasible for 50 candidates × MD
- FEP for all candidates: rejected — 120+ hrs per candidate
- AlphaFold-multimer affinity only (no MD): rejected — single static prediction misses CDR3 dynamics
```

### FILE: `adrs/ADR-0011-use-rfantibody.md`

```markdown
# ADR-0011: Use RFantibody as primary de novo engine

Status: accepted
Date: 2026-05-24
Deciders: K Kang

## Context

Generic RFdiffusion + ProteinMPNN works but produces backbones not specifically tuned for Ig fold. Common failure modes: hallmark residues lost during MPNN sequence design (mitigated but not eliminated by `--fix_positions`), CDR3 length distribution skewed short vs natural VHH, paratope geometry inconsistent with natural binding.

RFantibody (Baker lab, Bennett et al., *Nature* 2025) is a purpose-built three-step pipeline: antibody-finetuned RFdiffusion → antibody-specific ProteinMPNN wrapper → antibody-finetuned RoseTTAFold2 for fold-back validation. Released Nov 2025 under MIT license for academic, personal, and commercial use (training code separately licensed to Xaira; inference + weights MIT).

## Decision

Replace generic RFdiffusion + ProteinMPNN with RFantibody in Module 2b. Keep generic path as `denovo_legacy` for ablation comparison only.

## Consequences

+ Better Ig-fold backbones; hallmark residue preservation native
+ Antibody-finetuned RoseTTAFold2 fold-back filter built in
+ Published cryo-EM-validated successes (influenza HA 78 nM, TcdB scFvs, etc.)
+ Larger user community

− Larger compute per design (~30 s vs ~5 s for generic RFdiffusion)
− Compensated by lower number of designs needing paratope-sanity rejection (Module 4.5)
− RFantibody benefits from ~9,000 designs per target for high-quality hits; budget accordingly
```

### FILE: `adrs/ADR-0014-exclude-ibex-nc-weights.md`

```markdown
# ADR-0014: Exclude Ibex due to non-commercial weight license

Status: accepted
Date: 2026-05-24
Deciders: K Kang

## Context

Ibex (Prescient Design / Genentech, 2025) is a unified Ig/TCR structure predictor with apo/holo paired training — attractive for conformational ensemble support. Codebase is Apache 2.0. **However, model weights are released under "Genentech Apache 2.0 Non-Commercial license"**, meaning the weights cannot be used in commercial work, and Ibex is unusable without its weights.

## Decision

Exclude Ibex from the commercial pipeline. NanoBodyBuilder2 (BSD-3, fully open) remains primary VHH structure predictor for monomer prediction. Conformational ensemble handled by AlphaFlow / ESMFlow (per ADR-0018) and NNP-MD (per ADR-0005).

## Consequences

+ License compliance preserved
− Lose access to apo/holo paired conformational sampling specific to Ig
- Revisit if Genentech relaxes weights license

## Alternatives considered

- Try to obtain commercial license from Genentech: not pursued for initial deployment
- Use Ibex code with own-trained weights: infeasible (training data and procedure not fully reproducible)
- AlphaFlow / ESMFlow with antibody-specific calibration: chosen (see ADR-0018)
```

### FILE: `adrs/ADR-0015-exclude-nanobert-nc.md`

```markdown
# ADR-0015: Exclude nanoBERT (CC BY-NC-SA 4.0)

Status: accepted
Date: 2026-05-24
Deciders: K Kang

## Context

v2.2 patch included nanoBERT (NaturalAntibody, 2024) as primary VHH-nativeness scorer based on apparent open availability. Direct verification of HuggingFace model card (huggingface.co/NaturalAntibody/nanoBERT) reveals license: **CC BY-NC-SA 4.0 (Non-Commercial Share-Alike)**. This prohibits commercial use, including in commercial pipelines and derivative products.

## Decision

Exclude nanoBERT. Move to AbLang2 + OAS camelid fine-tuning (ADR-0019).

## Consequences

+ License compliance preserved
+ Calibration data from nanoBERT paper (18 therapeutic VHHs) still usable for replacement model calibration
− Lose access to a model already calibrated on INDI
− Must train own head (~6-12 hours one-time on H200)

## Alternatives considered

- Email NaturalAntibody for commercial license: possible future option; not pursued for initial deployment
- ESM-2 + OAS fine-tune: viable Path 2 alternative
- Train own RoBERTa from scratch on INDI: most expensive option, kept as Path 3 backup
```

### FILE: `adrs/ADR-0018-add-alphaflow-esmflow.md`

```markdown
# ADR-0018: Add AlphaFlow / ESMFlow for conformational ensemble

Status: accepted
Date: 2026-05-24
Deciders: K Kang

## Context

v2 base provided AF-cluster (MSA subsampling) and v2.1 added NNP-MD for conformational ensemble. v2.2 attempted Ibex for apo/holo conformational sampling but excluded due to NC weights (ADR-0014).

AlphaFlow / ESMFlow (Jing, Berger, Jaakkola, ICML 2024) are MIT licensed for code + weights. These flow-matching extensions of AlphaFold2 / ESMFold generate conformational ensembles from sequence with PDB-like or MD-like sampling distributions.

## Decision

Add AlphaFlow / ESMFlow as third ensemble option (`target.ensemble: alphaflow | esmflow`). Default: **ESMFlow-MD+Templates 12l-distilled** (~15 min for 50-sample ensemble on H200, no MSA needed).

## Consequences

+ Closes conformational dynamics gap from v2 risks
+ 10-20× faster than NNP-MD for comparable use cases
+ No MSA dependency for ESMFlow variants
+ MIT license for all components
− Adds OpenFold dependency (Apache 2.0, OK)
− Distilled models trade some accuracy for speed (full models available)

## Alternatives considered

- Ibex: excluded (ADR-0014, non-commercial weights)
- Only NNP-MD: kept as physics-based alternative; AlphaFlow/ESMFlow chosen as default for speed
- Only AF-cluster: kept as fallback for sequence-only contexts
```

### FILE: `adrs/ADR-0019-ablang2-oas-vhh-nativeness.md`

```markdown
# ADR-0019: AbLang2 + OAS camelid for VHH-nativeness (supersedes ADR-0012)

Status: accepted (supersedes ADR-0012)
Date: 2026-05-24
Deciders: K Kang

## Context

Following exclusion of nanoBERT (ADR-0015) and AbNatiV deferral (ADR-0008), VHH-nativeness scoring needs commercial-clean implementation.

Options considered:
1. AbLang2 (BSD-3, already in v2 spec) fine-tuned on OAS camelid subset
2. ESM-2 (MIT) fine-tuned on OAS camelid subset
3. RoBERTa trained from scratch on INDI

## Decision

Path 1: AbLang2 + OAS camelid fine-tuning.

Rationale: AbLang2 is antibody-specific (better starting point than ESM-2 for VHH context), BSD-3 licensed, and small fine-tuning head trains in ~6-12 hours on H200. Falls back to Path 2 (ESM-2) if calibration insufficient.

## Consequences

+ Fully commercial-clean stack for VHH-nativeness
+ Reuses dependency already in v2 (AbLang2)
+ One-time training cost, no per-target inference penalty

− Lose published calibration nanoBERT had
− Must establish own calibration against therapeutic VHH benchmark in Phase 2
− Risk that fine-tuned model under-performs nanoBERT until calibration validated

## Alternatives considered

- ESM-2 + OAS fine-tune: viable Path 2 backup
- Train RoBERTa from scratch on INDI: most expensive; kept as Path 3 last resort
- License nanoBERT commercially from NaturalAntibody: not pursued for initial deployment
```

---

## Section F — Default config + profiles

### FILE: `configs/default.yaml`

(see NANOBODY_PIPELINE_SPEC_v3.md §8.1 for the complete default config — extract that block verbatim)

### FILE: `configs/profiles/general.yaml`

```yaml
# General profile — soluble globular target, full pipeline
# Inherits all defaults from configs/default.yaml; no overrides
```

### FILE: `configs/profiles/therapeutic_strict.yaml`

```yaml
# Therapeutic-grade strict configuration
# Inherits from configs/default.yaml

fto:
  external_legal_review_warning: true
  pre_check_blocking: true

cross_reactivity:
  paralog_identity_threshold: 0.25
  boltz_spot_check: true

humanness:
  oasis_threshold: 0.75
  vhh_nativeness_threshold: 0.65

refinement:
  qm_enabled: true
  refinement_tier_ns: 100

maturation:
  tier3_fep: true

ranking:
  diversity_clustering_threshold: 0.6
```

### FILE: `configs/profiles/aav_capsid.yaml`

```yaml
# AAV capsid engineering — BBB receptor binders, CNS tropism
# Inherits from configs/default.yaml

target:
  ensemble: nnp_md
  glycan_modeling: full

library:
  cdr3_length_range: [12, 20]

cross_reactivity:
  custom_paralogs: [TfR1, LRP1, LRP8, BCAM]

brief:
  custom_assays:
    - "in vitro BBB transcytosis (hCMEC/D3 monolayer)"
    - "in vivo BBB transit (mouse IV injection + brain ELISA)"
    - "AAV capsid display IF/ELISA validation"
```

### FILE: `configs/profiles/soluble_research.yaml`

```yaml
# Research-tool VHH; no humanization, prioritize affinity
# Inherits from configs/default.yaml

humanization:
  enabled: false

humanness:
  vhh_nativeness_threshold: 0.3  # relaxed

ranking:
  top_n_final: 30  # more candidates for screening
```

### FILE: `configs/profiles/gpcr_allosteric.yaml`

```yaml
# GPCR allosteric site targeting
# Inherits from configs/default.yaml

target:
  ensemble: alphaflow                # better for crystallographically-resolved multi-state
  flow_model: alphaflow_12l_md_templates_distilled
  glycan_modeling: full              # GPCRs often glycosylated

refinement:
  refinement_tier_ns: 100            # GPCR conformational dynamics need more sampling
```

### FILE: `configs/profiles/ion_channel.yaml`

```yaml
# Ion channel selectivity filter / pore targeting
# Inherits from configs/default.yaml

target:
  ensemble: nnp_md                   # open/closed state ensemble
  nnp_md_ns: 100                     # longer MD for slow gating

refinement:
  refinement_tier_ns: 100
```

---

## Section G — README skeleton

### FILE: `README.md`

```markdown
# NanoForge

Automated nanobody (VHH) design and refinement pipeline.

**Status**: Phase 0 (bootstrapping).

## Overview

NanoForge takes a target protein sequence and outputs ranked, refined, and FTO-checked VHH candidates with full experimental briefs. Built on a fully license-cleared commercial-OK stack.

See [NANOBODY_PIPELINE_SPEC_v3.md](./NANOBODY_PIPELINE_SPEC_v3.md) for the complete specification.

## Quick start

```bash
# Phase 0 setup (one-time)
bash scripts/install_models.sh
bash scripts/build_containers.sh
bash scripts/init_db.sh
bash scripts/build_vhh_library.sh

# Run on a target (after Phase 1 implementation)
nanoforge run --uniprot P04626 --profile general
```

## License

This repository's wrapper code is MIT licensed. Dependencies use various open-source licenses; see [LICENSES/REGISTER.md](./LICENSES/REGISTER.md) for the complete register.

## Documentation

- [Specification (v3)](./NANOBODY_PIPELINE_SPEC_v3.md) — authoritative pipeline design
- [ADRs](./adrs/) — architecture decision records
- [License register](./LICENSES/REGISTER.md) — full dep license audit

## Known limitations

See §13 of the spec for the full list. Highlights:

- Affinity scores are relative ranking only (not absolute KD)
- De novo wet-lab success rate ~1-2% (RFantibody) to ~5-15% (other methods)
- MHC-II / T-cell epitope prediction is approximate (no NetMHCIIpan)
- FTO module is screening only — NOT legal advice

## Contributing

This is a research pipeline. Before commercial deployment:
- Verify all dependency licenses haven't changed (`scripts/license_audit.py`)
- Obtain external IP attorney review for FTO conclusions
- Pin all container digests by SHA256
```

### FILE: `LICENSES/REGISTER.md`

```markdown
# License Register

This document lists every external dependency, its license, and verification status.

**Audit principle**: every entry verified by direct inspection of repository LICENSE file
or model card. Re-audit quarterly.

**Last audited**: 2026-05-24

## Commercial-OK dependencies

| Tool | License | Verified | Verification method | Use site |
|---|---|---|---|---|
| Boltz-2 | MIT | ✅ | repo LICENSE | Module 2a, 3 |
| Chai-1 | Apache 2.0 | ✅ | repo LICENSE | Module 3 |
| ESM-2 / ESMFold | MIT | ✅ | repo LICENSE | Module 1 |
| RFantibody | MIT | ✅ | repo LICENSE + Baker Lab announcement Nov 2025 | Module 2b |
| ProteinMPNN | MIT | ✅ | repo LICENSE | Module 2b, 6 |
| LigandMPNN | MIT | ✅ | repo LICENSE | Module 2b |
| FreeBindCraft | MIT | ✅ | repo LICENSE | Module 2b (ablation) |
| NanoBodyBuilder2 / ImmuneBuilder | BSD-3 | ✅ | repo LICENSE | Module 4 |
| ANARCI | BSD-3 | ✅ | repo LICENSE | utility |
| AbLang2 | BSD-3 | ✅ | repo LICENSE | Module 4 (VHH-nativeness head) |
| Paragraph | BSD-3 | ✅ | setup.py | Module 4.5 |
| BioPhi / Sapiens | MIT | ✅ | repo LICENSE | Module 4 (humanization) |
| AlphaFlow / ESMFlow | MIT | ✅ | repo LICENSE + README | Module 1 (ensemble) |
| OpenFold | Apache 2.0 | ✅ | repo LICENSE | dependency |
| MACE-OFF / MACE-MP-0 | MIT | ✅ | repo LICENSE | Module 5 (NNP) |
| TorchANI / ANI-2x | MIT | ✅ | repo LICENSE | Module 5 (NNP) |
| AIMNet2 | MIT | ✅ | repo LICENSE | Module 5 (NNP) |
| OpenMM | MIT | ✅ | repo LICENSE | Module 5 (MD) |
| GROMACS | LGPL-2.1 | ✅ dynamic link | repo LICENSE | Module 6 (FEP) |
| pmx | LGPL | ✅ | repo LICENSE | Module 6 (FEP) |
| alchemlyb | BSD-3 | ✅ | repo LICENSE | Module 6 (FEP analysis) |
| xtb / GFN-xTB | LGPL | ✅ | repo LICENSE | Module 5 (QM) |
| PySCF | Apache 2.0 | ✅ | repo LICENSE | Module 5 (QM) |
| ASE | LGPL | ✅ | repo LICENSE | NNP driver |
| MHCflurry | Apache 2.0 | ✅ | repo LICENSE | Module 4 (MHC-I only) |
| MMseqs2 | MIT | ✅ | repo LICENSE | utility |
| P2Rank | Apache 2.0 | ✅ | repo LICENSE | Module 1 |
| fpocket | MIT | ✅ | repo LICENSE | Module 1 |
| Nextflow | Apache 2.0 | ✅ | repo LICENSE | workflow |
| Apptainer | BSD-3 | ✅ | repo LICENSE | containers |
| PostgreSQL | PostgreSQL License | ✅ | repo LICENSE | DB |
| FastAPI | MIT | ✅ | repo LICENSE | API |
| Streamlit | Apache 2.0 | ✅ | repo LICENSE | UI |
| pydantic v2 | MIT | ✅ | repo LICENSE | schemas |
| SQLAlchemy 2.x | MIT | ✅ | repo LICENSE | DB ORM |
| Alembic | MIT | ✅ | repo LICENSE | DB migrations |
| FreeSASA | LGPL | ✅ link only | repo LICENSE | surface area |
| propka | LGPL | ✅ | repo LICENSE | pKa |

## Data sources

| Source | License | Verified | Use site |
|---|---|---|---|
| OAS | CC BY 4.0 | ✅ | training (VHH-nativeness) |
| INDI | CC BY 4.0 | ✅ | training (alternative) |
| SAbDab | OPIG free download | ✅ | runtime fetch only; not bundled |
| PDB | public domain | ✅ | runtime fetch |
| UniProt | CC BY 4.0 | ✅ | target prep |
| IEDB | CC BY 4.0 | ✅ | MHC-II proxy |
| USPTO bulk patents | public domain | ✅ | FTO module |
| EPO OPS | free (with key) | ✅ | FTO module |
| NbThermo | published open data | ✅ | thermostability head |
| AVIDa-hIL6 | CC BY 4.0 | ✅ at adopt | auxiliary affinity training |
| NbBench | open benchmark | ✅ at adopt | CI validation |

## Explicitly excluded

| Tool | Reason | ADR |
|---|---|---|
| AlphaFold3 weights | CC BY-NC-SA 4.0 | ADR-0001 |
| AlphaProteo | not released | n/a |
| PyRosetta / Rosetta | commercial license required | ADR-0002 |
| NetMHCIIpan, NetMHCpan | academic only | ADR-0003 |
| FoldX | commercial license required | n/a |
| HADDOCK web | academic only | n/a |
| Schrödinger Suite | commercial license required | n/a |
| Gaussian | commercial license required | n/a |
| ORCA | commercial license required | n/a |
| AMBER pmemd.cuda | restricted | n/a |
| MODELLER | academic only | ADR-0016 (blocks Llamanade) |
| AbNatiV | license unclear | ADR-0008 |
| Ibex weights | Genentech NC | ADR-0014 |
| NbForge | license unclear | n/a |
| nanoBERT | CC BY-NC-SA 4.0 | ADR-0015 |
| Llamanade | MODELLER dep | ADR-0016 |
| NABP-BERT | no LICENSE file | ADR-0017 |
| TANGO | academic only | n/a |

## Audit procedure

1. **At adoption**: inspect LICENSE file in repo + model card; record SHA of LICENSE.
2. **In CI**: `pip-licenses` + `scancode-toolkit` on every PR.
3. **Quarterly**: re-inspect for each entry; update `last_audited` field.
4. **Before commercial deployment**: external legal counsel review.
```

---

## Section H — Quick-start sequence for Codex CLI

For Phase 0 bootstrap, execute in this order:

```bash
# 1. Extract all files from this bootstrap document into project tree
codex "Extract every '### FILE: <path>' block from NANOBODY_PIPELINE_v3_BOOTSTRAP.md
into the corresponding paths. Create directories as needed."

# 2. Initialize git + pre-commit
cd nanoforge
git init
mv pre-commit-config.yaml .pre-commit-config.yaml
git add . && git commit -m "Initial Phase 0 scaffold"
pre-commit install

# 3. Build base container
bash scripts/build_containers.sh  # builds base_cuda.sif at minimum

# 4. Download model weights (~80 GB, 1-3 hrs)
bash scripts/install_models.sh

# 5. Initialize database
bash scripts/init_db.sh

# 6. Verify license audit passes
python scripts/license_audit.py

# 7. Git tag
git tag phase-0-foundation-complete
```

After Phase 0 passes, proceed to Phase 1 per spec §7.

---

## Section I — Validation that bootstrap is complete

Phase 0 is "complete" when ALL of:

- [ ] Project skeleton matches spec §4 directory layout
- [ ] `pyproject.toml` installable: `pip install -e .[dev]` succeeds
- [ ] `pre-commit install` works and runs on first commit
- [ ] `base_cuda.sif` Apptainer container builds successfully
- [ ] `python scripts/license_audit.py` returns exit code 0
- [ ] PostgreSQL schema applied: `alembic upgrade head` succeeds
- [ ] `LICENSES/REGISTER.md` exists with all entries from this document
- [ ] At least ADR-0001, 0005, 0011, 0014, 0015, 0018, 0019 present in `adrs/`
- [ ] Git tag `phase-0-foundation-complete` applied

---

End of Phase 0 bootstrap package.
