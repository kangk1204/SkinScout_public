#!/usr/bin/env bash
# SkinScout adapter for the upstream DiffDock checkout.
#
# scripts/stage3_diffdock_blind.py calls:
#   diffdock --protein_path P --ligand L --out_dir D \
#            --samples_per_complex N --inference_steps M
# and then reads D/rank1_confidence*.sdf.
#
# Upstream inference.py differs in two ways this script absorbs:
#   * the ligand flag is --ligand_description, not --ligand
#   * results land in D/<complex_name>/, one level below where the caller looks
set -euo pipefail

DIFFDOCK_ROOT="${DIFFDOCK_ROOT:-$HOME/.local/opt/DiffDock}"
DIFFDOCK_PYTHON="${DIFFDOCK_PYTHON:-$HOME/.local/share/mamba/envs/skinscout-diffdock/bin/python}"

protein=""; ligand=""; out_dir=""; passthrough=()
while (($#)); do
    case "$1" in
        --protein_path) protein="$2"; shift 2 ;;
        --ligand|--ligand_description) ligand="$2"; shift 2 ;;
        --out_dir) out_dir="$2"; shift 2 ;;
        *) passthrough+=("$1"); shift ;;
    esac
done

if [[ -z "$protein" || -z "$ligand" || -z "$out_dir" ]]; then
    printf 'diffdock adapter: --protein_path, --ligand and --out_dir are required\n' >&2
    exit 2
fi
if [[ ! -x "$DIFFDOCK_PYTHON" ]]; then
    printf 'diffdock adapter: interpreter not found: %s\n' "$DIFFDOCK_PYTHON" >&2
    exit 2
fi

# Upstream inference runs from its own checkout, so every caller-supplied path
# has to be absolute before the directory change or prody rejects it.
mkdir -p "$out_dir"
protein="$(readlink -f "$protein")"
out_dir="$(readlink -f "$out_dir")"
if [[ -e "$ligand" ]]; then
    ligand="$(readlink -f "$ligand")"
fi
complex_name="skinscout"

( cd "$DIFFDOCK_ROOT" && "$DIFFDOCK_PYTHON" inference.py \
    --protein_path "$protein" \
    --ligand_description "$ligand" \
    --complex_name "$complex_name" \
    --out_dir "$out_dir" \
    "${passthrough[@]}" )

# Surface the top pose where the caller looks for it. The confidence is encoded
# in the filename and is what gets parsed, so the name is preserved verbatim.
shopt -s nullglob
found=0
for pose in "$out_dir/$complex_name"/rank1_confidence*.sdf; do
    cp -f "$pose" "$out_dir/$(basename "$pose")"
    found=1
done
if ((found == 0)); then
    printf 'diffdock adapter: no rank1_confidence*.sdf under %s\n' "$out_dir/$complex_name" >&2
    exit 3
fi
