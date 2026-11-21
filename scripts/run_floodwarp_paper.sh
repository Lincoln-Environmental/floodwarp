#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: ./scripts/run_floodwarp_paper.sh STAGE [--case GLOB] [--method NAME] [--dx METRES] [--dem METRES] [--force] [--dry-run]
Stages: check analytical convergence fixed-dt-space synthetic long-storms resolution terrain-initial-controls calibration performance
        million-terrain million-pilots million-storms million-auto large-references large-refinements heldout-references million-profiles execution-policy pareto
        terrain-features advisor-train advisor-validate tables figures manuscript verify all
Environment: FLOODWARP_PAPER_PYTHON, FLOODWARP_PAPER_CONFIG, FLOODWARP_PAPER_DEVICE.
Running check creates a Python 3.14 environment if needed and verifies the pinned packages.
HELP
}

bootstrap() {
    local interpreter=$1 constraints=$2
    if [[ ! -x "$interpreter" ]]; then
        if [[ -n "${FLOODWARP_PAPER_PYTHON:-}" ]]; then
            echo "The explicitly selected Python executable does not exist: $interpreter" >&2
            return 1
        fi
        if command -v uv >/dev/null; then
            uv venv --python 3.14.7 .venv
        else
            python3.14 -m venv .venv
        fi
    fi
    "$interpreter" -c 'import sys; sys.exit(sys.version_info[:2] != (3,14))' || {
        echo 'The paper numerical environment requires Python 3.14' >&2
        return 1
    }
    if "$interpreter" -m research.floodwarp_methods_paper.environment; then return; fi
    if command -v uv >/dev/null; then
        uv pip install --python "$interpreter" -c "$constraints" -e '.[raster,plot,test]'
    else
        "$interpreter" -m pip install -c "$constraints" -e '.[raster,plot,test]'
    fi
    "$interpreter" -m research.floodwarp_methods_paper.environment
}

main() {
    if [[ $# == 0 || $1 == --help || $1 == -h ]]; then usage; return; fi
    local stage=$1 python_bin root dry=false flag
    shift
    for flag in "$@"; do if [[ $flag == --dry-run ]]; then dry=true; fi; done
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    cd "$root"
    python_bin=${FLOODWARP_PAPER_PYTHON:-"$root/.venv/bin/python"}
    if [[ $dry == true && ! -x $python_bin ]]; then
        echo 'Plan only. A numerical run would first establish the pinned Python 3.14 environment.'
        python_bin=$(command -v python3)
    elif [[ $dry == false && ( $stage == check || $stage == all ) ]]; then
        bootstrap "$python_bin" research/floodwarp_methods_paper/supplementary/constraints.txt
    elif [[ ! -x $python_bin ]]; then
        echo 'Run check to establish the environment' >&2
        return 1
    fi
    exec "$python_bin" -m research.floodwarp_methods_paper.controller "$stage" "$@"
}
main "$@"
