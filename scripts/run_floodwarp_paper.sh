#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: ./scripts/run_floodwarp_paper.sh STAGE [--case GLOB] [--method NAME] [--force] [--dry-run]
Stages: check analytical convergence synthetic long-storms resolution calibration performance
        terrain-features advisor-train advisor-validate tables figures manuscript verify all
Environment: FLOODWARP_PAPER_PYTHON, FLOODWARP_PAPER_CONFIG, FLOODWARP_PAPER_DEVICE.
Running check creates .venv with Python 3.14 and installs dependencies if the environment is absent.
HELP
}

main() {
    if [[ $# == 0 || $1 == --help || $1 == -h ]]; then usage; return; fi
    local stage=$1 python_bin root dry=false
    shift
    for flag in "$@"; do if [[ $flag == --dry-run ]]; then dry=true; fi; done
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    cd "$root"
    python_bin=${FLOODWARP_PAPER_PYTHON:-"$root/.venv/bin/python"}
    if [[ ! -x $python_bin ]]; then
        if [[ $dry == true ]]; then printf '%s\n' 'Would bootstrap Python 3.14 and run the selected stage'; return; fi
        if [[ $stage != check && $stage != all ]]; then echo 'Run check to establish the environment' >&2; return 1; fi
        if command -v uv >/dev/null; then
            uv venv --python 3.14 .venv
            uv pip install --python "$python_bin" -e '.[raster,plot,test]'
        else
            python3.14 -m venv .venv
            "$python_bin" -m pip install -e '.[raster,plot,test]'
        fi
    fi
    "$python_bin" -m research.floodwarp_methods_paper.runner "$stage" "$@"
}
main "$@"
