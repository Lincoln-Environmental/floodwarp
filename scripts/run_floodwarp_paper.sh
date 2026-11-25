#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: ./scripts/run_floodwarp_paper.sh STAGE [--case GLOB] [--method NAME] [--dx METRES] [--dem METRES] [--resume | --force] [--dry-run]
Stages: check analytical convergence fixed-dt-space synthetic long-storms resolution terrain-initial-controls calibration calibration-fidelity performance
        million-terrain million-pilots million-storms million-auto large-references large-refinements heldout-references million-profiles million-repeats recession-controls execution-policy pareto
        terrain-features advisor-train advisor-validate tables figures manuscript verify hydrograph-fidelity fidelity-figures area-mean-terrain area-mean-storms native-storms native-status native-products gpu-timing-samples native-step-history native-timing-repeats native-publication summary strong-storm-controls fv1-temporal-control hancock-temporal-control native-grid-maps native-reach-balances embankment-transport free-plane free-reach outlet-storms outlet-results outlet-timing-repeats advisor-design advisor-cli-check orton-transfer outlet-qualification verify-kernels verify-native all historical-all
Environment: FLOODWARP_PAPER_PYTHON, FLOODWARP_PAPER_CONFIG, FLOODWARP_PAPER_DEVICE.
The native-input-audit stage reconstructs and verifies the incised storm rasters on a clean checkout.
The outlet-states stage verifies archived field hashes and restores missing workstation checkpoints.
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
    local stage=$1 python_bin root dry=false force=false resume=false flag
    local -a selected_options=()
    shift
    for flag in "$@"; do
        case "$flag" in
            --resume) resume=true; continue ;;
            --force) force=true ;;
            --dry-run) dry=true ;;
        esac
        selected_options+=("$flag")
    done
    if [[ $resume == true && $force == true ]]; then
        echo 'Choose either --resume or --force.' >&2
        return 2
    fi
    export MPLBACKEND="${MPLBACKEND:-Agg}"
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
    if [[ $stage == figures || $stage == tables || $stage == manuscript ]]; then
        if [[ -f research/floodwarp_methods_paper/fidelity/native_results/measured_windows.csv ]]; then
            if [[ $dry == true ]]; then
                ./scripts/run_hydrograph_fidelity.sh native-publication --dry-run
            else
                ./scripts/run_hydrograph_fidelity.sh native-publication
            fi
        fi
    fi
    if [[ $stage == terrain-features || $stage == advisor-train || $stage == advisor-validate ]]; then
        local native_stage="native-$stage"
        if [[ $stage == terrain-features ]]; then native_stage="native-features"; fi
        exec ./scripts/run_hydrograph_fidelity.sh "$native_stage" "${selected_options[@]}"
    fi
    if [[ $stage == million-terrain ]]; then
        exec ./scripts/run_hydrograph_fidelity.sh native-storms --terrain-only "${selected_options[@]}"
    fi
    if [[ $stage == resolution ]]; then
        exec ./scripts/run_hydrograph_fidelity.sh native-storms --suite fixed_physical_domain "${selected_options[@]}"
    fi
    if [[ $stage == million-storms ]]; then
        exec ./scripts/run_hydrograph_fidelity.sh native-storms --suite operational "${selected_options[@]}"
    fi
    if [[ $stage == large-references || $stage == large-refinements ]]; then
        exec ./scripts/run_hydrograph_fidelity.sh native-storms --suite references "${selected_options[@]}"
    fi
    if [[ $stage == heldout-references ]]; then
        exec ./scripts/run_hydrograph_fidelity.sh native-storms --suite heldout_references "${selected_options[@]}"
    fi
    if [[ $stage == historical-terrain-features || $stage == historical-advisor-train || $stage == historical-advisor-validate || $stage == historical-million-terrain || $stage == historical-resolution || $stage == historical-million-storms || $stage == historical-large-references || $stage == historical-large-refinements || $stage == historical-heldout-references ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.controller "${stage#historical-}" "${selected_options[@]}"
    fi
    if [[ $stage == native-input-audit ]]; then
        exec ./scripts/run_hydrograph_fidelity.sh "$stage" "${selected_options[@]}"
    fi
    if [[ $stage == outlet-states ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.outlet_states "${selected_options[@]}"
    fi
    if [[ $stage == native-status && ${#selected_options[@]} == 1 && ${selected_options[0]} == --inventory ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_status --inventory
    fi
    if [[ $stage == calibration-fidelity ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.calibration_fidelity "${selected_options[@]}"
    fi
    if [[ $stage == gpu-timing-samples ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.gpu_sharing "${selected_options[@]}"
    fi
    if [[ $stage == embankment-transport ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.embankment_transport "${selected_options[@]}"
    fi
    if [[ $stage == native-grid-maps || $stage == native-reach-balances ]]; then
        for flag in "${!selected_options[@]}"; do
            if [[ ${selected_options[flag]} == --force ]]; then selected_options[flag]=--recompute; fi
        done
    fi
    if [[ $stage == native-reach-balances ]]; then
        local reach_driver=research.floodwarp_methods_paper.fidelity.reach_balances spacing
        for flag in "${selected_options[@]}"; do
            if [[ $flag == --dx || $flag == --dx=* ]]; then
                exec "$python_bin" -m "$reach_driver" "${selected_options[@]}"
            fi
        done
        for spacing in 1 5 10; do
            "$python_bin" -m "$reach_driver" --dx "$spacing" "${selected_options[@]}"
        done
        return
    fi
    if [[ $stage == native-grid-maps ]]; then
        for flag in "${selected_options[@]}"; do
            if [[ $flag == --partition || $flag == --partition=* ]]; then
                exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_map_suite "${selected_options[@]}"
            fi
        done
        for flag in "${selected_options[@]}"; do
            if [[ $flag == --case || $flag == --case=* ]]; then
                exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_map_comparison "${selected_options[@]}"
            fi
        done
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_map_suite "${selected_options[@]}"
    fi
    if [[ $stage == hancock-temporal-control ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.fv1_clock --method hancock_dense "${selected_options[@]}"
    fi
    if [[ $stage == fv1-temporal-control ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.fv1_clock "${selected_options[@]}"
    fi
    if [[ $stage == outlet-qualification || $stage == all ]]; then
        for stage in free-plane native-input-audit free-reach outlet-states outlet-storms outlet-states outlet-results outlet-timing-repeats outlet-results verify-kernels analytical convergence million-terrain terrain-features advisor-design advisor-cli-check fv1-temporal-control hancock-temporal-control; do
            "$0" "$stage" "${selected_options[@]}"
        done
        for part in fit test; do
            "$0" native-storms --partition "$part" "${selected_options[@]}"
            "$0" native-products "${selected_options[@]}"
            "$0" native-grid-maps --partition "$part" "${selected_options[@]}"
            if [[ $part == fit ]]; then
                "$0" native-reach-balances "${selected_options[@]}"
                "$0" embankment-transport "${selected_options[@]}"
                "$0" native-step-history "${selected_options[@]}"
            fi
            "$0" native-timing-repeats "${selected_options[@]}"
            "$0" native-products "${selected_options[@]}"
            if [[ $part == fit ]]; then
                ./scripts/run_hydrograph_fidelity.sh native-advisor-train "${selected_options[@]}"
                "$0" advisor-cli-check "${selected_options[@]}"
            else
                ./scripts/run_hydrograph_fidelity.sh native-advisor-validate "${selected_options[@]}"
            fi
        done
        for stage in verify-native advisor-design orton-transfer verify-kernels calibration-fidelity native-publication summary tables figures manuscript verify; do
            "$0" "$stage" "${selected_options[@]}"
        done
        return
    fi
    if [[ $stage == native-step-history ]]; then
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_step_history "${selected_options[@]}"
    fi
    if [[ $stage == summary || $stage == outlet-timing-repeats || $stage == verify-kernels || $stage == native-timing-repeats || $stage == orton-transfer || $stage == outlet-results || $stage == advisor-cli-check || $stage == advisor-design || $stage == free-plane || $stage == free-reach || $stage == outlet-storms || $stage == native-products || $stage == native-publication || $stage == native-status || $stage == strong-storm-controls || $stage == verify-native || $stage == native-storms || $stage == area-mean-storms || $stage == area-mean-terrain ]]; then
        exec ./scripts/run_hydrograph_fidelity.sh "$stage" "${selected_options[@]}"
    fi
    if [[ $stage == hydrograph-fidelity || $stage == fidelity-figures ]]; then
        local substage=all
        if [[ $stage == fidelity-figures ]]; then substage=figures; fi
        exec ./scripts/run_hydrograph_fidelity.sh "$substage" "${selected_options[@]}"
    fi
    if [[ $stage == historical-all && ${#selected_options[@]} -eq 0 ]]; then
        "$python_bin" -m research.floodwarp_methods_paper.controller all
        exec ./scripts/run_hydrograph_fidelity.sh historical-all
    fi
    if [[ $stage == historical-all && ${#selected_options[@]} -eq 1 && ${selected_options[0]} == --dry-run ]]; then
        "$python_bin" -m research.floodwarp_methods_paper.controller all --dry-run
        exec ./scripts/run_hydrograph_fidelity.sh historical-all --dry-run
    fi
    exec "$python_bin" -m research.floodwarp_methods_paper.controller "$stage" "${selected_options[@]}"
}
main "$@"
