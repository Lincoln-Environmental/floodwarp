#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python_bin=${FLOODWARP_PAPER_PYTHON:-"$root/.venv/bin/python"}
export MPLBACKEND=${MPLBACKEND:-Agg}
run_stage() {
    local -a extra=()
    case "$1" in
        status) module=status ;;
        advisor-design) module=advisor_design ;;
        advisor-cli-check) module=advisor_cli_check ;;
        outlet-results) module=outlet_results ;;
        native-status) module=native_status ;;
        strong-storm-controls) module=strong_storms ;;
        summary) module=progress ;;
        verify-kernels) module=verification ;;
        verify) module=verify_products ;;
        pulses) module=pulse ;;
        pulse-recession) module=pulse_recession ;;
        sharp-pulses|sharp-reference) module=sharp_pulse; extra=("$1") ;;
        pulse-reference) module=pulse_reference ;;
        pulse-figures) module=pulse_products ;;
        orientation) module=oblique ;;
        bed-recession) module=bed_recession ;;
        policy-recession) module=policy_recession ;;
        endpoint-fields) module=endpoints ;;
        storage-tail-report) module=tail_report ;;
        native-horizons) module=native_horizons ;;
        incised-reference) module=incised_crest ;;
        area-mean-storms) module=mean_storms ;;
        area-mean-terrain) module=mean_windows ;;
        native-input-audit) module=native_audit ;;
        native-restart-control) module=native_cpu_control ;;
        free-plane) module=free_plane ;;
        free-reach) module=free_reach ;;
        native-outlet) module=native_outlet ;;
        native-states) module=restore_native ;;
        native-features) module=native_features ;;
        native-advisor-train|native-advisor-validate) module=native_advisor; extra=("$1") ;;
        native-publication) module=native_publication ;;
        native-infiltration) module=native_infiltration ;;
        crest-sources) module=crest_sources ;;
        native-tail|native-peak-replay) module=native_tail; extra=("$1") ;;
        outlet-report) module=outlet_report ;;
        native-storms) module=native_matrix ;;
        native-terrain) module=native_matrix; extra=(--terrain-only) ;;
        native-products|verify-native) module=native_products; extra=("$1") ;;
        incised-plane) module=incised_plane ;;
        incised-reach) module=incised_reach ;;
        incised-checkpoints) module=incised_checkpoint_audit ;;
        incised-storage) module=crest_report; extra=("$1") ;;
        native-review) module=operational_review ;;
        small-review) module=small_review ;;
        rolling-reference|storm-controls|recession|operational-preview|operational-pairs|storage-tail) module=continuation; extra=("$1") ;;
        analyse) module=publication ;;
        figures) module=render ;;
        *) echo "Unknown fidelity stage: $1" >&2; return 2 ;;
    esac
    if [[ $dry == true ]]; then
        printf '%s -m research.floodwarp_methods_paper.fidelity.%s' "$python_bin" "$module"
        if ((${#extra[@]})); then printf ' %s' "${extra[@]}"; fi
        printf '\n'
    else
        "$python_bin" -m "research.floodwarp_methods_paper.fidelity.$module" "${extra[@]}"
    fi
}
main() {
    local stage=${1:-help} dry=false module
    if [[ $stage == verify-kernels ]]; then
        shift
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.verification "$@"
    fi
    if [[ $stage == outlet-timing-repeats ]]; then
        shift
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.outlet_timing "$@"
    fi
    if [[ $stage == native-timing-repeats ]]; then
        shift
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.timing_repeats "$@"
    fi
    if [[ $stage == orton-transfer ]]; then
        shift
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.orton_transfer "$@"
    fi
    if [[ $stage == outlet-storms ]]; then
        shift
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.outlet_storms "$@"
    fi
    if [[ $stage == native-storms ]]; then
        shift
        local -a inventory_options=()
        for option in "$@"; do
            case "$option" in
                --dry-run|--terrain-only|--resume|--force) ;;
                *) inventory_options+=("$option") ;;
            esac
        done
        for option in "$@"; do
            if [[ $option == --dry-run ]]; then
                "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_matrix "$@"
                "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_status --inventory "${inventory_options[@]}"
                return
            fi
        done
        exec "$python_bin" -m research.floodwarp_methods_paper.fidelity.native_matrix "$@"
    fi
    if [[ $stage == help || $stage == --help ]]; then
        echo "Usage: $0 {all|status|summary|verify-kernels|pulses|pulse-recession|sharp-pulses|sharp-reference|pulse-reference|pulse-figures|operational-preview|operational-pairs|native-horizons|storage-tail|storage-tail-report|native-review|endpoint-fields|orientation|storm-controls|rolling-reference|incised-reference|incised-storage|area-mean-storms|area-mean-terrain|native-input-audit|native-restart-control|free-plane|free-reach|native-outlet|native-states|native-features|native-publication|native-advisor-train|native-advisor-validate|native-tail|native-peak-replay|native-infiltration|crest-sources|outlet-report|native-storms|native-products|verify-native|historical-all|incised-plane|incised-reach|incised-checkpoints|recession|policy-recession|bed-recession|small-review|analyse|figures|verify} [--dry-run]"
        return
    fi
    shift
    for option in "$@"; do
        case "$option" in --dry-run) dry=true ;; --resume) ;; *) echo "Unsupported option: $option" >&2; return 2 ;; esac
    done
    if [[ $stage == all ]]; then
        for item in verify-kernels pulses pulse-recession sharp-pulses pulse-reference pulse-figures orientation area-mean-terrain area-mean-storms native-states native-input-audit native-restart-control free-plane free-reach native-terrain strong-storm-controls native-storms native-features native-products native-outlet native-tail native-peak-replay crest-sources native-infiltration outlet-report verify-native native-advisor-train native-advisor-validate native-publication; do run_stage "$item"; done
    elif [[ $stage == historical-all ]]; then
        for item in verify-kernels pulses pulse-recession sharp-pulses pulse-reference operational-preview native-review orientation storm-controls storage-tail storage-tail-report operational-pairs native-horizons rolling-reference area-mean-terrain area-mean-storms incised-plane incised-reach recession small-review policy-recession bed-recession endpoint-fields analyse verify; do run_stage "$item"; done
    else
        run_stage "$stage"
    fi
}
main "$@"
