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
        printf '%s -m research.floodwarp_methods_paper.fidelity.%s %s\n' "$python_bin" "$module" "${extra[*]}"
    else
        "$python_bin" -m "research.floodwarp_methods_paper.fidelity.$module" "${extra[@]}"
    fi
}
main() {
    local stage=${1:-help} dry=false module
    if [[ $stage == help || $stage == --help ]]; then
        echo "Usage: $0 {all|status|summary|verify-kernels|pulses|pulse-recession|sharp-pulses|sharp-reference|pulse-reference|pulse-figures|operational-preview|operational-pairs|native-horizons|storage-tail|storage-tail-report|native-review|endpoint-fields|orientation|storm-controls|rolling-reference|incised-reference|incised-storage|area-mean-storms|area-mean-terrain|incised-plane|incised-reach|incised-checkpoints|recession|policy-recession|bed-recession|small-review|analyse|figures|verify} [--dry-run]"
        return
    fi
    shift
    for option in "$@"; do
        case "$option" in --dry-run) dry=true ;; --resume) ;; *) echo "Unsupported option: $option" >&2; return 2 ;; esac
    done
    if [[ $stage == all ]]; then
        for item in verify-kernels pulses pulse-recession sharp-pulses pulse-reference operational-preview native-review orientation storm-controls storage-tail storage-tail-report operational-pairs native-horizons rolling-reference area-mean-terrain area-mean-storms incised-plane incised-reach recession small-review policy-recession bed-recession endpoint-fields analyse verify; do run_stage "$item"; done
    else
        run_stage "$stage"
    fi
}
main "$@"
