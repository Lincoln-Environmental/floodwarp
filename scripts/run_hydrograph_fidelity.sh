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
        verify-kernels) module=verification ;;
        verify) module=verify_products ;;
        pulses) module=pulse ;;
        pulse-recession) module=pulse_recession ;;
        pulse-figures) module=pulse_products ;;
        orientation) module=oblique ;;
        bed-recession) module=bed_recession ;;
        policy-recession) module=policy_recession ;;
        endpoint-fields) module=endpoints ;;
        native-review) module=operational_review ;;
        small-review) module=small_review ;;
        rolling-reference|storm-controls|recession|operational-preview|operational-pairs) module=continuation; extra=("$1") ;;
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
        echo "Usage: $0 {all|status|verify-kernels|pulses|pulse-recession|pulse-figures|operational-preview|operational-pairs|native-review|endpoint-fields|orientation|storm-controls|rolling-reference|recession|policy-recession|bed-recession|small-review|analyse|figures|verify} [--dry-run]"
        return
    fi
    shift
    for option in "$@"; do
        case "$option" in --dry-run) dry=true ;; --resume) ;; *) echo "Unsupported option: $option" >&2; return 2 ;; esac
    done
    if [[ $stage == all ]]; then
        for item in verify-kernels pulses pulse-recession operational-preview native-review orientation storm-controls rolling-reference recession small-review policy-recession bed-recession endpoint-fields analyse verify; do run_stage "$item"; done
    else
        run_stage "$stage"
    fi
}
main "$@"
