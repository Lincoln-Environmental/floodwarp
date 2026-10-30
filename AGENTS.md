# FloodWarp maintenance

Work on dev; main receives validated ordinary updates. Do not rewrite either
branch. Keep the separate Overlander research repository untouched.
Do not spawn agents or create autonomous goals.

Read docs/DEVELOPMENT.md, docs/INPUTS.md and validation/ACCEPTANCE.md before
numerical changes. Preserve the frozen baseline defaults and reference data.
Both SGS modes remain explicitly experimental. No coupled-model interfaces,
prepared-model dependency, inclined SGS, routing, surrogate or campaign code.

Use Python 3.14 and the pinned environment. Run python -m pytest -q.
Scripts use named reusable functions, pathlib.Path, explicit keyword calls,
and adjustable settings under the exact if __name__ == "__main__": guard.
Avoid argparse, lambdas and path injection. Keep source provenance current.

The repository remains private while LICENSE and docs/RELEASE_STATUS.md list
unresolved distribution/qualification requirements.
