# Contributing

Open an issue with a reproducible problem or a focused proposal. Work on dev;
main receives validated ordinary updates. Keep numerical research separate
from release fixes. Do not rewrite published tags or historical evidence.

Use Python 3.14 and the pinned numerical environment:

```bash
python -m pip install '.[test,ui]'
python -m pytest -q
```

Read [development](docs/DEVELOPMENT.md), [inputs](docs/INPUTS.md) and the
[frozen acceptance protocol](validation/ACCEPTANCE.md) before numerical work.
Preserve physical operators, FP64 FV1, CFL, sources, boundaries and restart
identity unless a separately reviewed scientific change explicitly addresses
them. Document formulation, inputs, device, source revision and limitations.
Do not relax tolerances to hide failures or substitute outlet agreement for
inundation validation.

Use named reusable functions, explicit keyword calls and pathlib paths.
Ordinary scripts keep adjustable settings and calls under
`if __name__ == "__main__":`. Avoid argparse, lambdas and source-path injection.
Keep changes small, retain attribution and data licences, and include meaningful
checks appropriate to the change. UI browser checks and headless solver checks
are separate evidence. Optional UI dependencies must stay optional.
