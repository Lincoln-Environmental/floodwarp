"""Store large JSON measurement arrays with one complete experiment record per line."""

import json
from pathlib import Path


def compact_list(*, values):
    return (
        "[\n" + ",\n".join(json.dumps(row, separators=(",", ":"), allow_nan=False) for row in values) + "\n]"
    )


def archive(*, directory, threshold):
    for path in directory.rglob("*.json"):
        if path.stat().st_size < threshold:
            continue
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            content = compact_list(values=raw)
        elif isinstance(raw, dict):
            fields = []
            for key, value in raw.items():
                encoded = (
                    compact_list(values=value)
                    if isinstance(value, list)
                    else json.dumps(value, allow_nan=False)
                )
                fields.append(json.dumps(key) + ": " + encoded)
            content = "{\n" + ",\n".join(fields) + "\n}"
        else:
            continue
        if json.loads(content) != raw:
            raise ArithmeticError("Record changed during lossless serialisation")
        path.write_text(content + "\n")


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    threshold = 20000
    archive(directory=directory, threshold=threshold)
