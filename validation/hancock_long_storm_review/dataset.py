"""Read frozen runs, substituting verified repetitions of the integrated operator."""

import json


def records(*, review):
    folder = review / "integrated_events"
    verified = set()
    for receipt in folder.glob("*.verified"):
        verified.update(receipt.read_text().splitlines())
    rows = []
    for original in sorted((review / "events").glob("*.json")):
        source = folder / original.name if original.stem in verified else original
        record = json.loads(source.read_text())
        record["timing_dataset"] = source.parent.name
        rows.append(record)
    return rows
