"""Inspect the existing NARC self-play corpus stored on Modal."""

from pathlib import Path

import modal

app = modal.App("narc-next-inventory")
volume = modal.Volume.from_name("narc-data")


@app.function(volumes={"/data": volume}, timeout=1800)
def inventory(dataset: str = "gen3") -> dict:
    files = sorted((Path("/data") / dataset).glob("*.txt"))
    rows = []
    total_lines = 0
    total_bytes = 0
    for path in files:
        with path.open("rb") as stream:
            count = sum(1 for _ in stream)
        size = path.stat().st_size
        rows.append((path.name, count, size))
        total_lines += count
        total_bytes += size
    return {
        "dataset": dataset,
        "files": len(files),
        "lines": total_lines,
        "bytes": total_bytes,
        "smallest": min(rows, key=lambda row: row[1]) if rows else None,
        "largest": max(rows, key=lambda row: row[1]) if rows else None,
    }


@app.local_entrypoint()
def main(dataset: str = "gen3"):
    print(inventory.remote(dataset))

