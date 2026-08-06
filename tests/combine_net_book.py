"""Append a NARCBOOK payload to a NARCNET file for single-file embedding."""

from pathlib import Path
import sys

network = Path(sys.argv[1])
book = Path(sys.argv[2])
output = Path(sys.argv[3])
network_data = network.read_bytes()
book_data = book.read_bytes()
if not network_data.startswith(b"NARCNET7"):
    raise SystemExit("network is not NARCNET7")
if not book_data.startswith(b"NARCBOOK"):
    raise SystemExit("book is not NARCBOOK")
output.write_bytes(network_data + book_data)
print(f"Combined {len(network_data)} network bytes and {len(book_data)} book bytes -> {output}")
