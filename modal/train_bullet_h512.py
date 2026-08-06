"""Convert NARC data to bulletformat and train a pawn-free H512 NNUE."""

from __future__ import annotations

import glob
import os
from array import array
from pathlib import Path
import random
import shutil
import struct
import subprocess
import time

import modal


BULLET_COMMIT = "cebc78a093d92cbc87e56cfef049184c225270b0"
DEFAULT_DATASETS = (
    "sf18-gen3-v2-5m",
    "sf18-gen3-deep5m",
    "sf18-gen4-see-fixed-v1",
    "sf18-teacher-v1",
)
QA, QB = 255, 64
HIDDEN, OUTPUT_BUCKETS = 512, 8

app = modal.App("narc-next-bullet-h512")
volume = modal.Volume.from_name("narc-data")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("build-essential", "ca-certificates", "curl", "git", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain 1.88.0",
        "git clone https://github.com/jw1912/bullet /opt/bullet",
        f"cd /opt/bullet && git checkout {BULLET_COMMIT}",
    )
    .env({"CUDA_PATH": "/usr/local/cuda"})
    .run_commands(
        "cd /opt/bullet && /root/.cargo/bin/cargo build --release --package bullet-utils"
    )
    .add_local_dir("modal/bullet_h512", "/trainer", copy=True)
    .run_commands("cd /trainer && /root/.cargo/bin/cargo build --release")
)


def _dataset_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _write_tensor(stream, name: str, values: array) -> None:
    encoded = name.encode("ascii") + b"\n"
    stream.write(encoded)
    stream.write(struct.pack("<Q", len(values)))
    stream.write(values.tobytes())


def _expanded_champion_weights(source: Path, destination: Path) -> None:
    """Expand NARCNT14 H256 weights into Bullet's pawn-free H512 layout."""
    data = source.read_bytes()
    if data[:8] != b"NARCNT14":
        raise RuntimeError(f"{source} is not the expected NARCNT14 champion")
    hidden, buckets, pawns = struct.unpack_from("<iii", data, 8)
    if (hidden, buckets, pawns) != (256, OUTPUT_BUCKETS, 16):
        raise RuntimeError(f"unexpected champion dimensions: {(hidden, buckets, pawns)}")
    offset = 20
    old_w0 = array("h")
    old_w0.frombytes(data[offset : offset + 768 * hidden * 2])
    offset += 768 * hidden * 2
    old_b0 = array("h")
    old_b0.frombytes(data[offset : offset + hidden * 2])
    offset += hidden * 2
    old_width = 2 * hidden + 2 * pawns
    old_w1 = array("h")
    old_w1.frombytes(data[offset : offset + OUTPUT_BUCKETS * old_width * 2])
    offset += OUTPUT_BUCKETS * old_width * 2
    old_b1 = array("i")
    old_b1.frombytes(data[offset : offset + OUTPUT_BUCKETS * 4])

    rng = random.Random(0x4E415243)
    l0w = array("f")
    for feature in range(768):
        base = feature * hidden
        l0w.extend(value / QA for value in old_w0[base : base + hidden])
        # New neurons begin with useful nonzero features, while their zeroed
        # output weights initially leave the champion approximation intact.
        l0w.extend(rng.gauss(0.0, (2.0 / 768.0) ** 0.5) for _ in range(hidden))
    l0b = array("f", (value / QA for value in old_b0))
    l0b.extend([0.0] * hidden)

    # Bullet stores affine matrices column-major. Build [input][bucket]; its
    # SavedFormat transpose later emits NARC's [bucket][input] hot-path order.
    l1w = array("f")
    for input_index in range(2 * HIDDEN):
        for bucket in range(OUTPUT_BUCKETS):
            old_index = None
            if input_index < hidden:
                old_index = input_index
            elif HIDDEN <= input_index < HIDDEN + hidden:
                old_index = hidden + input_index - HIDDEN
            value = old_w1[bucket * old_width + old_index] / QB if old_index is not None else 0.0
            l1w.append(value)
    l1b = array("f", (value / (QA * QB) for value in old_b1))

    with destination.open("wb") as stream:
        # ModelWeights serializes its BTreeMap in lexical name order.
        _write_tensor(stream, "l0b", l0b)
        _write_tensor(stream, "l0w", l0w)
        _write_tensor(stream, "l1b", l1b)
        _write_tensor(stream, "l1w", l1w)


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=8192, timeout=3600)
def prepare(datasets: str = ",".join(DEFAULT_DATASETS), force: bool = False) -> dict:
    """CPU-only text-to-bulletformat conversion; this spends no GPU credit."""
    volume.reload()
    destination = Path("/data/bullet-h512-v1")
    destination.mkdir(parents=True, exist_ok=True)
    converted = skipped = positions = 0
    for dataset in _dataset_names(datasets):
        paths = sorted(Path("/data", dataset).glob("*.txt"))
        if not paths:
            raise RuntimeError(f"no text shards found in {dataset}")
        for index, source in enumerate(paths, 1):
            target = destination / f"{dataset}__{source.stem}.bf"
            if target.exists() and not force:
                skipped += 1
                positions += target.stat().st_size // 32
                continue
            temporary = target.with_suffix(".bf.tmp")
            subprocess.run(
                [
                    "/opt/bullet/target/release/bullet-utils",
                    "convert",
                    "--from",
                    "text",
                    "--input",
                    str(source),
                    "--output",
                    str(temporary),
                ],
                check=True,
            )
            temporary.replace(target)
            converted += 1
            positions += target.stat().st_size // 32
            if converted % 20 == 0:
                print(f"converted {converted} shards ({positions:,} positions)", flush=True)
    volume.commit()
    return {
        "datasets": _dataset_names(datasets),
        "converted_shards": converted,
        "cached_shards": skipped,
        "positions": positions,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    cpu=8,
    memory=16384,
    timeout=5000,
)
def train(
    output: str = "narc-next-h512-bullet-major-wdl30-v39.nnue",
    superbatches: int = 24,
    batches_per_superbatch: int = 1024,
    batch_size: int = 16384,
    initial: str = "",
    initial_lr: float = 0.001,
    final_lr: float = 0.00005,
    wdl: float = 0.30,
) -> dict:
    """Run a single wall-clock-capped Bullet campaign and export NARCNT15."""
    if not 1 <= superbatches <= 40:
        raise ValueError("superbatches must be between 1 and 40")
    if not 128 <= batches_per_superbatch <= 4096:
        raise ValueError("batches_per_superbatch must be between 128 and 4096")
    volume.reload()
    files = sorted(glob.glob("/data/bullet-h512-v1/*.bf"))
    if not files:
        raise RuntimeError("run prepare first; no bulletformat shards were found")
    positions = sum(os.path.getsize(path) // 32 for path in files)
    checkpoint_dir = "/tmp/narc-bullet-checkpoints"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    env = os.environ.copy()
    env.update(
        {
            "NARC_BULLET_FILES": "\n".join(files),
            "NARC_SUPERBATCHES": str(superbatches),
            "NARC_BATCHES_PER_SUPERBATCH": str(batches_per_superbatch),
            "NARC_BATCH_SIZE": str(batch_size),
            "NARC_CHECKPOINT_DIR": checkpoint_dir,
            "NARC_INITIAL_LR": str(initial_lr),
            "NARC_FINAL_LR": str(final_lr),
            "NARC_WDL": str(wdl),
        }
    )
    if initial:
        initial_path = Path("/data", initial)
        if not initial_path.exists():
            raise RuntimeError(f"initial network not found: {initial}")
        expanded_path = Path("/tmp/narc-expanded-champion.weights")
        _expanded_champion_weights(initial_path, expanded_path)
        env["NARC_INITIAL_WEIGHTS"] = str(expanded_path)
    started = time.time()
    subprocess.run(["/trainer/target/release/narc-bullet-h512"], env=env, check=True)
    elapsed = time.time() - started

    candidates = sorted(
        Path(checkpoint_dir).glob("**/quantised.bin"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError("Bullet completed without producing quantised.bin")
    raw = candidates[-1].read_bytes()
    payload_size = (
        768 * HIDDEN * 2
        + HIDDEN * 2
        + OUTPUT_BUCKETS * 2 * HIDDEN * 2
        + OUTPUT_BUCKETS * 4
    )
    if len(raw) < payload_size:
        raise RuntimeError(f"short Bullet network: {len(raw)} < {payload_size}")
    # NARCNT15 is H512 SCReLU, NARC major-piece output buckets, and no
    # handcrafted pawn features. Bullet's trailing 64-byte padding is omitted.
    output_path = Path("/data/networks", output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        b"NARCNT15"
        + struct.pack("<iii", HIDDEN, OUTPUT_BUCKETS, 0)
        + raw[:payload_size]
    )
    volume.commit()
    return {
        "positions": positions,
        "samples_seen": superbatches * batches_per_superbatch * batch_size,
        "seconds": elapsed,
        "output": f"networks/{output}",
        "bytes": output_path.stat().st_size,
        "bullet_commit": BULLET_COMMIT,
        "initial": initial,
        "wdl": wdl,
    }


@app.local_entrypoint()
def main(
    action: str = "prepare",
    datasets: str = ",".join(DEFAULT_DATASETS),
    output: str = "narc-next-h512-bullet-major-wdl30-v39.nnue",
    superbatches: int = 24,
    batches_per_superbatch: int = 1024,
    batch_size: int = 16384,
    initial: str = "",
    initial_lr: float = 0.001,
    final_lr: float = 0.00005,
    wdl: float = 0.30,
    force: bool = False,
):
    if action == "prepare":
        print(prepare.remote(datasets, force))
    elif action == "train":
        print(
            train.remote(
                output, superbatches, batches_per_superbatch, batch_size,
                initial, initial_lr, final_lr, wdl,
            )
        )
        print(f"modal volume get narc-data networks/{output} networks/{output}")
    else:
        raise ValueError("action must be 'prepare' or 'train'")
