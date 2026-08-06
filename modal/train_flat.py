# NARC Next — flat NNUE teacher-distillation training on Modal
#
# Usage:
#   modal run modal_train.py                       # train on every .txt in the volume
#   modal run modal_train.py --epochs 24 --hidden 128 --lam 0.7
#
# Reads all training shards from the "narc-data" volume (/shards/*.txt plus any
# files you uploaded to /local/), trains the (768->H)x2 perspective net on an
# A10G, quantizes, and writes /data/narc.nnue back to the volume.
#
# Cost: A10G is ~$1.10/h; a full run on ~7M positions is well under one hour.
#
# NOTE: the engine is compiled with H=128 (nnue::H in src/nnue_params.h).
# If you train with a different --hidden, change that constant and rebuild.
import modal

app = modal.App("narc-next-train-flat")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
)

vol = modal.Volume.from_name("narc-data", create_if_missing=True)

QA, QB, SCALE = 255, 64, 400
PAD = 768
PIECES = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
          'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}


@app.function(image=image, volumes={"/data": vol}, gpu="A10G",
              memory=32768, cpu=8.0, timeout=2 * 3600)
def train(epochs: int = 32, hidden: int = 128, batch: int = 32768,
          lr: float = 1e-3, lam: float = 0.9,
          dataset: str = "sf18-gen3-v1",
          output: str = "narc-next-flat128-sf18-v1.nnue") -> dict:
    import glob
    import os
    import time

    import numpy as np
    import torch
    import torch.nn as nn

    vol.reload()
    # train on one dataset folder (e.g. "shards" for gen-1, "gen2" for self-play)
    files = sorted(glob.glob(f"/data/{dataset}/*.txt"))
    if not files:
        raise RuntimeError(f"no training files in /data/{dataset} (run modal_datagen.py first)")
    print(f"dataset: {dataset}  ({len(files)} files)")

    # ---------------- parse ----------------
    wIdx, bIdx, scores, results, stms = [], [], [], [], []
    for path in files:
        t0 = time.time()
        n = 0
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                try:
                    ws = int(parts[1])
                    wr = float(parts[2])
                except ValueError:
                    continue
                fields = parts[0].split()
                if len(fields) < 2:
                    continue
                stm = 0 if fields[1] == "w" else 1
                wi, bi = [], []
                sq = 56
                ok = True
                for ch in fields[0]:
                    if ch == "/":
                        sq -= 16
                    elif ch.isdigit():
                        sq += int(ch)
                    elif ch in PIECES:
                        pc = PIECES[ch]
                        color, pt = pc // 6, pc % 6
                        wi.append((color * 6 + pt) * 64 + sq)
                        bi.append(((color ^ 1) * 6 + pt) * 64 + (sq ^ 56))
                        sq += 1
                    else:
                        ok = False
                        break
                if not ok or len(wi) < 3 or len(wi) > 32:
                    continue
                wi += [PAD] * (32 - len(wi))
                bi += [PAD] * (32 - len(bi))
                wIdx.append(wi); bIdx.append(bi)
                scores.append(ws); results.append(wr); stms.append(stm)
                n += 1
        print(f"  {os.path.basename(path)}: {n} positions ({time.time()-t0:.0f}s)")

    wIdx = np.array(wIdx, dtype=np.int16)
    bIdx = np.array(bIdx, dtype=np.int16)
    scores = np.array(scores, dtype=np.float32)
    results = np.array(results, dtype=np.float32)
    stms = np.array(stms, dtype=np.int8)
    N = len(scores)
    print(f"total positions: {N:,}")

    stm_b = stms.astype(bool)
    score_stm = np.clip(np.where(stm_b, -scores, scores), -30000, 30000)
    result_stm = np.where(stm_b, 1.0 - results, results)
    idx_stm = np.where(stm_b[:, None], bIdx, wIdx).astype(np.int64)
    idx_nstm = np.where(stm_b[:, None], wIdx, bIdx).astype(np.int64)
    target = (lam * (1.0 / (1.0 + np.exp(-score_stm / SCALE)))
              + (1.0 - lam) * result_stm).astype(np.float32)

    rng = np.random.default_rng(7)
    perm = rng.permutation(N)
    nval = max(20000, N // 50)
    val, trn = perm[:nval], perm[nval:]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    # CPU-resident tensors; batches are moved to GPU per step
    t_stm = torch.from_numpy(idx_stm)
    t_nstm = torch.from_numpy(idx_nstm)
    t_tgt = torch.from_numpy(target)

    class Net(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.ft = nn.EmbeddingBag(769, h, mode="sum", padding_idx=PAD)
            self.ftb = nn.Parameter(torch.zeros(h))
            self.out = nn.Linear(2 * h, 1)
            nn.init.uniform_(self.ft.weight, -0.05, 0.05)
            with torch.no_grad():
                self.ft.weight[PAD].zero_()

        def forward(self, i_stm, i_nstm):
            a = torch.clamp(self.ft(i_stm) + self.ftb, 0, 1)
            b = torch.clamp(self.ft(i_nstm) + self.ftb, 0, 1)
            return self.out(torch.cat([a, b], dim=1)).squeeze(1)

    net = Net(hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.3)

    def run_eval(idxs):
        net.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(idxs), 65536):
                j = idxs[s:s + 65536]
                out = net(t_stm[j].to(dev), t_nstm[j].to(dev))
                tot += ((torch.sigmoid(out) - t_tgt[j].to(dev)) ** 2).sum().item()
                cnt += len(j)
        net.train()
        return tot / cnt

    final_val = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        ep = rng.permutation(trn)
        tot, cnt = 0.0, 0
        for s in range(0, len(ep), batch):
            j = ep[s:s + batch]
            out = net(t_stm[j].to(dev), t_nstm[j].to(dev))
            loss = ((torch.sigmoid(out) - t_tgt[j].to(dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                net.ft.weight[PAD].zero_()
            tot += loss.item() * len(j)
            cnt += len(j)
        sched.step()
        final_val = run_eval(val)
        print(f"epoch {epoch:2}: train {tot/cnt:.6f}  val {final_val:.6f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- quantize & export ----------------
    with torch.no_grad():
        W0 = net.ft.weight[:768].cpu().numpy()
        B0 = net.ftb.cpu().numpy()
        W1 = net.out.weight.cpu().numpy().reshape(-1)
        B1 = float(net.out.bias.item())

    W0q = np.clip(np.round(W0 * QA), -32767, 32767).astype(np.int16)
    B0q = np.clip(np.round(B0 * QA), -32767, 32767).astype(np.int16)
    W1q = np.clip(np.round(W1 * QB), -32767, 32767).astype(np.int16)
    B1q = int(round(B1 * QA * QB))

    output_path = f"/data/{output}"
    with open(output_path, "wb") as f:
        f.write(b"NARCNET1")
        f.write(np.int32(hidden).tobytes())
        f.write(W0q.tobytes())
        f.write(B0q.tobytes())
        f.write(W1q.tobytes())
        f.write(np.int32(B1q).tobytes())
    vol.commit()
    size = 8 + 4 + W0q.nbytes + B0q.nbytes + W1q.nbytes + 4
    print(f"saved {output_path} to volume ({size//1024} KB)")
    return {"positions": N, "validation_loss": final_val, "output": output}


@app.local_entrypoint()
def main(epochs: int = 32, hidden: int = 128, batch: int = 32768,
         lr: float = 1e-3, lam: float = 0.9,
         dataset: str = "sf18-gen3-v1",
         output: str = "narc-next-flat128-sf18-v1.nnue"):
    result = train.remote(epochs=epochs, hidden=hidden, batch=batch, lr=lr, lam=lam,
                          dataset=dataset, output=output)
    print("\n================ TRAINING COMPLETE ================")
    print(result)
    print("download the net with:")
    print(f"  modal volume get narc-data {output} .")
