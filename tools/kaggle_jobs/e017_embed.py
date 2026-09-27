# E017 (Kaggle 2xT4): frozen multilingual embedding cosines for the E014-B train + val pairs.
# Inputs  (uploaded): /kaggle/working/e017/pairs.tsv.gz, texts.tsv.gz   (competition train data only)
# Output : /kaggle/working/e017/cos.npy  float32 [n_pairs, 3] = cos(name), cos(name+addr), cos(addr) (-1 if an addr empty)
#          /kaggle/working/e017/run_info.json
import os, time, json, gzip, threading
import numpy as np, torch
from sentence_transformers import SentenceTransformer

W = "/kaggle/working/e017"
MODEL = "intfloat/multilingual-e5-small"
REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"          # pinned commit (MIT license, verified from model card)
BATCH, MAXLEN = 512, 64
t0 = time.time()
info = dict(model=MODEL, revision=REV, batch=BATCH, max_seq_length=MAXLEN, gpus=torch.cuda.device_count())

ids, names, addrs = [], [], []
with gzip.open(f"{W}/texts.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t"); p += [""] * (4 - len(p))
        ids.append(p[0]); names.append(p[1].strip()); addrs.append(p[2].strip())
pos = {s: i for i, s in enumerate(ids)}
pa, pb = [], []
with gzip.open(f"{W}/pairs.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t"); pa.append(pos[p[3]]); pb.append(pos[p[4]])
pa = np.array(pa); pb = np.array(pb)
info.update(n_texts=len(ids), n_pairs=len(pa), t_load=time.time() - t0)
print(info, flush=True)

models = [SentenceTransformer(MODEL, revision=REV, device=f"cuda:{g}") for g in range(torch.cuda.device_count())]
for m in models:
    m.max_seq_length = MAXLEN; m.half()
info["n_params"] = int(sum(p.numel() for p in models[0].parameters()))
print("params", info["n_params"], flush=True)

def encode(texts):
    """Split across GPUs in parallel threads; L2-normalized float16 embeddings."""
    k = len(models); parts = np.array_split(np.arange(len(texts)), k); out = [None] * k
    def run(g):
        out[g] = models[g].encode([texts[i] for i in parts[g]], batch_size=BATCH, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    th = [threading.Thread(target=run, args=(g,)) for g in range(k)]
    [t.start() for t in th]; [t.join() for t in th]
    return np.vstack(out)

views = {"name": ["query: " + n for n in names],
         "full": ["query: " + (n + ", " + a if a else n) for n, a in zip(names, addrs)],
         "addr": ["query: " + a for a in addrs]}
cos = np.zeros((len(pa), 3), dtype=np.float32)
for j, (v, texts) in enumerate(views.items()):
    t = time.time(); E = encode(texts); te = time.time() - t
    Et = torch.from_numpy(E).cuda(0)
    for a in range(0, len(pa), 200000):
        ia = torch.from_numpy(pa[a:a + 200000]).cuda(0); ib = torch.from_numpy(pb[a:a + 200000]).cuda(0)
        cos[a:a + 200000, j] = (Et[ia].float() * Et[ib].float()).sum(1).cpu().numpy()
    del Et; torch.cuda.empty_cache()
    info[f"t_encode_{v}"] = te; info[f"texts_per_s_{v}"] = len(texts) / te
    print(v, f"{te:.0f}s", f"{len(texts)/te:.0f} texts/s", flush=True)
empty = np.array([not a for a in addrs])
cos[empty[pa] | empty[pb], 2] = -1.0
np.save(f"{W}/cos.npy", cos)
info["t_total"] = time.time() - t0
info["gpu_mem_peak_gb"] = [torch.cuda.max_memory_allocated(g) / 2 ** 30 for g in range(torch.cuda.device_count())]
json.dump(info, open(f"{W}/run_info.json", "w"), indent=1)
print(json.dumps(info), flush=True)
