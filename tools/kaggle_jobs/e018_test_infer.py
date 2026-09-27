# E018 test inference (Kaggle 2xT4): score test base-top-10 pairs with the saved full-data reranker.
# Env: COUNTRY. Inputs /kaggle/working/e017/test/<COUNTRY>_pairs.tsv.gz, <COUNTRY>_texts.tsv.gz
# Output /kaggle/working/e017/test/<COUNTRY>_rr.npy (float32 logits aligned with pairs file order).
import os, gzip, time, json, threading, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
W = "/kaggle/working/e017"; C = os.environ["COUNTRY"]; MD = f"{W}/reranker_e5small_full"; MAXLEN = 128; BS = 512
t0 = time.time()
texts = {}
with gzip.open(f"{W}/test/{C}_texts.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t"); p += [""] * (4 - len(p)); texts[p[0]] = (p[1].strip() + " | " + p[2].strip()).strip(" |")
A, B = [], []
with gzip.open(f"{W}/test/{C}_pairs.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s, c = line.rstrip("\n").split("\t"); A.append(texts[s]); B.append(texts[c])
n = len(A); out = np.zeros(n, np.float32); print(C, "pairs", n, f"load {time.time()-t0:.0f}s", flush=True)
tok = AutoTokenizer.from_pretrained(MD)
parts = np.array_split(np.arange(n), torch.cuda.device_count())
def run(g):
    dev = f"cuda:{g}"; m = AutoModelForSequenceClassification.from_pretrained(MD).to(dev).half().eval(); idx = parts[g]
    with torch.no_grad():
        for i in range(0, len(idx), BS):
            j = idx[i:i + BS]
            enc = tok([A[k] for k in j], [B[k] for k in j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
            out[j] = m(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy()
t = time.time(); th = [threading.Thread(target=run, args=(g,)) for g in range(len(parts))]; [x.start() for x in th]; [x.join() for x in th]
np.save(f"{W}/test/{C}_rr.npy", out)
info = dict(country=C, n_pairs=n, t_infer=time.time() - t, pairs_per_s=n / (time.time() - t), t_total=time.time() - t0)
json.dump(info, open(f"{W}/test/{C}_rr_info.json", "w")); print(json.dumps(info), flush=True)
