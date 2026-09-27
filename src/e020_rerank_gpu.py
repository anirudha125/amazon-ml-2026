"""
E020 GPU step: score the dense-slice base top-10 pairs with the PRODUCTION fine-tuned reranker (same code path as
tools/kaggle_jobs/e018_test_infer.py: text = "name | addr", max_len 128, fp16, logits). Output experiments/E020_maxclaimer/rr.npy.
"""
import os, gzip, time, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "experiments", "E020_maxclaimer"); MD = os.path.join(ROOT, "experiments", "E017_embed", "reranker_e5small_full")
MAXLEN, BS, TAB, NL = 128, 1024, chr(9), chr(10)
DEV = os.environ.get("DEV", "cuda"); LIMIT = int(os.environ.get("LIMIT", 0))   # LIMIT=n: CPU dry run on first n pairs
t0 = time.time(); texts = {}
with gzip.open(f"{W}/texts.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip(NL).split(TAB); p += [""] * (4 - len(p)); texts[p[0]] = (p[1].strip() + " | " + p[2].strip()).strip(" |")
A, B = [], []
with gzip.open(f"{W}/pairs.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        s, c = line.rstrip(NL).split(TAB); A.append(texts[s]); B.append(texts[c])
        if LIMIT and len(A) >= LIMIT: break
n = len(A); out = np.zeros(n, np.float32); print("pairs", n, f"load {time.time()-t0:.0f}s", flush=True)
tok = AutoTokenizer.from_pretrained(MD)
m = AutoModelForSequenceClassification.from_pretrained(MD).to(DEV).eval()
if DEV == "cuda": m = m.half()
t = time.time()
with torch.inference_mode():
    for i in range(0, n, BS):
        enc = tok(A[i:i + BS], B[i:i + BS], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        out[i:i + BS] = m(**{k: v.to(DEV) for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy()
np.save(f"{W}/rr{'_smoke' if LIMIT else ''}.npy", out)
info = dict(n_pairs=n, t_infer=round(time.time() - t, 1), pairs_per_s=round(n / (time.time() - t)), gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu")
json.dump(info, open(f"{W}/rr_info{'_smoke' if LIMIT else ''}.json", "w")); print(json.dumps(info), flush=True)
