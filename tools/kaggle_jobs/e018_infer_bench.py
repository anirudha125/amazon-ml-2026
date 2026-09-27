# Inference throughput benchmark for the cross-encoder (both T4s, fp16, pre-tokenized vs on-the-fly). Diagnostics only.
import time, threading, gzip, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL, REV = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"
W = "/kaggle/working/e017"
texts = []
with gzip.open(f"{W}/texts.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for i, line in enumerate(f):
        p = line.rstrip("\n").split("\t"); texts.append(p[1] + " | " + (p[2] if len(p) > 2 else ""))
        if i >= 60000: break
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
A = texts[:30000]; B = texts[30000:60000]
t = time.time(); enc = tok(A, B, truncation=True, max_length=128, padding="longest", return_tensors="pt"); t_tok = time.time() - t
print(f"tokenize 30k pairs: {t_tok:.1f}s ({30000/t_tok:.0f} pairs/s, 1 CPU thread)", flush=True)
res = {}
def run(g):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV, num_labels=1).to(f"cuda:{g}").half().eval()
    ids, am = enc["input_ids"].to(f"cuda:{g}"), enc["attention_mask"].to(f"cuda:{g}")
    with torch.no_grad():
        for bs in [512]:
            torch.cuda.synchronize(g); t = time.time()
            for i in range(0, 30000, bs):
                m(input_ids=ids[i:i+bs], attention_mask=am[i:i+bs])
            torch.cuda.synchronize(g); res[g] = 30000 / (time.time() - t)
th = [threading.Thread(target=run, args=(g,)) for g in range(torch.cuda.device_count())]
[x.start() for x in th]; [x.join() for x in th]
print("GPU inference pairs/s per GPU (pre-tokenized, padded to longest in 30k):", {g: round(v) for g, v in res.items()}, flush=True)
