# E018-final (Kaggle GPU): ONE reranker trained on ALL train top-10 pairs (same recipe as E018 folds), saved for test use.
# Val top-10 pairs are scored with this single model (val S1 never used in training) -> rerank_top10_full.npy
# (train rows keep the E018 OOF scores, so stage-2 training features stay out-of-fold).
import os, json, time, gzip, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

W = "/kaggle/working/e017"; MODEL = "intfloat/multilingual-e5-small"; REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAXLEN, BS, LR, EPOCHS = 128, 64, 3e-5, 1
SAVE = f"{W}/reranker_e5small_full"
t0 = time.time()
m = np.load(f"{W}/meta.npz"); ytr = m["y_tr"].astype(int); ntr = len(ytr)
rer_oof = np.load(f"{W}/rerank_top10.npy")
sel_tr = np.flatnonzero(np.isfinite(rer_oof[:ntr])); sel_va = ntr + np.flatnonzero(np.isfinite(rer_oof[ntr:]))
texts = {}
with gzip.open(f"{W}/texts.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t"); p += [""] * (4 - len(p)); texts[p[0]] = (p[1].strip() + " | " + p[2].strip()).strip(" |")
A, B = [], []
with gzip.open(f"{W}/pairs.tsv.gz", "rt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t"); A.append(p[3]); B.append(p[4])
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
def batches(idx, bs, shuffle, rng=None):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield j, tok([texts[A[k]] for k in j], [texts[B[k]] for k in j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
dev = "cuda:0"; torch.manual_seed(42)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV, num_labels=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(sel_tr) / BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
scaler = torch.amp.GradScaler(); yall = torch.tensor(ytr, dtype=torch.float32); rng = np.random.default_rng(42)
model.train(); t = time.time()
for ep in range(EPOCHS):
    for j, enc in batches(sel_tr, BS, True, rng):
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yall[j].to(dev))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
t_train = time.time() - t
model.eval(); s = np.zeros(len(sel_va), np.float32); pos = 0
with torch.no_grad():
    for j, enc in batches(sel_va, 256, False):
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            s[pos:pos + len(j)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        pos += len(j)
rer_full = rer_oof.copy(); rer_full[sel_va] = s
np.save(f"{W}/rerank_top10_full.npy", rer_full)
model.half().save_pretrained(SAVE); tok.save_pretrained(SAVE)   # fp16 weights (inference runs in fp16)
import shutil; shutil.make_archive(SAVE, 'zip', SAVE)
info = dict(model=MODEL, revision=REV, n_train_pairs=len(sel_tr), n_val_pairs=len(sel_va), t_train=t_train, epochs=EPOCHS, lr=LR, bs=BS,
            maxlen=MAXLEN, seed=42, corr_val_full_vs_foldmean=float(np.corrcoef(s, rer_oof[sel_va])[0, 1]), t_total=time.time() - t0)
json.dump(info, open(f"{W}/reranker_full_info.json", "w"), indent=1); print(json.dumps(info), flush=True)
