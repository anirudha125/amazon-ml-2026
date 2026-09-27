# E018 (Kaggle 2xT4) -- fine-tuned cross-encoder reranker, OOF, as an extra stage-2 feature. RUN ONLY IF E017 GATE PASSES.
# Candidates: top-K per S1 by BASE-model OOF score (22 feats; no stage-2 leakage). 5 group folds over train S1 (same KFold
# as E014). Train fold models on 4 folds' top-K pairs, score held-out fold (OOF) + all val top-K pairs (mean over folds).
# Output: /kaggle/working/e017/rerank.npy float32 [n_pairs] (NaN outside top-K), aligned with pairs.tsv order.
import os, json, time, gzip, threading, math
import numpy as np, torch, lightgbm as lgb
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification

W = "/kaggle/working/e017"
MODEL = os.environ.get("RR_MODEL", "intfloat/multilingual-e5-small"); REV = os.environ.get("RR_REV", "614241f622f53c4eeff9890bdc4f31cfecc418b3")
TAG = os.environ.get("RR_TAG", "")
TOPK, MAXLEN, BS, LR, EPOCHS = int(os.environ.get("TOPK", 10)), 128, 64, 3e-5, 1
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1)
t0 = time.time(); info = dict(model=MODEL, revision=REV, topk=TOPK, maxlen=MAXLEN, bs=BS, lr=LR, epochs=EPOCHS)

m = np.load(f"{W}/meta.npz"); s1tr, ytr = m["s1idx_tr"], m["y_tr"].astype(int); s1va, yva = m["s1idx_va"], m["y_va"].astype(int)
ntr, n_s1_tr = len(ytr), len(m["gt_tr"])
LF = np.vstack([np.load(f"{W}/LF_tr_orig.npy", mmap_mode="r")[:, :22], np.load(f"{W}/LF_tr_new.npy", mmap_mode="r")[:, :22]])
LFva = np.load(f"{W}/LF_va.npy", mmap_mode="r")[:, :22]

def folds(s1idx, n_s1, seed=42):
    for _, va_g in KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n_s1)):
        mk = np.zeros(n_s1, bool); mk[va_g] = True; va = mk[s1idx]; yield np.flatnonzero(~va), np.flatnonzero(va)

SMOKE = os.environ.get("SMOKE") == "1"
FOLDS = list(folds(s1tr, n_s1_tr))
if SMOKE:        # code-path test only: proxy selection score, no LightGBM
    oof = np.asarray(LF[:, 3], np.float32); pva = np.asarray(LFva[:, 3], np.float32)
else:
    oof = np.zeros(ntr, np.float32)
    for tr, va in FOLDS:
        oof[va] = lgb.LGBMClassifier(**P).fit(LF[tr], ytr[tr]).predict_proba(LF[va])[:, 1]
    pva = lgb.LGBMClassifier(**P).fit(LF, ytr).predict_proba(np.asarray(LFva))[:, 1]

def topk_mask(s1idx, p, k):
    order = np.lexsort((-p, s1idx)); g = s1idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, len(p)])
    rank = np.arange(len(p)) - np.repeat(starts, K); mk = np.zeros(len(p), bool); mk[order[rank < k]] = True; return mk

sel_tr = topk_mask(s1tr, oof, TOPK); sel_va = topk_mask(s1va, pva, TOPK)
info.update(sel_tr=int(sel_tr.sum()), sel_tr_pos=int(ytr[sel_tr].sum()), pos_tr=int(ytr.sum()),
            sel_va=int(sel_va.sum()), sel_va_pos=int(yva[sel_va].sum()), pos_va=int(yva.sum()), t_base=time.time() - t0)
print(info, flush=True)

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
    idx = np.array(idx); idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[A[k]] for k in j], [texts[B[k]] for k in j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        yield j, enc

def train_and_score(gpu, tr_idx, score_idx, out, key):
    dev = f"cuda:{gpu}"; torch.manual_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(tr_idx) / BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(); yall = torch.tensor(ytr, dtype=torch.float32)
    model.train(); rng = np.random.default_rng(42); t = time.time()
    for ep in range(EPOCHS):
        for j, enc in batches(tr_idx, BS, True, rng):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yall[j].to(dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    t_train = time.time() - t; model.eval(); res = {}
    with torch.no_grad():
        for name, idx in score_idx.items():
            s = np.zeros(len(idx), np.float32); pos = 0
            for j, enc in batches(idx, 256, False):
                enc = {k: v.to(dev) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    s[pos:pos + len(j)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
                pos += len(j)
            res[name] = s
    out[key] = dict(res=res, t_train=t_train, n_train=len(tr_idx), pairs_per_s_train=len(tr_idx) * EPOCHS / t_train)
    print(key, f"train {t_train:.0f}s ({len(tr_idx)*EPOCHS/t_train:.0f} pairs/s)", flush=True)
    del model; torch.cuda.empty_cache()

rer = np.full(ntr + len(yva), np.nan, np.float32)
va_idx = ntr + np.flatnonzero(sel_va)
out = {}
jobs = [(f, np.flatnonzero(sel_tr & np.isin(np.arange(ntr), FOLDS[f][0])), np.flatnonzero(sel_tr & np.isin(np.arange(ntr), FOLDS[f][1]))) for f in range(5)]
if SMOKE:
    jobs = [(f, tr_i[:3000], ho_i[:1000]) for f, tr_i, ho_i in jobs[:1]]; va_idx = va_idx[:1000]
    train_and_score(0, jobs[0][1], {"ho": jobs[0][2], "va": va_idx}, out, 0)
    print("SMOKE OK", {k: float(np.mean(v)) for k, v in out[0]["res"].items()}, out[0]["pairs_per_s_train"]); raise SystemExit
for i in range(0, 5, torch.cuda.device_count()):
    th = []
    for g, (f, tr_i, ho_i) in enumerate(jobs[i:i + torch.cuda.device_count()]):
        th.append(threading.Thread(target=train_and_score, args=(g, tr_i, {"ho": ho_i, "va": va_idx}, out, f)))
    [t.start() for t in th]; [t.join() for t in th]
for f, tr_i, ho_i in jobs:
    rer[ho_i] = out[f]["res"]["ho"]
rer[va_idx] = np.mean([out[f]["res"]["va"] for f in range(5)], axis=0)
np.save(f"{W}/rerank_top{TOPK}{TAG}.npy", rer)
info.update(t_total=time.time() - t0, fold_train_s=[out[f]["t_train"] for f in range(5)],
            train_pairs_per_s=[out[f]["pairs_per_s_train"] for f in range(5)])
info["n_params"] = None
json.dump(info, open(f"{W}/rerank_info_top{TOPK}{TAG}.json", "w"), indent=1); print(json.dumps(info), flush=True)
