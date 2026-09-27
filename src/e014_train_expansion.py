"""
E014 -- Training-set expansion with the scalable retrieval engine.

Steps
 1. Sample N_NEW extra labelled train S1 uniformly (seed 42), excluding the 3,995 RECON sample (train AND val S1).
 2. Retrieve pools with retrieval_engine (RECON-04 config unchanged) for N_NEW + the 3,995 sample
    (the latter only to measure agreement with the frozen pool).
 3. Features (E009-D layout, old normalize) via pair_features with FULL-corpus train statistics, for:
    - original train/val pairs on the FROZEN pool (val pool never changes)
    - new train pairs on engine pools
 4. Arms (validation = the same 2,001 val S1 on the frozen pool):
    A  train = original 1,994 S1 (same pipeline/stats -> isolates the data effect)
    B  train = 1,994 + N_NEW
    Base model (competition block) re-fit on each arm's training set (OOF on train, full fit for val).
 Labels are used only for training-set targets and evaluation. Val S1 never enter training or statistics fitting.
"""
import os, sys, json, time, pickle, random, gc, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import build_pools as BP
import pair_features as PF
from recon05_baseline_scorer import normalize
import feature_pipeline as FP
from translit import normalize_fixed

OUT = os.path.join(H.ROOT, "experiments", "E014"); os.makedirs(OUT, exist_ok=True)
TRAIN = os.path.join(H.ROOT, "student_resource", "dataset", "train")
N_NEW = int(os.environ.get("E014_N_NEW", 10000))
NORM = os.environ.get("E014_NORM", "translit")   # feature normalizer: "translit" (E011-C) or "e008"
FNORM = (lambda x: normalize_fixed(x, translit=True)) if NORM == "translit" else normalize
CHUNK = 1500


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sample_new(D):
    excl = set(D["s1_dict"])
    rows = []
    with open(os.path.join(TRAIN, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] not in excl:
                rows.append(p[0])
    rng = random.Random(42)
    ids = set(rng.sample(rows, N_NEW))
    del rows
    s1 = {}
    with open(os.path.join(TRAIN, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in ids:
                p += [""] * (4 - len(p))
                s1[p[0]] = {"name": normalize(p[1]), "addr": normalize(p[2]), "country": p[3], "gt": set(), "raw": (p[1], p[2])}
    with open(os.path.join(TRAIN, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s1 and len(p) > 1 and p[1].strip():
                s1[p[0]]["gt"] = set(p[1].split(",")) - {""}
    return s1


def pools_for(D, s1_new):
    pools = {}
    for ctry in ["US", "India"]:
        path = os.path.join(BP.POOL_DIR, f"train_{ctry}_e014_{N_NEW}.pkl")
        if os.path.exists(path):
            P = pickle.load(open(path, "rb"))
        else:
            q = [s for s in s1_new if s1_new[s]["country"] == ctry] + [s for s in D["s1_dict"] if D["s1_dict"][s]["country"] == ctry]
            info = lambda s: s1_new[s] if s in s1_new else D["s1_dict"][s]
            P = BP.retrieve("train", ctry, q, [info(s)["name"] for s in q], [info(s)["addr"] for s in q], f"e014_{N_NEW}")
        pools[ctry] = P
    return pools


def pool_agreement(D, pools):
    frozen = D["cands"]; ov, orc_f, orc_e, n_gt = [], 0, 0, 0
    for ctry, P in pools.items():
        eng = BP.to_pool_dicts(P, subset=set(D["s1_dict"]))
        for s, pool in eng.items():
            a, b = set(frozen[s]), set(pool)
            ov.append(len(a & b) / max(len(a | b), 1))
            gt = D["s1_dict"][s]["gt"]; n_gt += len(gt)
            orc_f += len(gt & a); orc_e += len(gt & b)
    return dict(mean_jaccard=float(np.mean(ov)), min_jaccard=float(np.min(ov)), frac_identical=float(np.mean([o == 1.0 for o in ov])),
                frozen_pool_recall=orc_f / n_gt, engine_pool_recall=orc_e / n_gt, n_s1=len(ov))


def features(meta, s1_dict, cands, stats, idf, addr_key_cands):
    out = np.zeros((len(meta), 87), dtype=np.float32)
    by = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by[m[0]].append(i)
    s1s = list(by)
    for k in range(0, len(s1s), CHUNK):
        idx = [i for s in s1s[k:k + CHUNK] for i in by[s]]
        out[idx] = PF.label_free_block([meta[i] for i in idx], s1_dict, cands, stats, idf=idf, addr_key_cands=addr_key_cands)
    return out


def main():
    t_all = time.time()
    D = H.load_e008(verbose=False)
    stats = pickle.load(open(os.path.join(H.SHARED, "corpus_stats_train.pkl"), "rb"))
    idf = PF.make_idf(stats)
    fpath = os.path.join(OUT, f"e014_feats_{N_NEW}_{NORM}.pkl")
    if os.path.exists(fpath):
        C = pickle.load(open(fpath, "rb")); log("loaded cached features")
    else:
        s1_new = sample_new(D)
        log(f"sampled {len(s1_new):,} new S1", collections.Counter(v['country'] for v in s1_new.values()),
            "singletons", sum(1 for v in s1_new.values() if not v["gt"]))
        t0 = time.time(); pools = pools_for(D, s1_new); t_ret = time.time() - t0
        agree = pool_agreement(D, pools); log("pool agreement", agree)
        new_cands = {}
        for P in pools.values():
            new_cands.update(BP.to_pool_dicts(P, subset=set(s1_new)))
        del pools; gc.collect()
        meta_new = [(s, c, int(c in s1_new[s]["gt"])) for s in s1_new for c in new_cands[s]]
        # feature text: FNORM(raw); retrieval text & address-count keys stay original normalize()
        new_raw = BP.fetch_raw("train", {c for s in new_cands for c in new_cands[s]})
        new_cands_f = {s: {c: dict(ci, name=FNORM(new_raw[c][0]), addr=FNORM(new_raw[c][1])) for c, ci in pool.items()}
                       for s, pool in new_cands.items()}
        s1_new_f = {s: dict(v, name=FNORM(v["raw"][0]), addr=FNORM(v["raw"][1])) for s, v in s1_new.items()}
        del new_raw
        log(f"new pairs {len(meta_new):,}, positives {sum(m[2] for m in meta_new):,}, "
            f"pool recall {sum(m[2] for m in meta_new)/max(sum(len(v['gt']) for v in s1_new.values()),1):.4f}")
        t0 = time.time()
        raw = pickle.load(open(os.path.join(H.SHARED, "raw_text.pkl"), "rb"))
        s1f, cf = FP.make_text(D, raw, FNORM); del raw
        LF_tr = features(D["train_meta"], s1f, cf, stats, idf, D["cands"]); log("LF orig train")
        LF_va = features(D["val_meta"], s1f, cf, stats, idf, D["cands"]); log("LF val")
        del s1f, cf; gc.collect()
        LF_new = features(meta_new, s1_new_f, new_cands_f, stats, idf, new_cands); log("LF new")
        del new_cands_f, s1_new_f
        t_feat = time.time() - t0
        np.save(os.path.join(OUT, f"e014_LFnew_{N_NEW}_{NORM}.npy"), LF_new); del LF_new; gc.collect()
        C = dict(LF_tr=LF_tr, LF_va=LF_va, meta_new=meta_new, new_ids=list(s1_new),
                 s1_new={s: {k: (list(v) if k == "gt" else v) for k, v in d.items() if k != "raw"} for s, d in s1_new.items()},
                 agree=agree, t_ret=t_ret, t_feat=t_feat)
        pickle.dump(C, open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        del new_cands; gc.collect()
    y_new = np.array([m[2] for m in C["meta_new"]], dtype=np.int32)

    s1_all = dict(D["s1_dict"]); s1_all.update({s: dict(v, gt=set(v["gt"])) for s, v in C["s1_new"].items()})

    def run_arm(name, parts, y_train, meta_train, train_ids, seeds, params):
        g = H.golden
        n = sum(len(p) for p in parts)
        Xtr = np.empty((n, 99), dtype=np.float32); o = 0
        for p in parts:                      # fill in place (parts may be memmaps)
            Xtr[o:o + len(p), :22] = p[:, :22]; Xtr[o:o + len(p), 34:] = p[:, 22:]; o += len(p)
        oof_base, val_base = g.generate_oof_and_val_base_scores(Xtr[:, :22], y_train, meta_train, train_ids, C["LF_va"][:, :22])
        Xtr[:, 22:34] = g.extract_block_a_competition(meta_train, oof_base)
        Xva = PF.assemble(C["LF_va"], g.extract_block_a_competition(D["val_meta"], val_base))
        Dx = dict(D, train_meta=meta_train, train_s1_ids=train_ids, s1_dict=s1_all)
        res = {}
        for seed in seeds:
            fit = H.fit_stage2(Xtr, y_train, Xva, meta_train, train_ids, seed=seed, params=params)
            r = H.evaluate_arm(name, fit, Dx); res[seed] = r
            log(f"seed={seed} " + H.fmt(r, "hist")); log(f"seed={seed} " + H.fmt(r, "oof"))
        return res

    lam = float(os.environ.get("E014_LAMBDA", 1.0))
    params = dict(H.LGB_PARAMS, reg_lambda=lam)
    seeds = [42, 43, 44]
    A = run_arm("A_orig1994", [C["LF_tr"]], D["y_tr"], D["train_meta"], D["train_s1_ids"], seeds, params)
    gc.collect()
    LF_new = np.load(os.path.join(OUT, f"e014_LFnew_{N_NEW}_{NORM}.npy"), mmap_mode="r")
    y_all = np.concatenate([D["y_tr"], y_new])
    meta_all = list(D["train_meta"]) + C["meta_new"]; ids_all = list(D["train_s1_ids"]) + C["new_ids"]
    Bres = run_arm(f"B_plus{N_NEW}", [C["LF_tr"], LF_new], y_all, meta_all, ids_all, seeds, params)
    rep = dict(n_new=N_NEW, norm=NORM, reg_lambda=lam, agree=C["agree"], t_ret=C.get("t_ret"), t_feat=C.get("t_feat"), arms={})
    for proto in ["hist", "oof"]:
        for seed in seeds:
            a, b = A[seed][proto], Bres[seed][proto]
            d, lo, hi, p = H.paired_bootstrap(a["scores"], b["scores"])
            rep["arms"][f"s{seed}_{proto}"] = dict(A={k: v for k, v in a.items() if k != "scores"}, B={k: v for k, v in b.items() if k != "scores"},
                                                   delta=d, ci_lo=lo, ci_hi=hi)
            log(f"{proto} s{seed}: A {a['macro']*100:.2f} -> B {b['macro']*100:.2f} ({d*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]) "
                f"US {a['us']*100:.2f}->{b['us']*100:.2f} IN {a['india']*100:.2f}->{b['india']*100:.2f} FP {a['fp']}->{b['fp']} TP {a['tp']}->{b['tp']}")
    for proto in ["hist", "oof"]:
        am = np.mean([A[s][proto]["scores"] for s in seeds], axis=0); bm = np.mean([Bres[s][proto]["scores"] for s in seeds], axis=0)
        d, lo, hi, p = H.paired_bootstrap(am, bm); rep[f"seedavg_{proto}"] = dict(delta=d, ci_lo=lo, ci_hi=hi)
        log(f"seed-avg {proto}: {d*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]")
    rep["runtime_s"] = time.time() - t_all
    json.dump(rep, open(os.path.join(OUT, f"e014_results_{N_NEW}_{NORM}_lam{lam}.json"), "w"), indent=1)
    log(f"total {rep['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
