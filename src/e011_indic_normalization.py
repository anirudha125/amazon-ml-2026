"""
E011 -- Indic normalization / transliteration attribution on the FROZEN pool, on top of E009-D.

Arms (stage-2 LightGBM, E008 params; seeds 42/43/44; both threshold protocols; paired bootstrap vs A):
  A  E009-D (cached 99 cols)
  B  fixed Unicode normalization (keep combining marks), no transliteration -- full rebuild
  C  fixed normalization + rule-based transliteration of Indic text -- full rebuild
  D  E009-D + 9 transliteration-derived features (no rebuild)
  E  fixed normalization + transliteration + accent folding -- full rebuild
Transliteration is rule-based from Unicode character names (src/translit.py): no labels, no external data.
The retrieval pool is NOT changed.
"""
import os, sys, json, time, pickle, collections
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import feature_pipeline as FP
from translit import normalize_fixed, transliterate, has_indic, _simplify
from recon05_baseline_scorer import normalize as norm_e008

OUT = os.path.join(H.ROOT, "experiments", "E011")
os.makedirs(OUT, exist_ok=True)

TR_NAMES = ["cand_name_indic", "cand_addr_indic", "tr_lev", "tr_jw", "tr_tsort", "tr_tset",
            "tr_soft_s1_cov", "skel_lev", "skel_tset"]
_VOW = set("aeiouy")

def phon(s):
    return " ".join(_simplify(w) for w in s.split())

def skel(s):
    out = []
    for w in s.split():
        w = w.replace("x", "ks").replace("q", "k").replace("z", "j").replace("f", "p").replace("c", "k")
        w = w.translate(str.maketrans("bdg", "ptk"))
        out.append(w[:1] + "".join(ch for ch in w[1:] if ch not in _VOW))
    return " ".join(out)

def tr_feats(meta, s1n, cn, raw):
    X = np.zeros((len(meta), len(TR_NAMES)), dtype=np.float32)
    cache = {}
    for i, (s, c, _) in enumerate(meta):
        rn, ra, _ = raw[c]
        t = cache.get(c)
        if t is None:
            tn = phon(normalize_fixed(rn, translit=True)); t = (tn, skel(tn), has_indic(rn), has_indic(ra)); cache[c] = t
        tn, sk, ind_n, ind_a = t
        sn = phon(s1n[s]["name"]); ss = skel(sn)
        toks_c = tn.split()
        cov = 0.0
        st = sn.split()
        if st and toks_c:
            cov = sum(1 for w in st if any(JaroWinkler.similarity(w, o) >= 0.85 for o in toks_c)) / len(st)
        X[i] = [float(ind_n), float(ind_a), Levenshtein.normalized_similarity(sn, tn), JaroWinkler.similarity(sn, tn),
                fuzz.token_sort_ratio(sn, tn) / 100, fuzz.token_set_ratio(sn, tn) / 100, cov,
                Levenshtein.normalized_similarity(ss, sk), fuzz.token_set_ratio(ss, sk) / 100]
    return X

def cached_build(name, D, raw, norm):
    p = os.path.join(OUT, f"feats_{name}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    Xtr, Xva = FP.build(D, raw, norm)
    pickle.dump((Xtr, Xva), open(p, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return Xtr, Xva

def main():
    t_all = time.time()
    D = H.load_e008(verbose=False)
    raw = pickle.load(open(os.path.join(H.SHARED, "raw_text.pkl"), "rb"))
    F9 = pickle.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_features.pkl"), "rb"))
    A_tr = np.hstack([D["X_tr"], F9["NUM_tr"], F9["TOK_tr"]]); A_va = np.hstack([D["X_va"], F9["NUM_va"], F9["TOK_va"]])
    del F9
    s1_orig = D["s1_dict"]
    t0 = time.time()
    p = os.path.join(OUT, "feats_TR.pkl")
    if os.path.exists(p):
        TR_tr, TR_va = pickle.load(open(p, "rb"))
    else:
        TR_tr = tr_feats(D["train_meta"], s1_orig, D["cands"], raw); TR_va = tr_feats(D["val_meta"], s1_orig, D["cands"], raw)
        pickle.dump((TR_tr, TR_va), open(p, "wb"))
    print(f"TR feats {time.time()-t0:.0f}s", flush=True)
    arms = {"A_E009D": (A_tr, A_va), "D_E009D+TR": (np.hstack([A_tr, TR_tr]), np.hstack([A_va, TR_va]))}
    builds = {"B_fixnorm": lambda x: normalize_fixed(x),
              "C_fixnorm_translit": lambda x: normalize_fixed(x, translit=True),
              "E_translit_fold": lambda x: normalize_fixed(x, translit=True, fold_accents=True)}
    build_s = {}
    for name, fn in builds.items():
        t0 = time.time()
        arms[name] = cached_build(name, D, raw, fn)
        build_s[name] = time.time() - t0
        print(f"built {name} in {build_s[name]:.0f}s", flush=True)
    seeds = [42, 43, 44]
    res = collections.defaultdict(dict); preds = {}
    for seed in seeds:
        for name, (Xtr, Xva) in arms.items():
            fit = H.fit_stage2(Xtr, D["y_tr"], Xva, D["train_meta"], D["train_s1_ids"], seed=seed)
            r = H.evaluate_arm(name, fit, D)
            res[seed][name] = r
            if seed == 42:
                preds[name] = dict(p_va=fit["p_va"], p_tr_oof=fit["p_tr_oof"])
            print(f"seed={seed} " + H.fmt(r, "hist")); print(f"seed={seed} " + H.fmt(r, "oof"), flush=True)
    rep = {"arms": {}, "build_s": build_s}
    print("\n=== SUMMARY vs A (same seed/protocol) ===")
    for name in arms:
        rep["arms"][name] = {}
        for proto in ["hist", "oof"]:
            row = []
            for seed in seeds:
                a = res[seed]["A_E009D"][proto]; b = res[seed][name][proto]
                d, lo, hi, pl = H.paired_bootstrap(a["scores"], b["scores"])
                rep["arms"][name][f"s{seed}_{proto}"] = {k: v for k, v in b.items() if k != "scores"} | dict(delta=d, ci_lo=lo, ci_hi=hi)
                row.append(f"s{seed}: {b['macro']*100:.2f} ({d*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]) IN={b['india']*100:.2f} US={b['us']*100:.2f} FP={b['fp']} TP={b['tp']}")
            print(f"{name:<20} {proto:<4} | " + " | ".join(row))
    rep["runtime_s"] = time.time() - t_all
    json.dump(rep, open(os.path.join(OUT, "e011_results.json"), "w"), indent=1)
    pickle.dump(preds, open(os.path.join(OUT, "e011_preds_seed42.pkl"), "wb"))
    print(f"total {rep['runtime_s']:.0f}s")

if __name__ == "__main__":
    main()
