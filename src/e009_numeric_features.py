"""
E009 -- Numeric address features + token asymmetry, attribution on the FROZEN E008 pool.

Arms (stage-2 LightGBM only; base model / competition block untouched):
  A  E008 (56 feats)
  B  E008 + NUM block
  C  E008 + TOK block
  D  E008 + NUM + TOK
Each arm: historical in-sample threshold protocol + corrected OOF threshold protocol,
paired bootstrap vs A over the 2,001 val S1 entities. Seed 42 is primary; seeds 43/44
reported as a robustness check only.
"""
import os, sys, re, math, time, json, pickle, collections, unicodedata
import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

OUT = os.path.join(H.ROOT, "experiments", "E009")
os.makedirs(OUT, exist_ok=True)
FEAT_CACHE = os.path.join(OUT, "e009_features.pkl")

UNIT_WORDS = {"unit", "apt", "apartment", "suite", "ste", "fl", "floor", "room", "rm", "bldg",
              "building", "no", "shop", "flat", "plot", "office", "door", "h", "hno", "sy", "sector",
              "block", "blk", "gala", "wing", "lot", "box", "po", "pmb", "dept", "level", "lvl"}

_DIGIT_MAP = {}
def ascii_digits(s):
    """Map any Unicode decimal digit (Devanagari, Bengali, ...) to ASCII."""
    out = []
    for ch in s:
        if ch.isdigit() and not ch.isascii():
            d = _DIGIT_MAP.get(ch)
            if d is None:
                try:
                    d = str(unicodedata.digit(ch))
                except (TypeError, ValueError):
                    d = ch
                _DIGIT_MAP[ch] = d
            out.append(d)
        else:
            out.append(ch)
    return "".join(out)

DIG = re.compile(r"\d+")

import functools

@functools.lru_cache(maxsize=2_000_000)
def _parse_cached(addr):
    return tuple(_parse_numbers(addr))

def parse_numbers(addr):
    """Returns list of (digits, is_pure_token, prev_token) in order (memoized per address string)."""
    return list(_parse_cached(addr)) if addr else []

def _parse_numbers(addr):
    if not addr or addr == "null":
        return []
    toks = ascii_digits(addr).split()
    res = []
    for i, t in enumerate(toks):
        for m in DIG.findall(t):
            res.append((m.lstrip("0") or "0", t.isdigit(), toks[i - 1] if i > 0 else ""))
    return res

def house_number(parsed):
    for d, pure, prev in parsed:
        if pure and prev not in UNIT_WORDS:
            return d
    return None

def longest(parsed):
    if not parsed:
        return None
    best = parsed[0][0]
    for d, _, _ in parsed[1:]:
        if len(d) > len(best):
            best = d
    return best

NUM_NAMES = ["s1_n_num", "c_n_num", "lnum_len", "lnum_status", "lnum_exact", "lnum_cand_nonum",
             "lnum_conflict", "lnum_absdiff_log", "lnum_reldiff", "lnum_edit", "lnum_closest_same_len",
             "lnum_transposition", "first_num_agree", "house_agree", "house_absdiff_log",
             "n_shared_nums", "n_s1_unmatched_nums", "n_c_unmatched_nums", "num_jaccard",
             "all_s1_nums_matched", "max_shared_len", "name_num_s1", "name_num_c", "name_num_shared",
             "name_num_c_only", "small_offset_same_len", "lnum_pool_frac"]

def num_block(s1_addr, s1_name, c_addr, c_name, s1p, pool_frac):
    c_missing = (not c_addr) or c_addr == "null"
    cp = parse_numbers(c_addr)
    A = [d for d, _, _ in s1p]; B = [d for d, _, _ in cp]
    As, Bs = set(A), set(B)
    L = longest(s1p)
    f = dict.fromkeys(NUM_NAMES, -1.0)
    f["s1_n_num"] = len(A); f["c_n_num"] = len(B)
    f["lnum_len"] = len(L) if L else 0
    f["name_num_s1"] = len(DIG.findall(ascii_digits(s1_name)))
    cn = set(DIG.findall(ascii_digits(c_name)))
    sn = set(DIG.findall(ascii_digits(s1_name)))
    f["name_num_c"] = len(cn); f["name_num_shared"] = len(cn & sn)
    f["name_num_c_only"] = len(cn - sn - As)
    f["lnum_pool_frac"] = pool_frac
    if L is None:
        f["lnum_status"] = 0
    elif c_missing:
        f["lnum_status"] = 1
    elif not B:
        f["lnum_status"] = 2; f["lnum_cand_nonum"] = 1.0
    else:
        f["lnum_cand_nonum"] = 0.0
        if L in Bs:
            f["lnum_status"] = 3; f["lnum_exact"] = 1.0; f["lnum_conflict"] = 0.0
        elif any((L in b) or (b in L) for b in Bs):
            f["lnum_status"] = 4; f["lnum_exact"] = 0.0; f["lnum_conflict"] = 0.0
        else:
            f["lnum_status"] = 5; f["lnum_exact"] = 0.0; f["lnum_conflict"] = 1.0
        a = int(L[:15])
        diffs = [(abs(a - int(b[:15])), b) for b in Bs]
        md, mb = min(diffs)
        f["lnum_absdiff_log"] = math.log1p(md)
        f["lnum_reldiff"] = md / max(a, int(mb[:15]), 1)
        edits = [(Levenshtein.distance(L, b), b) for b in Bs]
        me, eb = min(edits)
        f["lnum_edit"] = me
        f["lnum_closest_same_len"] = 1.0 if len(eb) == len(L) else 0.0
        f["lnum_transposition"] = 1.0 if any(b != L and sorted(b) == sorted(L) for b in Bs) else 0.0
        f["small_offset_same_len"] = 1.0 if any(0 < abs(a - int(b[:15])) <= 50 and len(b) == len(L) for b in Bs) else 0.0
    if A and B:
        f["first_num_agree"] = 1.0 if A[0] == B[0] else 0.0
        inter = As & Bs
        f["n_shared_nums"] = len(inter); f["n_s1_unmatched_nums"] = len(As - Bs)
        f["n_c_unmatched_nums"] = len(Bs - As); f["num_jaccard"] = len(inter) / len(As | Bs)
        f["all_s1_nums_matched"] = 1.0 if As <= Bs else 0.0
        f["max_shared_len"] = max((len(x) for x in inter), default=0)
    elif not c_missing:
        f["n_shared_nums"] = 0; f["n_s1_unmatched_nums"] = len(As); f["n_c_unmatched_nums"] = len(Bs)
    h1, h2 = house_number(s1p), house_number(cp)
    if h1 is not None and h2 is not None:
        f["house_agree"] = 1.0 if h1 == h2 else 0.0
        f["house_absdiff_log"] = math.log1p(abs(int(h1[:15]) - int(h2[:15])))
    return [float(f[k]) for k in NUM_NAMES]

TOK_NAMES = ["n_s1_only", "n_c_only", "idf_s1_only_sum", "idf_c_only_sum", "idf_c_only_max",
             "idf_s1_only_max", "frac_idf_s1_only", "frac_idf_c_only", "n_s1_only_soft",
             "n_c_only_soft", "idf_s1_only_soft", "idf_c_only_soft", "addr_n_s1_only",
             "addr_n_c_only", "addr_frac_s1_only", "addr_frac_c_only"]

def _soft_unmatched(src, other, other_joined):
    out = []
    for t in src:
        if t in other:
            continue
        if len(t) >= 3 and t in other_joined:
            continue
        if any(JaroWinkler.similarity(t, o) >= 0.88 for o in other):
            continue
        out.append(t)
    return out

def tok_block(s1_name, c_name, s1_addr, c_addr, idf):
    T1, T2 = set(s1_name.split()), set(c_name.split())
    s1o, co = T1 - T2, T2 - T1
    i1 = [idf(t) for t in s1o]; i2 = [idf(t) for t in co]
    tot1 = sum(idf(t) for t in T1) or 1e-5; tot2 = sum(idf(t) for t in T2) or 1e-5
    s1o_s = _soft_unmatched(s1o, T2, c_name.replace(" ", ""))
    co_s = _soft_unmatched(co, T1, s1_name.replace(" ", ""))
    r = [len(s1o), len(co), sum(i1), sum(i2), max(i2, default=0.0), max(i1, default=0.0),
         sum(i1) / tot1, sum(i2) / tot2, len(s1o_s), len(co_s),
         sum(idf(t) for t in s1o_s), sum(idf(t) for t in co_s)]
    if s1_addr and c_addr and c_addr != "null":
        A1 = {t for t in s1_addr.split() if not any(ch.isdigit() for ch in t)}
        A2 = {t for t in c_addr.split() if not any(ch.isdigit() for ch in t)}
        r += [len(A1 - A2), len(A2 - A1), len(A1 - A2) / max(len(A1), 1), len(A2 - A1) / max(len(A2), 1)]
    else:
        r += [-1.0, -1.0, -1.0, -1.0]
    return r

def build_blocks(meta, D, idf):
    s1d, C = D["s1_dict"], D["cands"]
    # per-S1 cache: parsed numbers + pool fraction of candidates containing S1's longest number
    s1_cache = {}
    for s in {m[0] for m in meta}:
        p = parse_numbers(s1d[s]["addr"]); L = longest(p)
        if L is None:
            frac = -1.0
        else:
            pool = C[s]
            hits = sum(1 for ci in pool.values() if L in {d for d, _, _ in parse_numbers(ci["addr"])})
            frac = hits / max(len(pool), 1)
        s1_cache[s] = (p, frac)
    NUM = np.zeros((len(meta), len(NUM_NAMES)), dtype=np.float32)
    TOK = np.zeros((len(meta), len(TOK_NAMES)), dtype=np.float32)
    for i, (s, c, _) in enumerate(meta):
        v = s1d[s]; ci = C[s][c]
        p, frac = s1_cache[s]
        NUM[i] = num_block(v["addr"], v["name"], ci["addr"], ci["name"], p, frac)
        TOK[i] = tok_block(v["name"], ci["name"], v["addr"], ci["addr"], idf)
    return NUM, TOK

def get_features(D):
    if os.path.exists(FEAT_CACHE):
        with open(FEAT_CACHE, "rb") as f:
            return pickle.load(f)
    with open(H.golden.TOKEN_IDF_FILE, "rb") as f:
        tid = pickle.load(f)
    df, N = tid["df"], tid["N"]
    memo = {}
    def idf(w):
        v = memo.get(w)
        if v is None:
            d = df.get(w, 0); v = math.log((N - d + 0.5) / (d + 0.5) + 1.0); memo[w] = v
        return v
    t0 = time.time()
    NUM_tr, TOK_tr = build_blocks(D["train_meta"], D, idf)
    NUM_va, TOK_va = build_blocks(D["val_meta"], D, idf)
    out = dict(NUM_tr=NUM_tr, TOK_tr=TOK_tr, NUM_va=NUM_va, TOK_va=TOK_va, build_s=time.time() - t0)
    with open(FEAT_CACHE, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[E009] features built in {out['build_s']:.1f}s")
    return out

def main():
    t_all = time.time()
    D = H.load_e008()
    F = get_features(D)
    print(f"[E009] feature build time (cached or fresh): {F['build_s']:.1f}s")
    arms = {
        "A_E008": (D["X_tr"], D["X_va"]),
        "B_num": (np.hstack([D["X_tr"], F["NUM_tr"]]), np.hstack([D["X_va"], F["NUM_va"]])),
        "C_tok": (np.hstack([D["X_tr"], F["TOK_tr"]]), np.hstack([D["X_va"], F["TOK_va"]])),
        "D_num_tok": (np.hstack([D["X_tr"], F["NUM_tr"], F["TOK_tr"]]),
                      np.hstack([D["X_va"], F["NUM_va"], F["TOK_va"]])),
    }
    seeds = [42, 43, 44]
    results = collections.defaultdict(dict)
    preds = {}
    for seed in seeds:
        for name, (Xtr, Xva) in arms.items():
            fit = H.fit_stage2(Xtr, D["y_tr"], Xva, D["train_meta"], D["train_s1_ids"], seed=seed)
            r = H.evaluate_arm(name, fit, D)
            results[seed][name] = r
            if seed == 42:
                preds[name] = dict(p_va=fit["p_va"], p_tr_in=fit["p_tr_in"], p_tr_oof=fit["p_tr_oof"])
                if name != "A_E008":
                    imp = fit["clf"].booster_.feature_importance("gain")
                    names = list(range(56)) + (NUM_NAMES if "num" in name else []) + (TOK_NAMES if "tok" in name else [])
                    top = sorted(zip(names[56:], imp[56:]), key=lambda x: -x[1])[:10]
                    r["top_new_gain"] = [(n, float(g)) for n, g in top]
                    r["new_gain_share"] = float(imp[56:].sum() / imp.sum())
            print(f"seed={seed} " + H.fmt(r, "hist")); print(f"seed={seed} " + H.fmt(r, "oof"), flush=True)

    # paired bootstrap vs A for each seed/protocol
    report = {"seeds": seeds, "arms": {}}
    for name in arms:
        report["arms"][name] = {}
        for seed in seeds:
            for proto in ["hist", "oof"]:
                a = results[seed]["A_E008"][proto]; b = results[seed][name][proto]
                d, lo, hi, p = H.paired_bootstrap(a["scores"], b["scores"])
                key = f"s{seed}_{proto}"
                report["arms"][name][key] = dict(
                    th=b["th"], macro=b["macro"], us=b["us"], india=b["india"], singleton=b["singleton"],
                    tp=b["tp"], fp=b["fp"], precision=b["precision"], cand_recall=b["cand_recall"],
                    e2e_recall=b["e2e_recall"], sing_fp_entities=b["sing_fp_entities"],
                    delta=d, ci_lo=lo, ci_hi=hi, p_le0=p)
        r42 = results[42][name]
        report["arms"][name]["top_new_gain"] = r42.get("top_new_gain")
        report["arms"][name]["new_gain_share"] = r42.get("new_gain_share")
        report["arms"][name]["fit_s"] = r42["fit_s"]
    report["total_runtime_s"] = time.time() - t_all
    with open(os.path.join(OUT, "e009_results.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(OUT, "e009_preds_seed42.pkl"), "wb") as f:
        pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n=== SUMMARY (delta vs A, paired bootstrap 95% CI, pp) ===")
    for name in arms:
        for proto in ["hist", "oof"]:
            row = []
            for seed in seeds:
                m = report["arms"][name][f"s{seed}_{proto}"]
                row.append(f"s{seed}: {m['macro']*100:.2f} ({m['delta']*100:+.2f} [{m['ci_lo']*100:+.2f},{m['ci_hi']*100:+.2f}]) FP={m['fp']} TP={m['tp']}")
            print(f"{name:<10} {proto:<4} | " + " | ".join(row))
    print(f"total runtime {report['total_runtime_s']:.0f}s")

if __name__ == "__main__":
    main()
