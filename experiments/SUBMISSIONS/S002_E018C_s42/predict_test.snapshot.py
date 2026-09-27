"""
predict_test.py -- end-to-end test inference (retrieval -> features -> two-stage LightGBM -> submission files).

Usage:
  python src/predict_test.py fit   <model_tag>                 # fit final base + stage-2 models on a training feature set
  python src/predict_test.py run   <model_tag> <country> [max_s1] # retrieval + features + predictions for one country
  python src/predict_test.py write <model_tag>                 # merge countries -> matching_results.tsv / candidate_pairs.tsv

Design
- Retrieval: retrieval_engine, RECON-04 configuration, per country (US / India / France hard partition; France is its own
  partition and is never mapped to another country). candidate_pairs = the FULL retrieval pool fed to the model.
- Features: E011-C representation (fixed normalize + transliteration) + test-split corpus statistics (corpus_stats_test.pkl).
  is_india = 1 only for India (France gets 0, same encoding as US -- flagged, see PROGRESS.md).
- Cascade: X22 + base model + competition block on the full pool; expensive blocks + stage 2 only for pairs with
  base prob >= CASCADE_TAU (pool-level features still use the full pool). Pairs below tau are predicted non-match.
- Compliance: candidate_pairs.tsv lists EXACTLY the pairs stage 2 scores (README: "whatever your model actually runs
  inference over"). With CASCADE_TAU=0 that is the full retrieval pool. Predictions are asserted to be a subset.
- STATUS: draft, NOT yet executed on test data.
"""
import os, sys, time, json, pickle, gc, collections
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import build_pools as BP
import pair_features as PF
from recon05_baseline_scorer import normalize
from translit import normalize_fixed

OUT = os.path.join(H.ROOT, "experiments", "TEST_PIPELINE"); os.makedirs(OUT, exist_ok=True)
TEST = os.path.join(H.ROOT, "student_resource", "dataset", "test")
FNORM = lambda x: normalize_fixed(x, translit=True)
CASCADE_TAU = float(os.environ.get("CASCADE_TAU", 0.0))   # 0 = no cascade (stage 2 scores the full pool)
CHUNK = 2000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit(model_tag, seed=42, lam=1.0, n_new=10000, norm="translit", extra=None):
    """Refit E014-B exactly (same data/params/seed) and save base + stage-2 models, threshold and val predictions."""
    g = H.golden
    D = H.load_e008(verbose=False)
    E = os.path.join(H.ROOT, "experiments", "E014")
    C = pickle.load(open(os.path.join(E, f"e014_feats_{n_new}_{norm}.pkl"), "rb"))
    LF_new = np.load(os.path.join(E, f"e014_LFnew_{n_new}_{norm}.npy"), mmap_mode="r")
    y = np.concatenate([D["y_tr"], np.array([m[2] for m in C["meta_new"]], dtype=np.int32)])
    meta = list(D["train_meta"]) + C["meta_new"]; ids = list(D["train_s1_ids"]) + C["new_ids"]
    n = len(y); Xtr = np.empty((n, 99), dtype=np.float32); o = 0
    for p in [C["LF_tr"], LF_new]:
        Xtr[o:o + len(p), :22] = p[:, :22]; Xtr[o:o + len(p), 34:] = p[:, 22:]; o += len(p)
    oof_base, val_base = g.generate_oof_and_val_base_scores(Xtr[:, :22], y, meta, ids, C["LF_va"][:, :22])
    base = lgb.LGBMClassifier(**g.LGB_PARAMS).fit(Xtr[:, :22], y)
    assert np.allclose(base.predict_proba(C["LF_va"][:, :22])[:, 1], val_base), "base model refit mismatch"
    Xtr[:, 22:34] = g.extract_block_a_competition(meta, oof_base)
    Xva = PF.assemble(C["LF_va"], g.extract_block_a_competition(D["val_meta"], val_base))
    if extra == "rerank":   # E018: train column = OOF fold-model scores; val column = deployable full-data reranker
        rr = np.load(os.path.join(H.ROOT, "experiments", "E017_embed", "rerank_top10_full.npy"))
        assert len(rr) == n + len(Xva)
        Xtr = np.hstack([Xtr, rr[:n, None]]).astype(np.float32); Xva = np.hstack([Xva, rr[n:, None]]).astype(np.float32)
    params = dict(H.LGB_PARAMS, reg_lambda=lam)
    fitres = H.fit_stage2(Xtr, y, Xva, meta, ids, seed=seed, params=params)
    s1_all = dict(D["s1_dict"]); s1_all.update({s: dict(v, gt=set(v["gt"])) for s, v in C["s1_new"].items()})
    r = H.evaluate_arm(model_tag, fitres, dict(D, train_meta=meta, train_s1_ids=ids, s1_dict=s1_all))
    log(H.fmt(r, "oof")); log(H.fmt(r, "hist"))
    M = dict(base=base, stage2=fitres["clf"], th=r["oof"]["th"], th_hist=r["hist"]["th"], seed=seed, reg_lambda=lam,
             norm=norm, n_train_s1=len(ids), val_macro_oof=r["oof"]["macro"], val_macro_hist=r["hist"]["macro"],
             lgb_params=params, extra=extra,
             feature_layout="E009-D 99 cols: X22|A12|B7|C6|D6|E3|NUM27|TOK16" + (" + reranker(top10 by base, NaN otherwise)" if extra else ""))
    pickle.dump(M, open(os.path.join(OUT, f"model_{model_tag}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict(val_meta=D["val_meta"], p_va=fitres["p_va"], th=M["th"]),
                open(os.path.join(OUT, f"valpreds_{model_tag}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(f"saved model_{model_tag}.pkl th={M['th']:.2f} val OOF-protocol macro={M['val_macro_oof']*100:.2f}")


def load_test_s1(country, max_s1=None):
    s1 = {}
    with open(os.path.join(TEST, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t"); p += [""] * (4 - len(p))
            if p[3] != country:
                continue
            s1[p[0]] = {"name_r": normalize(p[1]), "addr_r": normalize(p[2]), "name": FNORM(p[1]), "addr": FNORM(p[2]),
                        "country": country}
            if max_s1 and len(s1) >= max_s1:
                break
    return s1


SHARD = 5000
_W = {}


def _init_worker(model_path, stats_path):
    _W["M"] = pickle.load(open(model_path, "rb"))
    _W["stats"] = pickle.load(open(stats_path, "rb"))
    _W["idf"] = PF.make_idf(_W["stats"])


def _score_shard(path):
    """Worker: features + two-stage scoring for one shard payload. Returns preds, scored candidate lists, timings."""
    t0 = time.time()
    d = pickle.load(open(path, "rb"))
    M, stats, idf = _W["M"], _W["stats"], _W["idf"]
    s1, pools, pools_r, freq = d["s1"], d["pools"], d["pools_r"], d["freq"]
    preds, cands_out = {}, {s: [] for s in s1}
    base_top10 = {}
    top2 = {}                                   # diagnostics only: per-S1 (top, second, n_scored) stage-2 scores
    meta = [(s, c, 0) for s in s1 for c in pools.get(s, {})]
    n_kept = 0
    if meta:
        X22 = PF.x22_only(meta, s1, pools, freq, workers=1)
        pb = M["base"].predict_proba(X22)[:, 1]
        A = H.golden.extract_block_a_competition(meta, pb)
        bt = {}                                  # top-10 candidates per S1 by BASE score (reranker input; no stage-2 use)
        for (s, c, _), p in zip(meta, pb):
            bt.setdefault(s, []).append((float(p), c))
        base_top10 = {s: [c for _, c in sorted(L, reverse=True)[:10]] for s, L in bt.items()}
        keep = np.where(pb >= CASCADE_TAU)[0]; n_kept = len(keep)
        if n_kept:
            km = [meta[i] for i in keep]
            R = PF.rest_blocks(km, X22[keep], s1, pools, stats, idf, addr_key_cands=pools_r)
            X = np.hstack([X22[keep], A[keep], R]).astype(np.float32)
            if M.get("extra") == "rerank":
                rrp = path.replace("shard_", "rr_")
                rr = pickle.load(open(rrp, "rb"))            # {s1: {cand: score}} for base top-10 pairs
                col = np.array([rr.get(s, {}).get(c, np.nan) for s, c, _ in km], np.float32)[:, None]
                X = np.hstack([X, col])
            p2 = M["stage2"].predict_proba(X)[:, 1]
            by = {}
            for (s, c, _), p in zip(km, p2):
                by.setdefault(s, []).append(float(p))
            for s, L in by.items():
                L.sort(reverse=True); top2[s] = (L[0], L[1] if len(L) > 1 else 0.0, len(L))
            for (s, c, _), p in zip(km, p2):
                cands_out[s].append(c)          # candidate_pairs = pairs the stage-2 model scores
                if p >= M["th"]:
                    preds.setdefault(s, []).append((c, float(p)))
    for s, v in preds.items():
        assert {c for c, _ in v} <= set(cands_out[s]), s
    import e009_numeric_features as E9
    E9._parse_cached.cache_clear()     # bound per-worker memory across shards (values unaffected)
    import psutil
    rss = psutil.Process().memory_info().rss / 2 ** 20
    return dict(preds=preds, cands=cands_out, top2=top2, base_top10=base_top10, n_pairs=len(meta), rss_mb=rss, n_kept=n_kept, secs=time.time() - t0, path=path)


def _base_shard(path):
    """Worker: X22 + base model only -> per-S1 top-10 candidates by base score (reranker input)."""
    d = pickle.load(open(path, "rb"))
    s1, pools, freq = d["s1"], d["pools"], d["freq"]
    meta = [(s, c, 0) for s in s1 for c in pools.get(s, {})]
    out = {}
    if meta:
        pb = _W["M"]["base"].predict_proba(PF.x22_only(meta, s1, pools, freq, workers=1))[:, 1]
        bt = {}
        for (s, c, _), p in zip(meta, pb):
            bt.setdefault(s, []).append((float(p), c))
        out = {s: [c for _, c in sorted(L, reverse=True)[:10]] for s, L in bt.items()}
    return out


def basetop(model_tag, country, workers=3, tag="full"):
    import multiprocessing as mp
    info = json.load(open(os.path.join(OUT, f"shards_{country}_{tag}", "prepare_info.json")))
    slim = os.path.join(H.SHARED, "corpus_stats_test_slim.pkl"); res = {}; t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                      initargs=(os.path.join(OUT, f"model_{model_tag}.pkl"), slim)) as pool:
        for r in pool.imap_unordered(_base_shard, info["shards"]):
            res.update(r)
    pickle.dump(res, open(os.path.join(OUT, f"basetop10_{model_tag}_{country}_{tag}.pkl"), "wb"))
    log(f"basetop {country}: {len(res):,} S1 in {time.time()-t0:.0f}s")


def prepare(country, max_s1=None):
    """Retrieval + shard payloads for one country (main process)."""
    from translit import has_indic
    stats = pickle.load(open(os.path.join(H.SHARED, "corpus_stats_test.pkl"), "rb"))
    t0 = time.time()
    s1 = load_test_s1(country, max_s1)
    ids = list(s1)
    tag = "full" if not max_s1 else f"n{max_s1}"
    log(f"{country}: {len(ids):,} test S1 ({tag})")
    ppath = os.path.join(BP.POOL_DIR, f"test_{country}_{tag}.pkl")
    P = pickle.load(open(ppath, "rb")) if os.path.exists(ppath) else         BP.retrieve("test", country, ids, [s1[s]["name_r"] for s in ids], [s1[s]["addr_r"] for s in ids], tag)
    t_ret = time.time() - t0
    t1 = time.time()
    need_raw = [c for c, (n, a) in P["text"].items() if has_indic(n) or has_indic(a)]
    raw = BP.fetch_raw("test", need_raw) if need_raw else {}
    t_raw = time.time() - t1
    fnorm_text = {c: ((FNORM(raw[c][0]), FNORM(raw[c][1])) if c in raw else (n, a)) for c, (n, a) in P["text"].items()}
    sdir = os.path.join(OUT, f"shards_{country}_{tag}"); os.makedirs(sdir, exist_ok=True)
    paths = []
    for k in range(0, len(ids), SHARD):
        sub = ids[k:k + SHARD]
        pools_r = BP.to_pool_dicts(P, subset=set(sub))
        pools = {s: {c: dict(ci, name=fnorm_text[c][0], addr=fnorm_text[c][1]) for c, ci in pool.items()} for s, pool in pools_r.items()}
        payload = dict(s1={s: {k2: s1[s][k2] for k2 in ("name", "addr", "country")} for s in sub},
                       pools=pools, pools_r=pools_r, freq={s: stats["name_freq"].get(s1[s]["name_r"], 1) for s in sub})
        path = os.path.join(sdir, f"shard_{k // SHARD:04d}.pkl")
        pickle.dump(payload, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL); paths.append(path)
    info = dict(country=country, tag=tag, n_s1=len(ids), n_indic_raw=len(need_raw), t_ret=t_ret, t_raw=t_raw,
                t_prep=time.time() - t0, retrieval_timings=P["timings"], shards=paths)
    json.dump(info, open(os.path.join(sdir, "prepare_info.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in info.items() if k not in ("shards", "retrieval_timings")}))
    return info


def run(model_tag, country, max_s1=None, workers=3):
    import multiprocessing as mp
    tag = "full" if not max_s1 else f"n{max_s1}"
    sdir = os.path.join(OUT, f"shards_{country}_{tag}")
    info = json.load(open(os.path.join(sdir, "prepare_info.json"))) if os.path.exists(os.path.join(sdir, "prepare_info.json"))         else prepare(country, max_s1)
    slim = os.path.join(H.SHARED, "corpus_stats_test_slim.pkl")
    if not os.path.exists(slim):
        st = pickle.load(open(os.path.join(H.SHARED, "corpus_stats_test.pkl"), "rb")); st.pop("name_freq")
        pickle.dump(st, open(slim, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    t0 = time.time()
    preds, cands, top2, btop = {}, {}, {}, {}; n_pairs = n_kept = 0
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                      initargs=(os.path.join(OUT, f"model_{model_tag}.pkl"), slim)) as pool:
        for i, r in enumerate(pool.imap_unordered(_score_shard, info["shards"])):
            preds.update(r["preds"])
            cands.update({k: len(v) for k, v in r["cands"].items()} if os.environ.get("SLIM_CANDS") == "1" else r["cands"])
            top2.update(r.get("top2", {})); btop.update(r.get("base_top10", {})); n_pairs += r["n_pairs"]; n_kept += r["n_kept"]
            el = time.time() - t0
            log(f"  shard {i + 1}/{len(info['shards'])} ({r['secs']:.0f}s, rss {r.get('rss_mb', 0):.0f}MB) | pairs {n_pairs:,} | {el:.0f}s | {n_pairs / el:.0f} pairs/s")
    res = dict(country=country, tag=tag, n_s1=info["n_s1"], n_pairs=n_pairs, n_kept=n_kept, t_score=time.time() - t0,
               t_prep=info["t_prep"], workers=workers, tau=CASCADE_TAU, n_pred_nonempty=len(preds),
               n_pred_pairs=sum(len(v) for v in preds.values()))
    pickle.dump(dict(preds=preds, cands=cands, top2=top2, base_top10=btop, info=res), open(os.path.join(OUT, f"pred_{model_tag}_{country}_{tag}.pkl"), "wb"))
    log(json.dumps(res))
    return res


def sample_stats(model_tag, country, n_shards=12, workers=3, tag="full"):
    """Diagnostics only: re-score a random sample of prepared shards to get per-S1 top-2 stage-2 scores."""
    import multiprocessing as mp, random
    info = json.load(open(os.path.join(OUT, f"shards_{country}_{tag}", "prepare_info.json")))
    paths = random.Random(0).sample(info["shards"], min(n_shards, len(info["shards"])))
    slim = os.path.join(H.SHARED, "corpus_stats_test_slim.pkl")
    out = dict(preds={}, cands={}, top2={})
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                      initargs=(os.path.join(OUT, f"model_{model_tag}.pkl"), slim)) as pool:
        for r in pool.imap_unordered(_score_shard, paths):
            for k in out: out[k].update(r[k])
    pickle.dump(dict(out, shards=paths), open(os.path.join(OUT, f"sample_{model_tag}_{country}.pkl"), "wb"))
    log(f"sampled {len(paths)} shards, {len(out['cands']):,} S1")


def _cands_from_shards(country, tag="full"):
    """Rebuild the exact scored candidate lists (CASCADE_TAU=0 -> full pool, pool-dict order) from shard payloads."""
    info = json.load(open(os.path.join(OUT, f"shards_{country}_{tag}", "prepare_info.json")))
    for sp in info["shards"]:
        d = pickle.load(open(sp, "rb"))
        for s in d["s1"]:
            yield s, list(d["pools"].get(s, {}))


def write2(model_tag, tag="full"):
    """Country-by-country streaming writer (bounded memory). Same output format/rules as write()."""
    assert CASCADE_TAU == 0.0, "write2 rebuilds candidate lists assuming the full pool was scored"
    TAB, NL = chr(9), chr(10)
    ctry = {}
    with open(os.path.join(TEST, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]
    sub = os.path.join(OUT, f"submission_{model_tag}"); os.makedirs(sub, exist_ok=True)
    seen = 0
    with open(os.path.join(sub, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fm,          open(os.path.join(sub, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id" + TAB + "matched_entity_ids" + NL)
        fc.write("source1_entity_id" + TAB + "candidate_entity_ids" + NL)
        for c in ["US", "India", "France"]:
            d = pickle.load(open(os.path.join(OUT, f"pred_{model_tag}_{c}_{tag}.pkl"), "rb"))
            preds = {s: [x for x, _ in v] for s, v in d["preds"].items()}; cands = d["cands"]
            slim = isinstance(next(iter(cands.values())), int)
            it = _cands_from_shards(c, tag) if slim else ((s, v) for s, v in cands.items())
            n_c = 0
            for s, cl in it:
                if slim:
                    assert len(cl) == cands[s], (s, len(cl), cands[s])
                pl = preds.get(s, [])
                assert set(pl) <= set(cl), s
                fm.write(s + TAB + ",".join(pl) + NL); fc.write(s + TAB + ",".join(cl) + NL); n_c += 1
            assert n_c == sum(1 for v in ctry.values() if v == c), (c, n_c)
            seen += n_c; log(f"  wrote {c}: {n_c:,} S1 (slim={slim})")
            del d, preds, cands
    assert seen == len(ctry), (seen, len(ctry))
    log(f"wrote {sub} ({seen:,} rows)")


def write(model_tag, tag="full"):
    all_ids = []
    with open(os.path.join(TEST, "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            all_ids.append(line.split("\t", 1)[0])
    preds, cands = {}, {}
    for ctry in ["US", "India", "France"]:
        d = pickle.load(open(os.path.join(OUT, f"pred_{model_tag}_{ctry}_{tag}.pkl"), "rb"))
        for s, v in d["preds"].items():
            preds[s] = [c for c, _ in v]
        cands.update(d["cands"])
    missing = [s for s in all_ids if s not in cands]
    assert not missing, f"{len(missing)} test S1 without candidate rows"
    sub = os.path.join(OUT, f"submission_{model_tag}"); os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, "matching_results.tsv"), "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in all_ids:
            f.write(f"{s}\t{','.join(preds.get(s, []))}\n")
    with open(os.path.join(sub, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in all_ids:
            f.write(f"{s}\t{','.join(cands[s])}\n")
    log(f"wrote {sub}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2], extra=(sys.argv[3] if len(sys.argv) > 3 else None))
    elif cmd == "basetop":
        basetop(sys.argv[2], sys.argv[3], int(os.environ.get("WORKERS", 3)))
    elif cmd == "sample":
        sample_stats(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)
    elif cmd == "prepare":
        prepare(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "0" else None)
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "0" else None,
            int(os.environ.get("WORKERS", 3)))
    elif cmd == "write2":
        write2(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "full")
    elif cmd == "write":
        write(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "full")
