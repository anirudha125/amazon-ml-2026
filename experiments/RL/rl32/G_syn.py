"""RL-32 G-SYN: country-agnostic synthetic hard negatives for an UNSEEN country, tested in the labelled LOCO lab (US -> India).
Mechanism under test: a matcher that never saw a country produces decoy false positives there (LOCO: 14x FP, 86-93% decoys no S1 owns).
The generator makes decoys as one-field perturbations of an S1 (RL_REPORT sec. 2). So synthesise, from the TARGET country's own S1
reference records (no labels), (a) noisy positive copies and (b) one-field decoys, with the vocabulary (legal tails, common words, name-core
index) MINED AUTOMATICALLY from that country's S1 names. No hand-built country rules. Then fine-tune the US-only reranker on synthetic target
pairs + a replay of true-labelled US TR pairs, and evaluate on V1 India (frozen LOCO_US stage 2, and a US-rows stage-2 refit).
Target S1 set for the lab: the India S1 of T (their labels are never read). Evaluation: V1 India, all and Latin-script subset.
Operators (per S1, rng-seeded): positives x3 = name noise (case, legal-tail drop/swap/paren/move, word drop/add/swap/dup, typo/OCR,
acronym, junk prefix, domain) + address noise (empty 4.5%, case, component drop/reorder, truncation abbreviation, number format, typo);
decoys = D1 house-number change (+-1..3 / one digit / random), D2 one content word swapped for a mined common word, D3 fake name
(another S1's name) at the same address, D4 same name at another S1's address, D5 a noisy copy of a DIFFERENT S1 with the same name core.
Usage: taskset -c 14-21 nice -n 10 python G_syn.py [TAG]
"""
import os, sys, json, pickle, time, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
E26 = os.path.join(ROOT, "experiments", "E026_rrL")
sys.path.insert(0, HERE); sys.path.insert(0, E26); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("LGB_THREADS", "8")


def mine_vocab(names):
    """legal tails (unigram / bigram), common words, name-core -> S1 index; all from one country's S1 names."""
    toks = [n.split() for n in names]
    low = [[t.lower() for t in x] for x in toks]
    n = len(names)
    last1 = collections.Counter(x[-1] for x in low if x); last2 = collections.Counter(" ".join(x[-2:]) for x in low if len(x) > 1)
    legal1 = [t for t, c in last1.most_common(40) if c / n > 0.002]
    legal2 = [t for t, c in last2.most_common(20) if c / n > 0.002 and all(w in legal1 or len(w) <= 7 for w in t.split())]
    allw = collections.Counter(t for x in low for t in x)
    common = [t for t, c in allw.most_common(400) if t not in legal1 and t.isalpha() and len(t) >= 3][:250]
    return dict(legal1=legal1, legal2=legal2, common=common)


def core_of(name, V):
    x = name.lower().split()
    while x and (x[-1] in V["legal1"]):
        x = x[:-1]
    return " ".join(x)


OCR = {"o": "0", "l": "1", "s": "5", "e": "3", "i": "1", "a": "4", "b": "8", "g": "6"}


def typo(w, rng):
    if len(w) < 4:
        return w
    k = int(rng.integers(0, len(w))); r = rng.random()
    if r < 0.25 and w[k].lower() in OCR:
        return w[:k] + OCR[w[k].lower()] + w[k + 1:]
    if r < 0.45:
        return w[:k] + w[k + 1:]
    if r < 0.65:
        return w[:k] + chr(97 + int(rng.integers(0, 26))) + w[k:]
    if r < 0.85 and k < len(w) - 1:
        return w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return w[:k] + w[k] + w[k:]


def case(s, rng):
    r = rng.random()
    return s.upper() if r < 0.3 else s.lower() if r < 0.4 else s.title() if r < 0.6 else s


def noisy_name(name, V, rng):
    w = name.split()
    if not w:
        return name
    low = [t.lower() for t in w]
    tail = 2 if len(w) > 2 and " ".join(low[-2:]) in V["legal2"] else 1 if low[-1] in V["legal1"] else 0
    body, leg = w[:len(w) - tail], w[len(w) - tail:]
    if leg:
        r = rng.random()
        if r < 0.25:
            leg = []
        elif r < 0.40:
            alt = V["legal2"] + V["legal1"]; leg = alt[int(rng.integers(0, len(alt)))].split()
        elif r < 0.45:
            leg = ["(" + " ".join(leg) + ")"]
        elif r < 0.50:
            body, leg = leg + body, []
    if len(body) >= 3 and rng.random() < 0.12:
        body.pop(int(rng.integers(0, len(body))))
    if rng.random() < 0.12:
        cw = V["common"][int(rng.integers(0, len(V["common"])))].title()
        body = body + [cw] if rng.random() < 0.6 else [cw] + body
    if len(body) >= 2 and rng.random() < 0.08:
        k = int(rng.integers(0, len(body) - 1)); body[k], body[k + 1] = body[k + 1], body[k]
    if body and rng.random() < 0.03:
        k = int(rng.integers(0, len(body))); body.insert(k, body[k])
    if body and rng.random() < 0.25:
        k = int(rng.integers(0, len(body))); body[k] = typo(body[k], rng)
    out = " ".join(body + leg)
    r = rng.random()
    if r < 0.03 and len(body) >= 2:
        out = "".join(t[0] for t in body if t).upper()
    elif r < 0.05:
        out = ("*** " if rng.random() < 0.5 else "-- ") + out
    elif r < 0.07:
        out = re.sub(r"[^a-z0-9]", "", " ".join(body).lower()) + ".com"
    return case(out, rng)


def noisy_addr(addr, rng):
    if not addr or rng.random() < 0.045:
        return ""
    comp = [c.strip() for c in addr.split(",") if c.strip()]
    if len(comp) >= 3 and rng.random() < 0.3:
        comp.pop(int(rng.integers(0, len(comp))))
    if len(comp) >= 2 and rng.random() < 0.25:
        comp = [comp[i] for i in rng.permutation(len(comp))]
    s = ", ".join(comp)
    w = s.split(" ")
    if rng.random() < 0.25:
        long_ = [i for i, t in enumerate(w) if len(t) >= 5 and t.isalpha()]
        if long_:
            i = long_[int(rng.integers(0, len(long_)))]; k = int(rng.integers(2, 5)); w[i] = w[i][:k] + ("." if rng.random() < 0.5 else "")
    s = " ".join(w)
    if rng.random() < 0.15:
        m = re.search(r"\d+", s)
        if m:
            f = ["No. {}", "#{}", "({})", "0{}", "N {}"][int(rng.integers(0, 5))]
            s = s[:m.start()] + f.format(m.group(0)) + s[m.end():]
    if rng.random() < 0.15:
        w = s.split(" "); k = int(rng.integers(0, len(w))); w[k] = typo(w[k], rng); s = " ".join(w)
    return case(s, rng)


def decoys(i, names, addrs, V, core_idx, cores, rng):
    n, a = names[i], addrs[i]; out = []
    m = re.search(r"\d+", a or "")
    if m:                                                    # D1 house-number change
        v = int(m.group(0)); r = rng.random()
        if r < 0.4:
            nv = max(0, v + int(rng.choice([-3, -2, -1, 1, 2, 3])))
        elif r < 0.7:
            s = list(m.group(0)); k = int(rng.integers(0, len(s))); s[k] = str((int(s[k]) + int(rng.integers(1, 10))) % 10); nv = int("".join(s))
        else:
            nv = int(rng.integers(1, 2 * v + 10))
        if nv != v:
            out.append((n, a[:m.start()] + str(nv) + a[m.end():]))
    w = n.split(); body = [k for k, t in enumerate(w) if t.lower() not in V["legal1"]]
    if body:                                                 # D2 one content word swapped
        k = body[int(rng.integers(0, len(body)))]; cw = V["common"][int(rng.integers(0, len(V["common"])))].title()
        if cw.lower() != w[k].lower():
            w2 = list(w); w2[k] = cw; out.append((" ".join(w2), a))
    j = int(rng.integers(0, len(names)))                     # D3 fake name at the same address
    if j != i and a:
        out.append((names[j], a))
    j = int(rng.integers(0, len(names)))                     # D4 same name, another S1's address
    if j != i and addrs[j]:
        out.append((n, addrs[j]))
    sib = [k for k in core_idx.get(cores[i], []) if k != i]  # D5 a different S1 with the same name core
    if sib:
        k = sib[int(rng.integers(0, len(sib)))]; out.append((names[k], addrs[k]))
    return out


def synth(target_ids, country, s1df, seed=3206, n_pos=3):
    """synthetic (S1 text, record text, y) triples for the target S1 ids; vocabulary from ALL S1 of that country."""
    import e023_rerank as RR
    c = s1df[s1df.country == country]; names = c.name.fillna("").tolist(); addrs = c.addr.fillna("").tolist(); ids = c.id.tolist()
    V = mine_vocab(names); cores = [core_of(x, V) for x in names]
    core_idx = collections.defaultdict(list)
    for k, cc in enumerate(cores):
        core_idx[cc].append(k)
    pos_of = {s: k for k, s in enumerate(ids)}; rng = np.random.default_rng(seed)
    A, B, y = [], [], []
    for s in target_ids:
        i = pos_of[s]; q = RR.fmt(names[i], addrs[i])
        for _ in range(n_pos):
            A.append(q); B.append(RR.fmt(noisy_name(names[i], V, rng), noisy_addr(addrs[i], rng))); y.append(1.0)
        for dn, da in decoys(i, names, addrs, V, core_idx, cores, rng):
            A.append(q); B.append(RR.fmt(noisy_name(dn, V, rng), noisy_addr(da, rng))); y.append(0.0)
    return A, B, np.asarray(y, np.float32), dict(V_legal1=V["legal1"][:15], V_legal2=V["legal2"][:8], V_common=V["common"][:20],
                                               n_target=len(target_ids), n_pairs=len(y), pos_share=float(np.mean(y)))


def main(tag="US_SYN"):
    import G_loco_s2 as GS
    import rrL_lib as L
    import e023_rerank as RR
    import lightgbm as lgb
    from boot import S2
    log = L.log; t0 = time.time()
    MDIR = os.path.join(HERE, f"G_model_{tag}"); OUT = os.path.join(HERE, f"G_rr_{tag}.pkl"); RES = os.path.join(HERE, f"G_syn_{tag}_results.json")
    tr = pickle.load(open(os.path.join(ROOT, "experiments", "RL", "cache", "train.pkl"), "rb")); s1df = tr["s1"]
    T, V = GS.load(); c1 = GS.s1_country(T)
    target = [str(s) for s in T["s1_ids"][c1 == "India"]]
    A, B, y, sinfo = synth(target, "India", s1df); log("synthetic", json.dumps(sinfo))
    for k in range(8):
        log("  example", int(y[k * 97 % len(y)]), A[k * 97 % len(y)][:70], "->", B[k * 97 % len(y)][:70])
    trp = pickle.load(open(os.path.join(E26, "tr_pairs.pkl"), "rb"))
    s1c = s1df.set_index("id")["country"]
    us_idx = np.flatnonzero(s1c.reindex([a for a, _ in trp["pairs"]]).values == "US")
    rep = np.random.default_rng(3207).choice(us_idx, 200000, replace=False)
    top = pickle.load(open(os.path.join(E26, "cache", "top10_pairs.pkl"), "rb"))["all"]
    tx = RR.load_texts({x for i in rep for x in trp["pairs"][i]} | {x for p in top for x in p})
    AA = A + [tx[trp["pairs"][i][0]] for i in rep]; BB = B + [tx[trp["pairs"][i][1]] for i in rep]
    yy = np.concatenate([y, np.asarray(trp["y"], np.float32)[rep]])
    if not os.path.exists(os.path.join(MDIR, "train_info.json")):
        L.train(AA, BB, yy, MDIR, init=os.path.join(HERE, "G_model_US"), rev=None, bs=256, micro=256, lr=2e-5, epochs=1.0, seed=42, n_tok=3)
    if not os.path.exists(OUT):
        sc, si = L.score(MDIR, [tx[a] for a, _ in top], [tx[b] for _, b in top], bs=1024, n_tok=5)
        pickle.dump(dict(zip(top, sc.tolist())), open(OUT, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    rr = pickle.load(open(OUT, "rb"))
    M = pickle.load(open(os.path.join(HERE, "G_s2_LOCO_US.pkl"), "rb")); clf, th = M["stage2"], M["th"]
    cv = GS.s1_country(V); res = dict(synthetic=sinfo)
    pv = clf.predict_proba(GS.assemble(V, rr))[:, 1]; np.save(os.path.join(HERE, f"G_p_{tag}_frozen_V1.npy"), pv.astype(np.float32))
    res["frozen_India"] = GS.calib(V, pv, th, cv == "India"); res["frozen_US"] = GS.calib(V, pv, th, cv == "US")
    keep_s1 = c1 == "US"; rows = np.flatnonzero(keep_s1[T["s1idx"]]); X = GS.assemble(T, rr)[rows]; yt = T["y"][rows].astype(np.int32)
    s1map = -np.ones(len(c1), np.int64); s1map[keep_s1] = np.arange(int(keep_s1.sum())); sidx = s1map[T["s1idx"][rows]]
    P = S2.lgbm(42); P["n_jobs"] = int(os.environ["LGB_THREADS"])
    clf2 = lgb.LGBMClassifier(**P).fit(X, yt); poof = np.zeros(len(yt)); f = np.random.default_rng(42).permutation(int(keep_s1.sum())) % 3
    for k in range(3):
        va = f[sidx] == k; poof[va] = lgb.LGBMClassifier(**P).fit(X[~va], yt[~va]).predict_proba(X[va])[:, 1]
    th2, _ = S2.best_th(sidx, yt, poof, T["n_gt"][keep_s1]); pv2 = clf2.predict_proba(GS.assemble(V, rr))[:, 1]
    np.save(os.path.join(HERE, f"G_p_{tag}_refit_V1.npy"), pv2.astype(np.float32))
    res["refit_India"] = GS.calib(V, pv2, th2, cv == "India"); res["refit_US"] = GS.calib(V, pv2, th2, cv == "US")
    res["secs"] = round(time.time() - t0, 1); json.dump(res, open(RES, "w"), indent=1); log(json.dumps(res))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
