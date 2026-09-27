"""
AN03 -- Retrieval-miss audit + oracle simulation for candidate retrieval changes (label use: diagnosis only).

For every ground-truth record of the 3,995 RECON sample S1 (train+val), per (country, source) index:
  addr  : current n4 name+addr index, deep rank (top 1000)          [current config uses top 50]
  name  : current n4 name-only index, deep rank (top 1000)          [current config uses top 10]
  tname : n4 index on transliterated names (normalize_fixed+translit), top 50   [proposed]
  aonly : n4 index on address only, top 50                           [proposed]
Also per gt record: script, address missingness, name / address similarity to the S1.
Outputs experiments/AN03/an03_ranks.pkl (+ summary printed). Oracle F0.5 per config computed from ranks, where
config pool = union of top-k per index; frozen pool = addr<=50 or name<=10 (verified equal to frozen cache).
"""
import os, sys, time, pickle, gc, collections
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import retrieval_engine as RE
from recon05_baseline_scorer import normalize
from translit import normalize_fixed, has_indic

OUT = os.path.join(H.ROOT, "experiments", "AN03"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(H.ROOT, "student_resource", "dataset", "train")
DEEP, SHALLOW = 1000, 50
FN = lambda x: normalize_fixed(x, translit=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(src, country):
    ids, raw_n, raw_a = [], [], []
    with open(os.path.join(DATA, f"train_source{src}.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t"); p += [""] * (4 - len(p))
            if p[3] == country:
                ids.append(p[0]); raw_n.append(p[1]); raw_a.append(p[2])
    return ids, raw_n, raw_a


def main():
    D = H.load_e008(verbose=False)
    s1d = D["s1_dict"]
    raw = pickle.load(open(os.path.join(H.SHARED, "raw_text.pkl"), "rb"))
    gt_owner = {c: s for s, v in s1d.items() for c in v["gt"]}
    rec = {}   # gt id -> info
    for ctry in ["US", "India"]:
        q = [s for s in s1d if s1d[s]["country"] == ctry]
        qn = [s1d[s]["name"] for s in q]; qa = [s1d[s]["addr"] for s in q]
        qna = [(n + " " + a) if a else n for n, a in zip(qn, qa)]
        qfn = [FN(raw[s][0]) for s in q]
        for src in ["2", "3"]:
            t0 = time.time()
            ids, rn, ra = load(src, ctry)
            nn = [normalize(x) for x in rn]; na = [normalize(x) for x in ra]
            want = {i: j for j, i in enumerate(ids) if i in gt_owner}
            log(f"{ctry} S{src}: {len(ids):,} docs, {len(want):,} gt docs, load {time.time()-t0:.0f}s")
            for j_id, j in want.items():
                s = gt_owner[j_id]
                rec[j_id] = dict(s1=s, country=ctry, src=src, raw_name=rn[j], raw_addr=ra[j], name=nn[j], addr=na[j],
                                 indic_name=has_indic(rn[j]), indic_addr=has_indic(ra[j]), addr_missing=not na[j],
                                 name_tset=fuzz.token_set_ratio(s1d[s]["name"], nn[j]) / 100,
                                 addr_tset=(fuzz.token_set_ratio(s1d[s]["addr"], na[j]) / 100) if na[j] else -1.0,
                                 name_tset_tr=fuzz.token_set_ratio(FN(raw[s][0]), FN(rn[j])) / 100,
                                 in_frozen=j_id in D["cands"][s])
            del ra; gc.collect()
            idx_of = {j: j_id for j_id, j in want.items()}
            specs = [("addr", lambda: [(n + " " + a) if a else n for n, a in zip(nn, na)], qna, DEEP),
                     ("name", lambda: nn, qn, DEEP),
                     ("tname", lambda: [FN(x) for x in rn], qfn, SHALLOW),
                     ("aonly", lambda: na, qa, SHALLOW)]
            for kind, docs_fn, qtexts, k in specs:
                tb = time.time()
                eng = RE.BM25Engine().build(docs_fn(), verbose=False)
                res = eng.query(qtexts, k)
                del eng; gc.collect()
                for qi, s in enumerate(q):
                    for r, j in enumerate(res[qi]):
                        if j < 0:
                            break
                        j_id = idx_of.get(int(j))
                        if j_id is not None and gt_owner[j_id] == s:
                            rec[j_id][f"rank_{kind}"] = r + 1
                log(f"   {kind} built+queried in {time.time()-tb:.0f}s")
            del ids, rn, nn, na, want, idx_of; gc.collect()
    pickle.dump(rec, open(os.path.join(OUT, "an03_ranks.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(f"saved {len(rec):,} gt records")


if __name__ == "__main__":
    main()
