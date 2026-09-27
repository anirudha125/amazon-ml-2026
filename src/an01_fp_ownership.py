"""
AN01 -- FP ownership analysis on frozen E008 validation predictions.

For each validation FP (E008, both threshold protocols): does the FP candidate record
belong to ANOTHER train S1 (ownership conflict -> reverse competition could remove it)
or to no S1 at all (pure distractor -> reverse competition cannot help)?
For owned FPs, compare how similar the candidate is to its true owner vs to the query S1.
Uses train ground truth only for diagnosis (never as a feature).
"""
import os, sys, json, pickle, collections
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from recon05_baseline_scorer import normalize

OUT = os.path.join(H.ROOT, "experiments", "AN01")
os.makedirs(OUT, exist_ok=True)
TRAIN = os.path.join(H.ROOT, "student_resource", "dataset", "train")

def sim(n1, a1, n2, a2):
    return fuzz.token_set_ratio(n1, n2) / 100.0, (fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else -1.0)

def main():
    D = H.load_e008(verbose=False)
    s1d, C = D["s1_dict"], D["cands"]
    P8 = pickle.load(open(os.path.join(H.SHARED, "e008_stage2_preds.pkl"), "rb"))
    P9 = pickle.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_preds_seed42.pkl"), "rb"))["D_num_tok"]
    r9 = json.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_results.json")))["arms"]["D_num_tok"]
    sources = {"E008_hist": (P8["fit_p_va"], 0.52), "E008_oof": (P8["fit_p_va"], P8["res"]["oof"]["th"]),
               "E009D_hist": (P9["p_va"], r9["s42_hist"]["th"]), "E009D_oof": (P9["p_va"], r9["s42_oof"]["th"])}
    protos = {k: v[1] for k, v in sources.items()}
    fps = {k: [] for k in protos}; tps = {k: [] for k in protos}
    for i, (s, c, y) in enumerate(D["val_meta"]):
      for k, (pv, th) in sources.items():
        p = pv[i]
        if p >= th:
            (tps if y else fps)[k].append((s, c, float(p)))
    need = {c for k in fps for _, c, _ in fps[k]}
    owner = {}
    with open(os.path.join(TRAIN, "train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1]:
                for cid in p[1].split(","):
                    if cid in need:
                        owner[cid] = p[0]
    owners = set(owner.values())
    orec = {}
    with open(os.path.join(TRAIN, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in owners:
                orec[p[0]] = (normalize(p[1]), normalize(p[2]), p[3])

    sample_ids = set(s1d)
    report = {}
    examples = []
    for k in protos:
        L = fps[k]
        owned = [(s, c, p) for s, c, p in L if c in owner]
        sing = [(s, c, p) for s, c, p in L if len(s1d[s]["gt"]) == 0]
        owned_sing = [x for x in sing if x[1] in owner]
        closer_owner = 0; margins = []
        for s, c, p in owned:
            ci = C[s][c]; o = orec[owner[c]]
            nq, aq = sim(s1d[s]["name"], s1d[s]["addr"], ci["name"], ci["addr"])
            no, ao = sim(o[0], o[1], ci["name"], ci["addr"])
            sq = nq + max(aq, 0); so = no + max(ao, 0)
            margins.append(so - sq)
            if so > sq:
                closer_owner += 1
            if k == "E009D_oof" and len(examples) < 40:
                examples.append(dict(p=p, query=f"{s1d[s]['name']} | {s1d[s]['addr']}",
                                     cand=f"{ci['name']} | {ci['addr']}", owner=f"{o[0]} | {o[1]}",
                                     q_sim=[nq, aq], o_sim=[no, ao]))
        # entity-level impact: how much macro F0.5 is lost to FPs, split by owned vs distractor
        by_s1 = collections.defaultdict(lambda: [0, 0])
        for s, c, p in L:
            by_s1[s][0 if c in owner else 1] += 1
        report[k] = dict(
            threshold=protos[k], n_fp=len(L), n_tp=len(tps[k]),
            fp_owned_by_other_s1=len(owned), fp_pure_distractor=len(L) - len(owned),
            owned_frac=len(owned) / max(len(L), 1),
            owner_in_our_3995_sample=sum(1 for _, c, _ in owned if owner[c] in sample_ids),
            singleton_fp=len(sing), singleton_fp_owned=len(owned_sing),
            owned_where_owner_is_closer_by_tokenset=closer_owner,
            owned_owner_margin_median=float(np.median(margins)) if margins else None,
            s1_with_fp=len(by_s1),
            s1_with_only_owned_fps=sum(1 for v in by_s1.values() if v[1] == 0),
            s1_with_only_distractor_fps=sum(1 for v in by_s1.values() if v[0] == 0),
            fp_country=dict(collections.Counter(s1d[s]["country"] for s, _, _ in L)),
            owned_country=dict(collections.Counter(s1d[s]["country"] for s, _, _ in owned)),
        )
        print(k, json.dumps(report[k], indent=1))
    json.dump(dict(report=report, examples=examples), open(os.path.join(OUT, "an01_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for e in examples[:15]:
        print(f"\n{e['p']:.2f} Q: {e['query']}\n     C: {e['cand']}\n     O: {e['owner']}  q_sim={e['q_sim']} o_sim={e['o_sim']}")

if __name__ == "__main__":
    main()
