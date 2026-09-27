"""GAP-03: composition of the uncertain band (and of rejected mass) by address agreement x name difference,
France test (S005 p5 / S006 p6, no labels) vs V1 (RRL / NEW, labelled). Same string rules for all countries.
Label-free on test. Output gap03_band_classes.json"""
import re, json, unicodedata, numpy as np, pandas as pd
import gap_lib as G, rl31_lib as L
LEGAL = set("llc inc corp corporation ltd limited pvt private llp co company lp pc pllc plc sarl sas sasu eurl sa sci snc selarl scop".split())
ABBR = {"st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane", "ct": "court", "blvd": "boulevard",
        "bd": "boulevard", "pl": "place", "r": "rue", "all": "allee", "imp": "impasse", "ch": "chemin", "chem": "chemin", "rte": "route",
        "hwy": "highway", "pkwy": "parkway", "cir": "circle", "sq": "square", "fg": "faubourg"}
def fold(s):
    s = unicodedata.normalize("NFKD", s.lower()); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()
def ntoks(s):
    t = fold(s).split(); return [w for w in t if w not in LEGAL]
def atoks(s):
    return [ABBR.get(w, w) for w in fold(s).split()]
def num1(s):
    m = re.search(r"\d+", s or ""); return m.group(0).lstrip("0") if m else None
def acronym(s1n, rn):
    t = [w for w in ntoks(s1n) if w]; a = "".join(w[0] for w in t)
    r = fold(rn).replace(" ", "")
    return len(a) >= 2 and (r == a or r == a[:len(r)] and len(r) >= 2 and len(fold(rn).split()) == 1 and r in (a, a[1:], a[:-1]))
def classify(s1n, s1a, rn, ra):
    if not fold(ra): acls = "E"   # record address empty
    else:
        n1, n2 = num1(s1a), num1(ra)
        A1, A2 = set(w for w in atoks(s1a) if not w.isdigit()), set(w for w in atoks(ra) if not w.isdigit())
        jac = len(A1 & A2) / max(1, len(A1 | A2))
        if n1 and n2 and n1 == n2 and jac >= 0.5: acls = "A=num&street"
        elif (not n2 or not n1) and jac >= 0.5: acls = "A~nonum"
        elif n1 and n2 and n1 != n2 and jac >= 0.5: acls = "D:num_diff"
        else: acls = "D:street_diff"
    a, b = ntoks(s1n), ntoks(rn); A, B = set(a), set(b)
    if acronym(s1n, rn): ncls = "acronym"
    elif A == B: ncls = "same_core"
    elif A < B: ncls = "rec_adds" if len(B - A) == 1 else "rec_adds2+"
    elif B < A: ncls = "rec_drops" if len(A - B) == 1 else "rec_drops2+"
    elif len(A - B) == 1 and len(B - A) == 1: ncls = "swap1"
    elif not (A & B): ncls = "disjoint"
    else: ncls = "other"
    return acls, ncls
def tab(rows, w=None):
    df = pd.DataFrame(rows, columns=["a", "n"]); df["w"] = 1.0 if w is None else w
    return df
def main():
    out = {}
    lo, hi = 0.25, 0.75
    # France test
    F = G.france_scores("France"); te = G.records("test"); nS1 = len(F["u"])
    for pk, th in (("p5", 0.78), ("p6", 0.72)):
        p = F[pk]
        for nm, m in (("band", (p >= lo) & (p <= hi)), ("rej_p05_th", (p >= 0.05) & (p < th))):
            idx = np.flatnonzero(m)
            rows = [classify(te[F["s1"][i]][0], te[F["s1"][i]][1], te[F["cand"][i]][0], te[F["cand"][i]][1]) for i in idx]
            df = pd.DataFrame(rows, columns=["a", "n"])
            ct = (df.groupby(["a", "n"]).size() / nS1 * 1000).round(2)
            out[f"France_{pk}_{nm}_per1kS1"] = {f"{a}|{n}": v for (a, n), v in ct.items()}
            out[f"France_{pk}_{nm}_total_per1kS1"] = round(len(idx) / nS1 * 1000, 2)
            print(f"France {pk} {nm}: {len(idx):,} pairs, {len(idx)/nS1*1000:.1f} per 1k S1")
    # V1 labelled
    V = L.load_v1(); tr = G.records("train")
    cty = V["country"][V["s1idx"]]
    for pk, th in (("p_RRL", 0.72), ("p_NEW", 0.78)):
        p = V[pk]
        for nm, m in (("band", (p >= lo) & (p <= hi)), ("rej_p05_th", (p >= 0.05) & (p < th))):
            idx = np.flatnonzero(m)
            rows = [classify(tr[V["s1_ids"][V["s1idx"][i]]][0], tr[V["s1_ids"][V["s1idx"][i]]][1], tr[V["cand"][i]][0], tr[V["cand"][i]][1]) for i in idx]
            df = pd.DataFrame(rows, columns=["a", "n"]); df["y"] = V["y"][idx]; df["c"] = cty[idx]
            for c in ("US", "India"):
                d = df[df.c == c]; nS = (V["country"] == c).sum()
                g = d.groupby(["a", "n"]).agg(k=("y", "size"), prec=("y", "mean"))
                out[f"V1_{c}_{pk}_{nm}_per1kS1"] = {f"{a}|{n}": [round(r.k / nS * 1000, 2), round(r.prec, 3)] for (a, n), r in g.iterrows()}
                out[f"V1_{c}_{pk}_{nm}_total_per1kS1"] = [round(len(d) / nS * 1000, 2), round(float(d.y.mean()), 3)]
                print(f"V1 {c} {pk} {nm}: {len(d):,} pairs, {len(d)/nS*1000:.1f} per 1k S1, prec {d.y.mean():.3f}")
    json.dump(out, open("gap03_band_classes.json", "w"), indent=1)

if __name__ == '__main__':
    main()
