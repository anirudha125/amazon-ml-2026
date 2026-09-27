"""E027_FR step 3: mirror census on the full France street-equal pool, per pre-registered class."""
import json, pickle, collections, numpy as np
P = json.load(open("preregistration.json")); L = P["lists"]; O = P["offsets"]
ROLE, FILLER, LEGAL = set(L["ROLE"]), set(L["FILLER"]), set(L["LEGAL"])
SHIFT, MIRROR, NEIGH = O["shift"], O["mirror"], O["neighbour_background"]
E = pickle.load(open("ent.pkl", "rb")); z = np.load("pairs_streq.npz")
df = collections.Counter(t for n in E["names"] for t in set(n))
e, ta, tb, d = z["e"], z["ta"], z["tb"], z["d"]


def classify(e, a, b):
    if e == 1:
        if a in LEGAL and b in LEGAL: return "D_legal_flip"
        if a in ROLE and b in ROLE: return "A_generic_role_sub"
        if min(df[a], df[b]) < 200: return "C_rare_sub"
        return "X_content_sub"
    if e in (2, 3):
        t = b if e == 2 else a
        return "B_filler_legal_attach" if (t in FILLER or t in LEGAL) else "Y_other_attach"
    return {0: "Z_exact_name", 4: "W_multi_token"}[e]


if __name__ == "__main__":
    win = (d >= -25) & (d <= 25)
    idx = np.nonzero(win)[0]
    cls = np.array([classify(e[i], ta[i], tb[i]) for i in idx])
    np.save("cls_win.npy", cls); np.save("idx_win.npy", idx)
    dd = d[idx]
    res = {}
    def stats(mask):
        c = collections.Counter(dd[mask].tolist())
        n = {k: c.get(k, 0) for k in range(-25, 26)}
        sh = sum(n[k] for k in SHIFT); mi = sum(n[k] for k in MIRROR); bg = np.mean([n[k] for k in NEIGH])
        return dict(n0=n[0], shift_sum=sh, mirror_sum=mi, shift_mirror_ratio=(sh / mi if mi else float("inf")),
                    background_per_offset=float(bg), n0_over_bg=(n[0] / bg if bg else float("inf")),
                    mirror_per_offset=mi / len(MIRROR), shift_per_offset=sh / len(SHIFT),
                    per_k_ratio={k: (n[k] / n[-k] if n[-k] else None) for k in SHIFT},
                    counts={str(k): v for k, v in n.items()})
    for c in sorted(set(cls)):
        res[c] = stats(cls == c)
    # top role token pairs and B tokens
    isA = cls == "A_generic_role_sub"
    pairs = np.array(["/".join(sorted((ta[i], tb[i]))) for i in idx[isA]])
    top = collections.Counter(pairs[dd[isA] == 0].tolist()).most_common(15)
    res["A_top_pairs"] = {p: stats(np.isin(np.arange(len(idx)), np.nonzero(isA)[0][pairs == p])) for p, _ in top}
    isB = cls == "B_filler_legal_attach"
    btok = np.array([(tb[i] if e[i] == 2 else ta[i]) for i in idx[isB]])
    topb = collections.Counter(btok[dd[isB] == 0].tolist()).most_common(15)
    res["B_top_tokens"] = {t: stats(np.isin(np.arange(len(idx)), np.nonzero(isB)[0][btok == t])) for t, _ in topb}
    json.dump(res, open("census.json", "w"), indent=1)
    rows = ["| bucket | n(0) | bg/offset (-2,-1,+6,+8,+10) | n(0)/bg | shift sum | mirror sum | shift/mirror |", "|---|---|---|---|---|---|---|"]
    def row(name, s):
        rows.append(f"| {name} | {s['n0']} | {s['background_per_offset']:.1f} | {s['n0_over_bg']:.1f} | {s['shift_sum']} | {s['mirror_sum']} | {s['shift_mirror_ratio']:.2f} |")
    for c in sorted(k for k in res if not k.startswith(("A_top", "B_top"))): row(c, res[c])
    for p, s in res["A_top_pairs"].items(): row("A: " + p, s)
    for p, s in res["B_top_tokens"].items(): row("B: " + p, s)
    open("census.md", "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))
    for c in ("A_generic_role_sub", "B_filler_legal_attach", "D_legal_flip", "Z_exact_name"):
        print(c, {k: res[c]["counts"][str(k)] for k in range(-5, 14)})
    print("D per-k ratio", res["D_legal_flip"]["per_k_ratio"])
