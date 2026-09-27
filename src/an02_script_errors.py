"""AN02 -- error/acceptance breakdown by candidate-name script (India val), E008 vs E009-D, OOF thresholds."""
import os, sys, json, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from translit import has_indic

def main():
    D = H.load_e008(verbose=False)
    raw = pickle.load(open(os.path.join(H.SHARED, "raw_text.pkl"), "rb"))
    P8 = pickle.load(open(os.path.join(H.SHARED, "e008_stage2_preds.pkl"), "rb"))
    P9 = pickle.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_preds_seed42.pkl"), "rb"))["D_num_tok"]
    r9 = json.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_results.json")))["arms"]["D_num_tok"]
    s1d = D["s1_dict"]
    out = {}
    for tag, pv, th in [("E008_oof", P8["fit_p_va"], P8["res"]["oof"]["th"]), ("E009D_oof", P9["p_va"], r9["s42_oof"]["th"])]:
        st = collections.defaultdict(lambda: collections.Counter())
        for (s, c, y), p in zip(D["val_meta"], pv):
            ctry = s1d[s]["country"]
            nm, ad, _ = raw[c]
            key = f"{ctry}|{'indic_name' if has_indic(nm) else ('indic_addr_only' if has_indic(ad) else 'latin')}"
            acc = p >= th
            st[key]["pairs"] += 1
            st[key]["pos"] += int(y); st[key]["neg"] += int(1 - y)
            st[key]["tp"] += int(acc and y); st[key]["fn"] += int((not acc) and y); st[key]["fp"] += int(acc and not y)
        res = {}
        for k, v in sorted(st.items()):
            res[k] = dict(v, acceptance=v["tp"] / max(v["pos"], 1), fp_per_1k_neg=1000 * v["fp"] / max(v["neg"], 1),
                          precision=v["tp"] / max(v["tp"] + v["fp"], 1))
            print(tag, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in res[k].items()})
        out[tag] = res
    json.dump(out, open(os.path.join(H.ROOT, "experiments", "AN02", "an02_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
