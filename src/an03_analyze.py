"""AN03 analysis: retrieval-miss breakdown + oracle macro-F0.5 for candidate retrieval configs (val 2,001 and all 3,995)."""
import os, sys, pickle, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

OUT = os.path.join(H.ROOT, "experiments", "AN03")
BIG = 10 ** 9


def oracle(s1_ids, s1d, rec, inpool):
    sc = []
    for s in s1_ids:
        gt = s1d[s]["gt"]
        if not gt:
            sc.append(1.0); continue
        hit = sum(1 for c in gt if c in rec and inpool(rec[c]))
        r = hit / len(gt)
        sc.append(1.25 * r / (0.25 + r) if r > 0 else 0.0)
    return np.array(sc)


def main():
    D = H.load_e008(verbose=False)
    s1d = D["s1_dict"]; val = D["val_s1_ids"]; alls = list(s1d)
    rec = pickle.load(open(os.path.join(OUT, "an03_ranks.pkl"), "rb"))
    for r in rec.values():
        for k in ("rank_addr", "rank_name", "rank_tname", "rank_aonly"):
            r.setdefault(k, BIG)
    n_gt = sum(len(s1d[s]["gt"]) for s in alls)
    print(f"gt records found {len(rec):,} of {n_gt:,}")
    cons = sum(1 for r in rec.values() if r["in_frozen"] != (r["rank_addr"] <= 50 or r["rank_name"] <= 10))
    print("frozen-pool membership inconsistent with recomputed ranks:", cons)
    miss = [r for r in rec.values() if not r["in_frozen"]]
    missv = [r for r in miss if r["s1"] in set(val)]
    print(f"misses: all {len(miss)} | val {len(missv)}")
    def brk(L, key):
        return dict(collections.Counter(key(r) for r in L).most_common())
    for name, L in [("val", missv), ("all", miss)]:
        print(f"\n--- {name} misses ({len(L)}) ---")
        print("country/src", brk(L, lambda r: f"{r['country']}/S{r['src']}"))
        print("script", brk(L, lambda r: "indic_name" if r["indic_name"] else ("indic_addr" if r["indic_addr"] else "latin")))
        print("addr missing", brk(L, lambda r: r["addr_missing"]))
        print("name tset bins", brk(L, lambda r: f"{min(int(r['name_tset']*10),9)/10:.1f}"))
        print("translit name tset bins", brk(L, lambda r: f"{min(int(r['name_tset_tr']*10),9)/10:.1f}"))
        print("addr tset bins", brk(L, lambda r: "missing" if r["addr_tset"] < 0 else f"{min(int(r['addr_tset']*10),9)/10:.1f}"))
        print("addr rank", brk(L, lambda r: "51-100" if r["rank_addr"] <= 100 else ("101-200" if r["rank_addr"] <= 200 else ("201-1000" if r["rank_addr"] <= 1000 else ">1000"))))
        print("name rank", brk(L, lambda r: "11-20" if r["rank_name"] <= 20 else ("21-50" if r["rank_name"] <= 50 else ("51-1000" if r["rank_name"] <= 1000 else ">1000"))))
        print("tname<=10", sum(r["rank_tname"] <= 10 for r in L), "tname<=20", sum(r["rank_tname"] <= 20 for r in L),
              "| aonly<=10", sum(r["rank_aonly"] <= 10 for r in L), "aonly<=20", sum(r["rank_aonly"] <= 20 for r in L))
        unrec = [r for r in L if min(r["rank_addr"], r["rank_name"] * 5) > 200 and r["rank_tname"] > 20 and r["rank_aonly"] > 20]
        print("not recoverable by any tested mechanism (addr>200, name>40, tname>20, aonly>20):", len(unrec))
        for r in L[:0]:
            pass
    base = lambda r: r["in_frozen"]
    configs = {
        "frozen (addr50+name10)": base,
        "+tname10": lambda r: base(r) or r["rank_tname"] <= 10,
        "+tname20": lambda r: base(r) or r["rank_tname"] <= 20,
        "+aonly10": lambda r: base(r) or r["rank_aonly"] <= 10,
        "+aonly20": lambda r: base(r) or r["rank_aonly"] <= 20,
        "name20": lambda r: base(r) or r["rank_name"] <= 20,
        "addr100": lambda r: base(r) or r["rank_addr"] <= 100,
        "addr100+name20": lambda r: base(r) or r["rank_addr"] <= 100 or r["rank_name"] <= 20,
        "+tname10+aonly10": lambda r: base(r) or r["rank_tname"] <= 10 or r["rank_aonly"] <= 10,
        "+tname20+aonly20": lambda r: base(r) or r["rank_tname"] <= 20 or r["rank_aonly"] <= 20,
        "addr100+name20+tname20+aonly20": lambda r: base(r) or r["rank_addr"] <= 100 or r["rank_name"] <= 20 or r["rank_tname"] <= 20 or r["rank_aonly"] <= 20,
    }
    extra_upper = {"frozen (addr50+name10)": 0, "+tname10": 20, "+tname20": 40, "+aonly10": 20, "+aonly20": 40, "name20": 20,
                   "addr100": 100, "addr100+name20": 120, "+tname10+aonly10": 40, "+tname20+aonly20": 80,
                   "addr100+name20+tname20+aonly20": 200}
    b_v = oracle(val, s1d, rec, base); b_a = oracle(alls, s1d, rec, base)
    res = {}
    print(f"\n{'config':<34} {'val oracle':>10} {'d_val':>7} {'CI':>16} {'all3995':>8} {'d_all':>7} {'recall_all':>10} {'+cands<=':>8}")
    for name, f in configs.items():
        v = oracle(val, s1d, rec, f); a = oracle(alls, s1d, rec, f)
        d, lo, hi, _ = H.paired_bootstrap(b_v, v)
        rc = sum(1 for r in rec.values() if f(r)) / n_gt
        res[name] = dict(val=v.mean(), d_val=d, ci=[lo, hi], all=a.mean(), d_all=a.mean() - b_a.mean(), recall_all=rc, extra_upper=extra_upper[name])
        print(f"{name:<34} {v.mean()*100:10.2f} {d*100:+7.2f} [{lo*100:+.2f},{hi*100:+.2f}] {a.mean()*100:8.2f} {(a.mean()-b_a.mean())*100:+7.2f} {rc*100:10.2f} {extra_upper[name]:8d}")
    json.dump(res, open(os.path.join(OUT, "an03_oracle.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
