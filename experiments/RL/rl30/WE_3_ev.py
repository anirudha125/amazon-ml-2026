"""RL-30 / WE -- part 3: expected value of vetoing the S005 France kept F2_ge pairs under (i) E's uniform false fraction 6/7 and
(ii) per-pair false probabilities taken from the label-only train audit (WE_2: owned records of 788 train S1 with a sibling), split by
tie/strict and character-level side. ESTIMATE: other kept pairs of the S1 are assumed TP and n_gt = #TP kept (no missed records).
Output: rl30/WE_3_results.json"""
import os, sys, time, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import load, accepted
from WE_1_verify import ns, sibling_table
from WE_2_charcheck import flags, char_side

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    t0 = time.time()
    W2 = json.load(open(os.path.join(HERE, "WE_2_results.json")))["train_label_only_audit_owned_records"]
    q = {("gt", "S1"): 1 - W2["F2_gt"]["char_S1"]["P_this"], ("gt", "sib"): 1 - W2["F2_gt"]["char_sib"]["P_this"], ("gt", "eq"): 0.5,
         ("tie", "S1"): 1 - W2["F2_tie_only"]["char_S1"]["P_this"], ("tie", "sib"): 1 - W2["F2_tie_only"]["char_sib"]["P_this"],
         ("tie", "eq"): 1 - W2["F2_tie_only"]["char_eq"]["P_this"]}
    T = load("test", verbose=False)
    s1 = T["s1"]; s1 = s1[s1.country == "France"]; S1 = s1.set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    sib, _, _ = sibling_table(s1)
    a = accepted("S005_France"); a = a[a.kept_final].reset_index(drop=True)
    n_other = a.groupby("s1").size().to_dict()
    rows = []
    for s, r in zip(a.s1.values, a.rec.values):
        if s not in sib:
            continue
        rn = recs.name.loc[r]; L = sib[s]
        ge, gtf = flags(ns(rn), ns(S1.name.loc[s]), [ns(S1.name.loc[j]) for j in L])
        if ge:
            side, _, _ = char_side(rn, S1.name.loc[s], [S1.name.loc[j] for j in L])
            rows.append(dict(s1=s, kind="gt" if gtf else "tie", side=side))
    d = pd.DataFrame(rows); d["q"] = [q[(k, s)] for k, s in zip(d.kind, d.side)]
    g = d.groupby("s1"); S = list(g.groups.keys())
    v_list = [g.get_group(s).q.values for s in S]; oth = np.array([n_other[s] - len(v) for s, v in zip(S, v_list)])
    rng = np.random.default_rng(11); n_fr = int((T["s1"].country == "France").sum()); n_all = len(T["s1"])

    def ev(qfun, n=4000):
        tot = np.zeros(n)
        for qs, o in zip(v_list, oth):
            fp = rng.random((n, len(qs))) < qfun(qs)          # which flagged pairs are FP
            tp = (~fp).sum(1); k = len(qs) + o; g_ = o + tp        # n_gt = other kept (TP) + flagged TP
            before = np.where(g_ == 0, 0.0, 1.25 * g_ / (0.25 * g_ + k))
            after = np.where(g_ == 0, 1.0, np.where(o > 0, 1.25 * o / (0.25 * g_ + o), 0.0))
            tot += after - before
        return dict(S1eq_mean=round(float(tot.mean()), 2), S1eq_p05_p95=[round(float(np.percentile(tot, 5)), 2), round(float(np.percentile(tot, 95)), 2)],
                    France_pp=round(float(tot.mean() / n_fr * 100), 5), LB_pp=round(float(tot.mean() / n_all * 100), 5))
    out = dict(n_pairs=int(len(d)), n_s1=len(S), composition=d.groupby(["kind", "side"]).size().to_dict(), q_used={f"{k}|{s}": round(v, 4) for (k, s), v in q.items()},
               expected_FP_fraction_label_analog=round(float(d.q.mean()), 4),
               EV_E_uniform_6of7=ev(lambda qs: np.full(len(qs), 6 / 7)),
               EV_label_analog=ev(lambda qs: qs),
               EV_label_analog_veto_only_q_gt_0_5=None)
    # restricted veto: only pairs with q > 0.5 (strict or char-closer-to-sibling)
    tot = np.zeros(4000)
    for qs, o in zip(v_list, oth):
        fp = rng.random((4000, len(qs))) < qs; veto = qs > 0.5
        tp = (~fp).sum(1); k = len(qs) + o; g_ = o + tp
        before = np.where(g_ == 0, 0.0, 1.25 * g_ / (0.25 * g_ + k))
        kv = k - veto.sum(); tpv = o + (~fp[:, ~veto]).sum(1)
        after = np.where(g_ == 0, (kv == 0).astype(float), np.where(tpv > 0, 1.25 * tpv / (0.25 * g_ + kv), 0.0))
        tot += after - before
    out["EV_label_analog_veto_only_q_gt_0_5"] = dict(n_vetoed=int((d.q > 0.5).sum()), S1eq_mean=round(float(tot.mean()), 2),
                                                     S1eq_p05_p95=[round(float(np.percentile(tot, 5)), 2), round(float(np.percentile(tot, 95)), 2)],
                                                     France_pp=round(float(tot.mean() / n_fr * 100), 5), LB_pp=round(float(tot.mean() / n_all * 100), 5))
    out["composition"] = {f"{k[0]}|{k[1]}": int(v) for k, v in out["composition"].items()}
    log(out, f"{time.time()-t0:.0f}s")
    json.dump(out, open(os.path.join(HERE, "WE_3_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
