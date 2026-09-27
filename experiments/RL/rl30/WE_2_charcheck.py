"""RL-30 / WE -- part 2: is F2 (token-Jaccard tie/strict vs a co-located near-duplicate sibling) measuring ownership, or a tokenization
artifact? (READ-ONLY; writes rl30/WE_2_results.json)
(a) Label-only audit with larger support: every train S1 that has a sibling (788 S1); for every GT record owned by that S1 or by one of
    its siblings, compute F2_ge / F2_gt / tie w.r.t. this S1 and ask P(owner == this S1 | flag). No model involved.
    Also a character-level tiebreak: token_sort_ratio(rec, this S1) vs max over siblings.
(b) Test France / India / US kept flagged pairs (S005, after max-claimer): character-level closeness of the record to this S1 vs to the
    sibling, and how many flags survive when dotted initials (S.A.S, S.A.R.L) are collapsed into one token."""
import os, sys, time, json, re, collections
import numpy as np, pandas as pd
from rapidfuzz import fuzz
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import load, toks, fold, accepted
from WE_1_verify import ns, jac, sibling_table

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def ns_collapse(s):
    """token set where runs of single-letter tokens are joined (s a r l -> sarl)"""
    out, run = [], []
    for t in toks(s):
        if len(t) == 1 and t.isalpha():
            run.append(t)
        else:
            if run:
                out.append("".join(run)); run = []
            out.append(t)
    if run:
        out.append("".join(run))
    return frozenset(out)


def srt(s):
    return " ".join(sorted(toks(s)))


def flags(Rn, A, sibs):
    jr = jac(Rn, A); js = max(jac(Rn, B) for B in sibs)
    return (js > 0 and js >= jr), js > jr


def char_side(rec, s1name, sibnames):
    rs = fuzz.token_sort_ratio(srt(rec), srt(s1name)); rb = max(fuzz.token_sort_ratio(srt(rec), srt(b)) for b in sibnames)
    return "S1" if rs > rb else ("sib" if rb > rs else "eq"), rs, rb


def main():
    t0 = time.time(); R = {}
    # ---------------- (a) label-only audit on all train S1 with siblings
    D = load("train", verbose=False)
    s1 = D["s1"]; S1 = s1.set_index("id"); recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
    sib, _, _ = sibling_table(s1)
    gt = D["gt"]; own = gt.groupby("s1").rec.apply(list).to_dict()
    rows = []
    for s, L in sib.items():
        A = ns(S1.name.loc[s]); sibsets = [ns(S1.name.loc[j]) for j in L]
        for o in [s] + L:
            for r in own.get(o, []):
                rn = recs.name.loc[r]; Rn = ns(rn)
                ge, gtf = flags(Rn, A, sibsets)
                side, rs, rb = char_side(rn, S1.name.loc[s], [S1.name.loc[j] for j in L])
                Ac = ns_collapse(S1.name.loc[s]); gec, gtc = flags(ns_collapse(rn), Ac, [ns_collapse(S1.name.loc[j]) for j in L])
                rows.append(dict(s1=s, c=S1.country.loc[s], rec=r, mine=o == s, fge=ge, gtx=gtf, side=side, gec=gec))
    df = pd.DataFrame(rows)
    a = {}
    for nm, m in (("F2_ge", df.fge), ("F2_gt", df.gtx), ("F2_tie_only", df.fge & ~df.gtx), ("not_flagged", ~df.fge), ("all", df.fge | ~df.fge),
                  ("F2_ge_collapsed", df.gec)):
        sub = df[m]
        a[nm] = dict(records=int(len(sub)), s1=int(sub.s1.nunique()), P_owner_is_this_s1=round(float(sub.mine.mean()), 4) if len(sub) else None,
                     by_country={c: dict(records=int((sub.c == c).sum()), P_this=round(float(sub[sub.c == c].mine.mean()), 4) if (sub.c == c).any() else None)
                                 for c in ("US", "India")})
        for side in ("S1", "sib", "eq"):
            ss = sub[sub.side == side]
            a[nm][f"char_{side}"] = dict(records=int(len(ss)), P_this=round(float(ss.mine.mean()), 4) if len(ss) else None)
    # S1-clustered view for the tie subset: per S1 share of its flagged records that are its own
    tie = df[df.fge & ~df.gtx]
    per = tie.groupby("s1").mine.mean()
    a["F2_tie_only"]["per_s1_mean_P_this"] = round(float(per.mean()), 4) if len(per) else None
    a["F2_tie_only"]["n_s1"] = int(len(per))
    gtt = df[df.gtx]; per2 = gtt.groupby("s1").mine.mean()
    a["F2_gt"]["per_s1_mean_P_this"] = round(float(per2.mean()), 4) if len(per2) else None
    R["train_label_only_audit_owned_records"] = a
    log("train audit", json.dumps(a)[:3000], f"{time.time()-t0:.0f}s")
    del D, s1, S1, recs, gt, own

    # ---------------- (b) test kept flagged pairs: char-level side, collapse robustness
    T = load("test", verbose=False)
    s1 = T["s1"]; S1 = s1.set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    sib, _, _ = sibling_table(s1)
    B = {}
    for tag in ("S005_France", "S005_US", "S005_India"):
        a_ = accepted(tag).reset_index(drop=True)
        m = a_.s1.map(lambda s: s in sib).values
        sub = a_[m]
        out = []
        for s, r, p, kept, nc in zip(sub.s1.values, sub.rec.values, sub.p.values, sub.kept_final.values, sub.n_claims.values):
            A = ns(S1.name.loc[s]); L = sib[s]; rn = recs.name.loc[r]
            ge, gtf = flags(ns(rn), A, [ns(S1.name.loc[j]) for j in L])
            if not ge:
                continue
            gec, gtc = flags(ns_collapse(rn), ns_collapse(S1.name.loc[s]), [ns_collapse(S1.name.loc[j]) for j in L])
            side, rs, rb = char_side(rn, S1.name.loc[s], [S1.name.loc[j] for j in L])
            out.append(dict(kept=bool(kept), gtx=gtf, side=side, rs=rs, rb=rb, gec=gec, gtc=gtc, mc=nc > 1))
        d = pd.DataFrame(out)
        res = {}
        for nm, mm in (("accepted_F2_ge", np.ones(len(d), bool)), ("kept_F2_ge", d.kept.values), ("kept_tie_only", (d.kept & ~d.gtx).values),
                       ("kept_gt", (d.kept & d.gtx).values)):
            x = d[mm]
            res[nm] = dict(n=int(len(x)), char_closer_to_this_S1=int((x.side == "S1").sum()), char_closer_to_sibling=int((x.side == "sib").sum()),
                           char_equal=int((x.side == "eq").sum()), still_F2_ge_after_collapsing_initials=int(x.gec.sum()),
                           still_F2_ge_after_collapse_and_not_char_closer_to_S1=int((x.gec & (x.side != "S1")).sum()))
        B[tag] = res
        log(tag, res, f"{time.time()-t0:.0f}s")
    R["test_char_side"] = B
    json.dump(R, open(os.path.join(HERE, "WE_2_results.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
