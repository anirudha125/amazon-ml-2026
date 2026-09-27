"""
E024 arms -- named stage-2 configurations on the E024 feature sets, evaluated on V0 (2,001) and V1 (20,000) S1.
  python src/e024_arms.py run <arm> [<arm> ...]
  python src/e024_arms.py compare <armA|PROD> <armB>          (paired bootstrap per seed + seed-averaged, V0 and V1)
Results: experiments/E024/arm_<name>.pkl (with per-S1 scores) and arm_<name>.json.
Reranker score dicts {(s1, cand): logit} come from experiments/E023/rr_<tag>.pkl (models trained on DISJOINT S1 only).
"""
import os, sys, json, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import e023_stage2 as S2

E23 = os.path.join(H.ROOT, "experiments", "E023"); E24 = S2.E24
T2 = ["T0", "E014"]                        # the E014-B / E018C stage-2 training set (11,994 S1)
T2BIG = ["T0", "E014", "T2X"]              # + 40,000 (51,994 S1)
DENSE = ("rank_dense", "dcos")
ARMS = {
    "B0_lex":        dict(train=T2, pooltag="a50n10"),
    "B1_lex_dense":  dict(train=T2, pooltag="a50n10d0a", extra_cols=DENSE, base_cols=DENSE),
    "B2_union":      dict(train=T2, pooltag="a50n10d10a", extra_cols=DENSE, base_cols=DENSE),
    "B2_big":        dict(train=T2BIG, pooltag="a50n10d10a", extra_cols=DENSE, base_cols=DENSE),
    "B3_union_dcomp": dict(train=T2, pooltag="a50n10d10a", extra_cols=DENSE + ("dcomp",), base_cols=DENSE),
    "C0_lex_rrA":    dict(train=T2, pooltag="a50n10", rr="rrA_a50n10"),
    "C2_union_rrU":  dict(train=T2, pooltag="a50n10d10a", extra_cols=DENSE, base_cols=DENSE, rr="rrU_a50n10d10a"),
    "C2b_union_rrUb": dict(train=T2, pooltag="a50n10d10a", extra_cols=DENSE, base_cols=DENSE, rr="rrUb_a50n10d10a"),
    "D2b_union_rrUb_big": dict(train=T2BIG, pooltag="a50n10d10a", extra_cols=DENSE, base_cols=DENSE, rr="rrUb_big", seeds=(42,)),
    "D2_union_rrU_big": dict(train=T2BIG, pooltag="a50n10d10a", extra_cols=DENSE, base_cols=DENSE, rr="rrU_a50n10d10a"),
}


def load_rr(tag):
    return pickle.load(open(os.path.join(E23, f"rr_{tag}.pkl"), "rb")) if tag else None


def run(name, **over):
    cfg = dict(ARMS[name]); cfg.update(over)
    rr = load_rr(cfg.pop("rr", None))
    cfg.setdefault("save_model", os.path.join(E24, f"model_{name}.pkl"))
    res = S2.run_arm(name, cfg.pop("train"), rr_train=rr, rr_val=rr, **cfg)
    pickle.dump(res, open(os.path.join(E24, f"arm_{name}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(S2.strip(res), open(os.path.join(E24, f"arm_{name}.json"), "w"), indent=1, default=float)
    return res


def load_arm(name):
    if name == "PROD":   # S002 model applied unchanged; single "seed", fixed threshold .72
        r = {"seeds": {0: {}}}
        for vn in ["V0", "V1"]:
            r["seeds"][0][vn] = {"oof": {"scores": np.load(os.path.join(E23, f"f_PROD_{vn}.npy"))}}
            r["seeds"][0][vn]["hist"] = r["seeds"][0][vn]["oof"]
        return r
    return pickle.load(open(os.path.join(E24, f"arm_{name}.pkl"), "rb"))


def compare(a, b):
    A, B = load_arm(a), load_arm(b)
    if a == "PROD":        # compare every seed of B against the single PROD vector
        A = {"seeds": {s: A["seeds"][0] for s in B["seeds"]}}
    out = {p: S2.compare(A, B, p) for p in ("oof", "hist")}
    print(f"{b} vs {a}:", json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        for n in sys.argv[2:]:
            run(n)
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
