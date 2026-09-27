"""
E010 -- Learning-curve diagnostic on the frozen E008 pipeline.

Subsample train S1 entities (500 / 1000 / 1994), rebuild the FULL E008 two-stage pipeline on
the subset (base OOF -> competition block -> stage-2), evaluate on the unchanged 2,001 val S1.
Only the competition block depends on the training subset; all other blocks are per-pair and
label-free, so they are sliced from the cached E008 matrices. Val is never touched for fitting.
Optional extra feature block (e.g. E009 NUM) can be appended via --with-num.
"""
import os, sys, json, time, argparse
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

OUT = os.path.join(H.ROOT, "experiments", "E010")
os.makedirs(OUT, exist_ok=True)


def run_subset(D, sub_ids, extra_tr=None, extra_va=None, seed=42):
    sub = set(sub_ids)
    mask = np.array([m[0] in sub for m in D["train_meta"]])
    tm = [m for m, k in zip(D["train_meta"], mask) if k]
    X22, y = D["X22_tr"][mask], D["y_tr"][mask]
    ordered = [s for s in D["train_s1_ids"] if s in sub]
    oof_base, val_base = H.golden.generate_oof_and_val_base_scores(X22, y, tm, ordered, D["X22_va"])
    A_tr = H.golden.extract_block_a_competition(tm, oof_base)
    A_va = H.golden.extract_block_a_competition(D["val_meta"], val_base)
    # columns: 0-21 base, 22-33 competition, 34-55 other label-free blocks
    X_tr = np.hstack([X22, A_tr, D["X_tr"][mask][:, 34:]])
    X_va = np.hstack([D["X22_va"], A_va, D["X_va"][:, 34:]])
    if extra_tr is not None:
        X_tr = np.hstack([X_tr, extra_tr[mask]]); X_va = np.hstack([X_va, extra_va])
    fit = H.fit_stage2(X_tr, y, X_va, tm, ordered, seed=seed)
    Dsub = dict(D, train_meta=tm, train_s1_ids=ordered)
    return H.evaluate_arm(f"n={len(ordered)}", fit, Dsub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-num", action="store_true", help="append E009 NUM+TOK blocks (E009-D feature set)")
    args = ap.parse_args()
    D = H.load_e008()
    extra_tr = extra_va = None
    tag = "e008"
    if args.with_num:
        import pickle
        F = pickle.load(open(os.path.join(H.ROOT, "experiments", "E009", "e009_features.pkl"), "rb"))
        extra_tr, extra_va, tag = np.hstack([F["NUM_tr"], F["TOK_tr"]]), np.hstack([F["NUM_va"], F["TOK_va"]]), "e009D"
    rows = []
    t0 = time.time()
    ids = list(D["train_s1_ids"])
    for n, draws in [(500, 3), (1000, 3), (len(ids), 1)]:
        for d in range(draws):
            rng = np.random.default_rng(100 + d)
            sub = ids if n == len(ids) else list(rng.choice(ids, size=n, replace=False))
            r = run_subset(D, sub, extra_tr, extra_va)
            row = dict(n=n, draw=d, hist_th=r["hist"]["th"], hist=r["hist"]["macro"], oof_th=r["oof"]["th"],
                       oof=r["oof"]["macro"], oof_us=r["oof"]["us"], oof_in=r["oof"]["india"],
                       oof_tp=r["oof"]["tp"], oof_fp=r["oof"]["fp"], oof_sing=r["oof"]["singleton"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    summ = {}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        summ[n] = dict(hist_mean=float(np.mean([r["hist"] for r in rs])), oof_mean=float(np.mean([r["oof"] for r in rs])),
                       oof_std=float(np.std([r["oof"] for r in rs])), draws=len(rs))
        print(n, summ[n])
    json.dump(dict(rows=rows, summary=summ, runtime_s=time.time() - t0),
              open(os.path.join(OUT, f"e010_learning_curve_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
