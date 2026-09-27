"""RL-28 -- verification of the RECON-08_GOLDEN compatibility shim (user-approved 2026-09-27). Read-only except the new
result file experiments/RL/rl28_shim_verification.json (refuses to overwrite).
  1. import-check: experiments/RL/rl27_arm.py, src/e023_train_rr.py, src/p3_test.py, src/e024_arms.py (main-guarded; no run)
  2. LightGBM params built by the current pipeline vs the saved D2b model (base and stage 2 seed 42), full get_params()
  3. re-score V1 with the saved D2b model through the exact run_arm assembly and compare with the stored predictions
"""
import os, sys, json, pickle, importlib, importlib.util, traceback
import numpy as np

os.environ["LGB_THREADS"] = "16"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src"); sys.path.insert(0, SRC)
OUTF = os.path.join(HERE, "rl28_shim_verification.json")
assert not os.path.exists(OUTF), "refusing to overwrite an existing verification record"
res = {"imports": {}}

# 1) import checks
for name, path in [("rl27_arm", os.path.join(HERE, "rl27_arm.py")), ("e023_train_rr", os.path.join(SRC, "e023_train_rr.py")),
                   ("p3_test", os.path.join(SRC, "p3_test.py")), ("e024_arms", os.path.join(SRC, "e024_arms.py"))]:
    try:
        spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod); res["imports"][name] = "OK"
    except Exception as e:
        res["imports"][name] = f"FAIL: {type(e).__name__}: {e}"
import harness as H
import e023_stage2 as S2
import lightgbm as lgb

# 2) parameter verification
M = pickle.load(open(os.path.join(ROOT, "experiments", "E024", "model_D2b_union_rrUb_big.pkl"), "rb"))
def cmp(built, saved):
    b, s = built.get_params(), saved.get_params()
    return {k: [b.get(k), s.get(k)] for k in sorted(set(b) | set(s)) if b.get(k) != s.get(k)}
res["params_base_diff"] = cmp(lgb.LGBMClassifier(**S2.BASE_P), M["base"])
res["params_stage2_s42_diff"] = cmp(lgb.LGBMClassifier(**S2.lgbm(42)), M["stage2"][42])
res["golden_LGB_PARAMS"] = dict(H.LGB_PARAMS)

# 3) V1 re-score parity
rr = pickle.load(open(os.path.join(ROOT, "experiments", "E023", "rr_rrUb_big.pkl"), "rb"))
V = S2.load_set("V1", "a50n10d10a"); V["country_s1"] = V["country"]
DENSE = tuple(M["base_cols"]); assert DENSE == ("rank_dense", "dcos") or list(DENSE) == ["rank_dense", "dcos"], DENSE
Xb = np.ascontiguousarray(np.hstack([V["LF"][:, :22]] + [V[c].astype(np.float32)[:, None] for c in DENSE]).astype(np.float32))
pb = M["base"].predict_proba(Xb)[:, 1]; sel = S2.topk_mask(V["s1idx"], pb, M["topk"])
cols = [V["LF"][:, :22], S2.block_a_vec(V["s1idx"], pb), V["LF"][:, 22:]] + [V[c].astype(np.float32)[:, None] for c in M["extra_cols"]]
cols.append(S2.rr_col(V, sel, rr)[:, None])
X = np.hstack(cols).astype(np.float32)
assert X.shape[1] == M["n_feat"], (X.shape, M["n_feat"])
p = M["stage2"][42].predict_proba(X)[:, 1].astype(np.float32)
ref = np.load(os.path.join(ROOT, "experiments", "E024", "p_D2b_union_rrUb_big_V1_s42.npy")).astype(np.float32)
th = M["th_oof"][42]
res["rescore"] = dict(n_pairs=int(len(p)), n_feat=int(X.shape[1]), max_abs_diff=float(np.max(np.abs(p - ref))),
                      n_exact_equal=int(np.sum(p == ref)), decisions_changed_at_th=int(np.sum((p >= th) != (ref >= th))), th=float(th),
                      macro_rescored=round(S2.summarize(V, p, th)["macro"] * 100, 4), macro_stored=round(S2.summarize(V, ref, th)["macro"] * 100, 4))
res["verdict"] = ("PASS" if all(v == "OK" for v in res["imports"].values()) and not res["params_base_diff"]
                  and not res["params_stage2_s42_diff"] and res["rescore"]["max_abs_diff"] == 0.0 else "CHECK")
json.dump(res, open(OUTF, "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
