"""RL-11 -- France proxy: how much does the country flag matter for a country with low-entropy house numbers?

France is scored with is_india=0 (US encoding) although its house numbers look like India's (RL-09: 7.0 / 8.2 / 12.5 bits
for France / India / US). Counterfactual on the frozen E018C models (no refit): flip is_india for val rows and re-score
(base -> competition block -> stage 2; reranker column unchanged since it is text-only). Read-only on all inputs.
"""
import sys, os, pickle, collections
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import harness as H
import pair_features as PF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl02_error_decomp import f05


def main():
    M = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/model_E018C_s42.pkl"), "rb"))
    D = H.load_e008(verbose=False)
    C = pickle.load(open(os.path.join(ROOT, "experiments/E014/e014_feats_10000_translit.pkl"), "rb"))
    rr = np.load(os.path.join(ROOT, "experiments/E017_embed/rerank_top10_full.npy"))
    LF = np.asarray(C["LF_va"], dtype=np.float32)
    vm = D["val_meta"]; th = M["th"]
    rr_va = rr[len(rr) - len(LF):]
    country = {s: D["s1_dict"][s].get("country") for s in D["val_s1_ids"]}
    is_ind = LF[:, 17] > 0.5
    rows_c = np.array([country[m[0]] for m in vm])
    assert np.all(is_ind == (rows_c == "India")), "column 17 is not is_india"

    def score(LFx):
        vb = M["base"].predict_proba(LFx[:, :22])[:, 1]
        X = PF.assemble(LFx, H.golden.extract_block_a_competition(vm, vb))
        X = np.hstack([X, rr_va[:, None]]).astype(np.float32)
        return M["stage2"].predict_proba(X)[:, 1]

    def evaluate(p, tag):
        A = collections.defaultdict(set)
        for (s, c, y), q in zip(vm, p):
            if q >= th:
                A[s].add(c)
        out = {}
        for ctry in ("US", "India"):
            ids = [s for s in D["val_s1_ids"] if country[s] == ctry]
            sc = [f05(set(D["s1_dict"][s]["gt"]), A.get(s, set())) for s in ids]
            fp = sum(len(A.get(s, set()) - set(D["s1_dict"][s]["gt"])) for s in ids)
            tp = sum(len(A.get(s, set()) & set(D["s1_dict"][s]["gt"])) for s in ids)
            out[ctry] = (100 * np.mean(sc), tp, fp)
        print(f"{tag:<32} US {out['US'][0]:.2f} (TP {out['US'][1]} FP {out['US'][2]}) | "
              f"India {out['India'][0]:.2f} (TP {out['India'][1]} FP {out['India'][2]})")
        return out

    p0 = score(LF)
    ref = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl"), "rb"))["p_va"]
    print("reproduction max |dp| vs production valpreds:", float(np.max(np.abs(p0 - ref))))
    evaluate(p0, "E018C as deployed")
    L1 = LF.copy(); L1[is_ind, 17] = 0.0
    evaluate(score(L1), "India rows encoded as US (=France)")
    L2 = LF.copy(); L2[~is_ind, 17] = 1.0
    evaluate(score(L2), "US rows encoded as India")


if __name__ == "__main__":
    main()
