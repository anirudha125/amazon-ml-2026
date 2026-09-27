"""E027 p4: rrL2 logits. `val` -> T/V0/V1 top-10 pairs of E026 (cache/top10_pairs.pkl, read-only) -> rr_rrL2_top10.pkl;
`test <countries>` -> P3_rrL/top10_<c>.pkl pairs -> rrcache_rrL2_<c>.pkl (all in experiments/E027_rrL2)."""
import os, sys, json, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E026_rrL"))
import rrL_lib as L, rr2_lib as T2
MD = os.path.join(HERE, "model_rrL2")
def main():
    ph = sys.argv[1]
    if ph == "val":
        P = pickle.load(open(os.path.join(ROOT, "experiments", "E026_rrL", "cache", "top10_pairs.pkl"), "rb"))["all"]
        tx = T2.train_texts({x for p in P for x in p})
        sc, info = L.score(MD, [tx[a] for a, _ in P], [tx[b] for _, b in P], bs=1024, n_tok=6)
        assert np.isfinite(sc).all()
        pickle.dump(dict(zip(P, sc.tolist())), open(os.path.join(HERE, "rr_rrL2_top10.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        json.dump(info, open(os.path.join(HERE, "score_val_info.json"), "w"))
    else:
        for c in sys.argv[2:]:
            out = os.path.join(HERE, f"rrcache_rrL2_{c}.pkl")
            if os.path.exists(out): continue
            P = pickle.load(open(os.path.join(ROOT, "experiments", "P3_rrL", f"top10_{c}.pkl"), "rb")); tx = T2.test_texts(c)
            sc, info = L.score(MD, [tx[a] for a, _ in P], [tx[b] for _, b in P], bs=1024, n_tok=6)
            assert np.isfinite(sc).all()
            pickle.dump(dict(zip(P, sc.tolist())), open(out + ".tmp", "wb"), protocol=pickle.HIGHEST_PROTOCOL); os.replace(out + ".tmp", out)
            info["country"] = c; json.dump(info, open(os.path.join(HERE, f"score_test_info_{c}.json"), "w")); L.log("test", c, json.dumps(info))


if __name__ == "__main__":
    main()
