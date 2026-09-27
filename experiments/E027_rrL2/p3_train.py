"""E027 p3: continuation of rrL (E026 model_rrL weights) on train_pairs.pkl (new TDa hard mix + TR replay), text = fmt2
(explicit missing-address token), 1 epoch, bs 256 (2x128), lr 2e-5 OneCycle, AdamW wd .01, bf16, seed 42, max 128 tokens."""
import os, sys, json, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E026_rrL"))
import rrL_lib as L, rr2_lib as T2
LR = float(os.environ.get("LR", 2e-5))
def main():
    d = pickle.load(open(os.path.join(HERE, "train_pairs.pkl"), "rb")); pairs, y = d["pairs"], d["y"]
    tx = T2.train_texts({x for p in pairs for x in p}); L.log(f"texts {len(tx):,}; pairs {len(pairs):,} pos {int(y.sum()):,}")
    info = L.train([tx[a] for a, _ in pairs], [tx[b] for _, b in pairs], y, os.path.join(HERE, "model_rrL2"),
                   init=os.path.join(ROOT, "experiments", "E026_rrL", "model_rrL"), rev=None, bs=256, micro=128, lr=LR, epochs=1.0, seed=42, n_tok=4)
    L.log("p3 done", json.dumps(info))


if __name__ == "__main__":
    main()
