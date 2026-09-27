"""E027 text format: 'name | addr' with an explicit missing-address token (train, val and test all use this)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
NOADDR = "[no address]"


def fmt2(nm, ad):
    return (nm or "").strip() + " | " + ((ad or "").strip() or NOADDR)


def train_texts(ids):
    import pickle, e021_foundation as F
    need = set(ids); out = {}
    s1 = pickle.load(open(os.path.join(ROOT, "experiments", "E021", "s1.pkl"), "rb"))
    for s in need:
        if s.startswith("S1-"):
            r = s1[s]; out[s] = fmt2(r["raw_name"], r["raw_addr"])
    for src in "23":
        ids_, nm, ad, _ = F.read_table("train", src)
        for i, n_, a_ in zip(ids_, nm, ad):
            if i in need:
                out[i] = fmt2(n_, a_)
    miss = need - set(out); assert not miss, f"{len(miss)} ids without text"
    return out


def test_texts(country):
    import e021_foundation as F
    tx = {}
    for src in "123":
        i2, n2, a2, _ = F.read_table("test", src, country); tx.update({i: fmt2(n, a) for i, n, a in zip(i2, n2, a2)})
    return tx
