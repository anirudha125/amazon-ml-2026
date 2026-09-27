"""
fetch_raw_text.py -- collect RAW (un-normalized) name/address for every S1 in the 3,995 sample and
every candidate in the frozen E008 pool. Output: experiments/_shared/raw_text.pkl {id: (name, addr, country)}.
Label-free: only reads source tables.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

TRAIN = os.path.join(H.ROOT, "student_resource", "dataset", "train")
OUT = os.path.join(H.SHARED, "raw_text.pkl")

def main():
    with open(H.golden.CAND_CACHE_FILE, "rb") as f:
        cd = pickle.load(f)
    need = set(cd["s1_dict"])
    for s, pool in cd["candidates_by_s1"].items():
        need.update(pool)
    del cd
    raw = {}
    for fn in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(TRAIN, fn), encoding="utf-8") as f:
            f.readline()
            for line in f:
                i = line.find("\t")
                if line[:i] in need:
                    p = line.rstrip("\r\n").split("\t")
                    p += [""] * (4 - len(p))
                    raw[p[0]] = (p[1], p[2], p[3])
        print(fn, len(raw), flush=True)
    missing = len(need - set(raw))
    print("needed", len(need), "found", len(raw), "missing", missing)
    with open(OUT, "wb") as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
