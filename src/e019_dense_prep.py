"""
E019 -- Dense retrieval diagnostic, CPU prep (no labels are used to build anything; GT is stored for evaluation only).

Builds, for train US/India S2/S3 (all records of the country, same partition as lexical retrieval) and the 3,995 RECON sample S1:
  raw text "query: <name>, <addr>" (E017 format), tokenized with the multilingual-e5-small tokenizer (max 64 tokens).
Outputs experiments/E019_dense/:
  docs_<country>_S<src>.npz  (ids, flat token ids int32, offsets int64)
  queries.npz                (sample S1 ids / country, flat tokens, offsets)
  meta.pkl                   (per sample S1: country, gt set, frozen pool ids, val flag)
Also pre-downloads the pinned model (intfloat/multilingual-e5-small @ 614241f6, MIT; see DOWNLOAD_LOG E017) into the HF cache.
"""
import os, sys, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

OUT = os.path.join(H.ROOT, "experiments", "E019_dense"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(H.ROOT, "student_resource", "dataset", "train")
MODEL, REV, MAXLEN = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", 64
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def fmt(name, addr):
    name, addr = name.strip(), addr.strip()
    return "query: " + (name + ", " + addr if addr else name)


def tokenize(tok, texts, bs=200000):
    flat, lens = [], []
    for i in range(0, len(texts), bs):
        for e in tok.encode_batch(texts[i:i + bs]):
            flat.append(np.asarray(e.ids, dtype=np.int32)); lens.append(len(e.ids))
    offs = np.zeros(len(lens) + 1, np.int64); np.cumsum(lens, out=offs[1:])
    return np.concatenate(flat), offs


def main():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    t0 = time.time()
    mdir = snapshot_download(MODEL, revision=REV, allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model", "*.txt"])
    log("model snapshot", mdir, f"{time.time()-t0:.0f}s")
    tok = Tokenizer.from_file(os.path.join(mdir, "tokenizer.json"))
    tok.enable_truncation(MAXLEN); tok.no_padding()

    fd = pickle.load(open(H.golden.CACHED_FEATS_07, "rb")); cd = pickle.load(open(H.golden.CAND_CACHE_FILE, "rb"))
    s1d, val = fd["s1_dict"], set(fd["val_s1_ids"]); cands = cd["candidates_by_s1"]
    sample = list(s1d)
    meta = {s: dict(country=s1d[s]["country"], gt=set(s1d[s]["gt"]), frozen=set(cands[s]), val=s in val) for s in sample}
    raw = {}
    with open(os.path.join(DATA, "train_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip(chr(13) + NL).split(TAB)
            if p[0] in meta:
                p += [""] * (4 - len(p)); raw[p[0]] = fmt(p[1], p[2])
    assert len(raw) == len(sample)
    qflat, qoffs = tokenize(tok, [raw[s] for s in sample])
    np.savez(os.path.join(OUT, "queries.npz"), ids=np.array(sample), country=np.array([meta[s]["country"] for s in sample]),
             flat=qflat, offs=qoffs)
    pickle.dump(meta, open(os.path.join(OUT, "meta.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(f"queries {len(sample)} (val {len(val)}), mean tokens {np.diff(qoffs).mean():.1f}")

    for src in ["2", "3"]:
        by = {"US": ([], []), "India": ([], [])}
        t = time.time()
        with open(os.path.join(DATA, f"train_source{src}.tsv"), encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip(chr(13) + NL).split(TAB); p += [""] * (4 - len(p))
                if p[3] in by:
                    by[p[3]][0].append(p[0]); by[p[3]][1].append(fmt(p[1], p[2]))
        log(f"S{src} read {time.time()-t:.0f}s", {c: len(v[0]) for c, v in by.items()})
        for c, (ids, texts) in by.items():
            t = time.time(); flat, offs = tokenize(tok, texts); dt = time.time() - t
            np.savez(os.path.join(OUT, f"docs_{c}_S{src}.npz"), ids=np.array(ids), flat=flat, offs=offs)
            log(f"  {c} S{src}: {len(ids):,} docs tokenized in {dt:.0f}s ({len(ids)/dt:,.0f}/s), mean tokens {np.diff(offs).mean():.1f}, "
                f"truncated {(np.diff(offs) >= MAXLEN).mean()*100:.2f}%")
        del by
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
