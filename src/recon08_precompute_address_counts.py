import os, sys, time, pickle

DATA_DIR = r"d:\amazon-ml-challenge-2026\student_resource\dataset\train"
SCRATCH_DIR_05 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\b15afa6e-4b1b-492d-ba1c-a5c13926ce28\scratch"
SCRATCH_DIR_08 = r"C:\Users\Anirudha Thakur\.gemini\antigravity-ide\brain\3724a200-73fc-4663-a405-ac5d9bf69da2\scratch"

CAND_CACHE_FILE = os.path.join(SCRATCH_DIR_05, "recon05_candidates.pkl")
ADDR_CACHE_FILE = os.path.join(SCRATCH_DIR_08, "recon08_addr_counts.pkl")

def main():
    if os.path.exists(ADDR_CACHE_FILE):
        print(f"Address counts already cached at {ADDR_CACHE_FILE}")
        return

    print("Loading candidate addresses...")
    with open(CAND_CACHE_FILE, "rb") as f:
        cand_data = pickle.load(f)
    cands = cand_data["candidates_by_s1"]

    target_addrs = set()
    for s1, cdict in cands.items():
        for cid, cinfo in cdict.items():
            ad = cinfo["addr"]
            if ad and ad != "null":
                target_addrs.add(ad)
    print(f"Target unique candidate addresses: {len(target_addrs):,}")

    # Build translation table matching normalize()
    # remove punctuation, collapse whitespace
    import re
    def normalize(text):
        if not text or text == "null": return ""
        t = text.lower()
        t = re.sub(r"[^\w\s-]", " ", t)
        t = re.sub(r"-+", " ", t)
        t = re.sub(r"\bnull\b", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    s2_counts = {}
    s3_counts = {}

    t0 = time.time()
    p2 = os.path.join(DATA_DIR, "train_source2.tsv")
    print("Streaming S2...")
    with open(p2, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 2:
                ad = normalize(p[2])
                if ad in target_addrs:
                    s2_counts[ad] = s2_counts.get(ad, 0) + 1
            if (i + 1) % 1000000 == 0:
                print(f"  S2: {i+1:,} lines in {time.time()-t0:.1f}s...")

    t1 = time.time()
    p3 = os.path.join(DATA_DIR, "train_source3.tsv")
    print("Streaming S3...")
    with open(p3, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 2:
                ad = normalize(p[2])
                if ad in target_addrs:
                    s3_counts[ad] = s3_counts.get(ad, 0) + 1
            if (i + 1) % 1000000 == 0:
                print(f"  S3: {i+1:,} lines in {time.time()-t1:.1f}s...")

    print(f"Done! S2 matches: {len(s2_counts):,} | S3 matches: {len(s3_counts):,}")
    with open(ADDR_CACHE_FILE, "wb") as f:
        pickle.dump({"s2_counts": s2_counts, "s3_counts": s3_counts}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {ADDR_CACHE_FILE}")

if __name__ == "__main__":
    main()
