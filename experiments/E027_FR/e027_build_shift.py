"""E027_FR follow-up (POST-HOC): build submission_S007FR_shift = S006 with every S006-accepted street-equal France link at a
shift-set offset removed for classes A, D, X, Y (pool and accepted shift/mirror >= 5x), empty guard on. Derived from e027_build.py."""
import os, json, hashlib, shutil, collections, sys
S6 = "../P3_rrL/submission_S006_rrL_mc"; OUT = "submission_S007FR_shift"
BUCKETS = ["POSTHOC_shift_drop"]
_sh = json.load(open("shift.json")); mv = {"POSTHOC_shift_drop": dict(pairs=[(s, c) for s, c, _ in _sh["pairs"]], s1_skipped_would_empty=_sh["s1_skipped_would_empty"])}
drop = collections.defaultdict(set); add = collections.defaultdict(list)
for b in BUCKETS:
    for s, c in mv[b]["pairs"]:
        (drop if b.endswith("_drop") else add)[s].add(c) if b.endswith("_drop") else add[s].append(c)
ctry = {}
for l in open("../../student_resource/dataset/test/test_source1.tsv", encoding="utf-8"):
    p = l.rstrip("\r\n").split("\t"); ctry[p[0]] = p[3]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
    return h.hexdigest()
assert not os.path.exists(OUT); os.makedirs(OUT)
shutil.copyfile(os.path.join(S6, "candidate_pairs.tsv"), os.path.join(OUT, "candidate_pairs.tsv"))
st = collections.Counter()
with open(os.path.join(S6, "matching_results.tsv"), "rb") as fi, open(os.path.join(OUT, "matching_results.tsv"), "wb") as fo:
    fo.write(fi.readline())
    for raw in fi:
        line = raw.decode("utf-8"); s, m = line.rstrip("\n").split("\t", 1)
        if ctry[s] != "France" or (s not in drop and s not in add):
            fo.write(raw); continue
        old = [x for x in m.split(",") if x]
        new = [x for x in old if x not in drop.get(s, ())]
        assert new, f"would empty {s}"
        new += [c for c in add.get(s, []) if c not in new]
        st["links_removed"] += len(old) - len([x for x in old if x in new]); st["links_added"] += len(set(new) - set(old))
        st["s1_changed"] += new != old; st["s1_empty_after"] += not new
        fo.write((s + "\t" + ",".join(new) + "\n").encode("utf-8"))
# byte-identity + order checks
ident = collections.Counter()
for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
    with open(os.path.join(S6, fn), "rb") as a, open(os.path.join(OUT, fn), "rb") as b:
        assert a.readline() == b.readline()
        for la, lb in zip(a, b):
            s = la.split(b"\t", 1)[0].decode(); assert s == lb.split(b"\t", 1)[0].decode(), "order"
            if ctry[s] != "France": ident[fn + ":usin_rows"] += 1; ident[fn + ":usin_diff"] += la != lb
            else: ident[fn + ":fr_diff"] += la != lb
        assert a.read(1) == b"" and b.read(1) == b""
assert ident["matching_results.tsv:usin_diff"] == 0 and ident["candidate_pairs.tsv:usin_diff"] == 0 and ident["candidate_pairs.tsv:fr_diff"] == 0
hs = {fn: sha(os.path.join(OUT, fn)) for fn in ("matching_results.tsv", "candidate_pairs.tsv")}
with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
    for fn, h in hs.items(): f.write(f"{h}  {fn}\n")
json.dump(dict(buckets=BUCKETS, status="POST-HOC hedge (shift-set drops for classes A,D,X,Y; see shift.json), NOT pre-registered", stats=st, identity=ident, sha256=hs,
               per_bucket={b: dict(links=len(mv[b]["pairs"]), s1=len(set(p[0] for p in mv[b]["pairs"])), s1_skipped_would_empty=mv[b].get("s1_skipped_would_empty")) for b in BUCKETS}),
          open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
print(json.dumps(dict(stats=st, identity=ident, sha256=hs), indent=1))
