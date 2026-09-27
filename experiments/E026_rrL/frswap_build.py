"""S006_FRswap hedge (built, NOT submitted): S006 rows for US + India, S005 rows for France, in both matching_results.tsv and
candidate_pairs.tsv (each row keeps its own submission's candidate list, so predictions ⊆ candidates holds). Pattern of
src/probe_frswap.py. Inputs (read-only, sha256-verified before use): S006 = experiments/P3_rrL/submission_S006_rrL_mc,
S005 = experiments/RL/submissions/S005_RL27NEW_s42_mc.
(Fixed after the 09:12 build: verify() originally skipped S005 because its SHA256SUMS uses repo-relative paths; S005 was
then verified with `sha256sum -c` from the repo root: OK.) Max-claimer is per country and records never cross countries, so
mixing countries from two max-claimed submissions keeps every record claimed by at most one S1.
Checks: US/India rows byte-identical to S006, France rows byte-identical to S005, row set complete, same S1 order in both files.
Output: experiments/P3_rrL/submission_S006_FRswap_mc/ (+ SHA256SUMS, MANIFEST.json, deltas.json). Refuses to overwrite.
"""
import os, json, hashlib, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S6 = os.path.join(ROOT, "experiments", "P3_rrL", "submission_S006_rrL_mc")
S5 = os.path.join(ROOT, "experiments", "RL", "submissions", "S005_RL27NEW_s42_mc")
OUT = os.path.join(ROOT, "experiments", "P3_rrL", "submission_S006_FRswap_mc")
FILES = ["matching_results.tsv", "candidate_pairs.tsv"]
TAB, NL = chr(9), chr(10)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def verify(d):
    n = 0
    for line in open(os.path.join(d, "SHA256SUMS")):
        h, fn = line.split()
        fn = os.path.basename(fn)          # S005's SHA256SUMS lists repo-relative paths, S006's bare names
        if fn in FILES:
            assert sha(os.path.join(d, fn)) == h, f"sha mismatch {d}/{fn}"; n += 1
    assert n == len(FILES), f"{d}: only {n} of {len(FILES)} files verified"


ctry = {}
with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip(chr(13) + NL).split(TAB); ctry[p[0]] = p[3]

verify(S6); verify(S5)
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"; os.makedirs(OUT)
stats = {"US": 0, "India": 0, "France": 0}; order = {}
for fname in FILES:
    with open(os.path.join(S6, fname), encoding="utf-8") as f6, open(os.path.join(S5, fname), encoding="utf-8") as f5, \
         open(os.path.join(OUT, fname), "w", encoding="utf-8", newline="") as fo:
        h6, h5 = f6.readline(), f5.readline(); assert h6 == h5; fo.write(h6)
        seen = []
        for line in f6:                                 # US + India rows from S006, in S006 order
            s = line.split(TAB, 1)[0]
            if ctry[s] != "France":
                fo.write(line); seen.append(s)
        for line in f5:                                 # France rows from S005, in S005 order
            s = line.split(TAB, 1)[0]
            if ctry[s] == "France":
                fo.write(line); seen.append(s)
        assert len(seen) == len(set(seen)) == len(ctry) and set(seen) == set(ctry), (fname, len(seen))
        order[fname] = seen
        if fname == "matching_results.tsv":
            for s in seen:
                stats[ctry[s]] += 1
assert order[FILES[0]] == order[FILES[1]], "S1 order differs between the two files"


def read(d, fn):
    out = {}
    with open(os.path.join(d, fn), encoding="utf-8") as f:
        f.readline()
        for line in f:
            out[line.split(TAB, 1)[0]] = line
    return out


# byte-identity checks (both files)
ident = {}
for fn in FILES:
    o, r6, r5 = read(OUT, fn), read(S6, fn), read(S5, fn)
    ident[fn] = dict(us_india_rows_not_identical_to_S006=sum(o[s] != r6[s] for s in o if ctry[s] != "France"),
                     france_rows_not_identical_to_S005=sum(o[s] != r5[s] for s in o if ctry[s] == "France"))
    assert all(v == 0 for v in ident[fn].values()), (fn, ident[fn])
    del o, r6, r5


def links(d):
    m = read(d, "matching_results.tsv")
    return {s: set(filter(None, l.rstrip(NL).split(TAB)[1].split(","))) for s, l in m.items()}


mo, m6, m5 = links(OUT), links(S6), links(S5)
deltas = {}
for ref, mr in (("vs_S005", m5), ("vs_S006", m6)):
    t = collections.defaultdict(collections.Counter)
    for s, a in mo.items():
        b = mr[s]; c = t[ctry[s]]
        c["rows_differ"] += a != b; c["links_added"] += len(a - b); c["links_removed"] += len(b - a); c["pred_pairs"] += len(a); c["ref_pred_pairs"] += len(b)
    tot = collections.Counter()
    for c in t.values():
        tot.update(c)
    t["ALL"] = tot
    deltas[ref] = {k: dict(v) for k, v in t.items()}
hashes = {fn: sha(os.path.join(OUT, fn)) for fn in FILES}
with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
    for fn, h in hashes.items():
        f.write(f"{h}  {fn}{NL}")
json.dump(dict(probe="S006_FRswap", status="BUILT, NOT SUBMITTED", us_india_rows_from="S006 " + S6, france_rows_from="S005 " + S5,
               rows=stats, sha256=hashes, byte_identity=ident,
               purpose="LB(S006) - LB(S006_FRswap) = France share * (F_France(S006) - F_France(S005)); "
                       "LB(S006_FRswap) - LB(S005) = US+India effect of rrL"),
          open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
json.dump(deltas, open(os.path.join(OUT, "deltas.json"), "w"), indent=1)
print(json.dumps(dict(rows=stats, sha256=hashes, byte_identity=ident, deltas=deltas), indent=1))
