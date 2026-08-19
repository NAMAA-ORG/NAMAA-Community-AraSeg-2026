"""Per-cluster error breakdown for the paper's RQ3 (CPU-only, no torch, no GPU).

Answers "where does the precision deficit live?" by scoring each locked system inside
TF-IDF/KMeans genre-proxy clusters instead of over all 262 test documents at once.

Two facts make this comparison clean:
  * all four subtasks share the SAME 262 test doc_ids (verified: the CSVs' ID columns are
    identical), so ONE cluster map built on PA text applies to all four heads. The cluster
    is a property of the document, not of the cue-deletion condition.
  * document macro-F1 is a plain mean over documents, so the size-weighted mean of the
    per-cluster macros reconstructs the overall lock score exactly -- which is what
    --selfcheck asserts.

Provenance, and it matters: the newest PRACTICE-TEST CSVs on this machine are from
2026-07-13. PA and NP are the submitted locks; both NoPnx CSVs are the decoder locks that
the 07-24 e70 and 08-02 joint re-locks later superseded (87.18 vs submitted 87.82, 85.89
vs 86.49). The script prints the macro it reconstructs for every head so the version in
use is never in doubt -- do not quote a NoPnx row as the submitted system.

Usage:
  python genre_proxy.py --subtask PA --split test --n_clusters 5   # once, builds the map
  python error_by_cluster.py
  python error_by_cluster.py --selfcheck
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bootstrap_paired import read_gold, read_preds
from scoring import _doc_f1, _macro

# (subtask, locked prediction CSV, macro-F1 this CSV is known to score)
LOCKS = [
    ("PA",       "outputs/phaseC_v2_pa_logit/test_predictions_PA.csv",      0.9449),
    ("NP",       "outputs/oof_stack_sat_np/test_predictions_NP.csv",        0.9284),
    ("NoPnx-PA", "outputs/decoder_nopnx_pa/test_predictions_NoPnx_PA.csv",  0.8718),
    ("NoPnx-NP", "outputs/decoder_nopnx_np/test_predictions_NoPnx_NP.csv",  0.8589),
]


def cluster_scores(subtask, csv_path, clusters, split="test"):
    """{cluster: macro P/R/F1} plus the overall macro, from a locked submission CSV."""
    gold, preds = read_gold(subtask, split), read_preds(csv_path)
    by_cluster, everything = defaultdict(list), []
    for doc_id in sorted(gold):
        s = _doc_f1(preds[doc_id], gold[doc_id])
        by_cluster[clusters[doc_id]].append(s)
        everything.append(s)
    return {c: _macro(v) for c, v in by_cluster.items()}, _macro(everything)


def cluster_ci(subtask, csv_path, clusters, target, split="test", n=10000, seed=42):
    """Percentile CI for ONE cluster's macro-F1. Two clusters here are n=18/20, so a
    point estimate on its own invites exactly the attack this paper's gate exists to
    prevent -- 100.00 on 18 documents is "no errors seen", not "solved"."""
    gold, preds = read_gold(subtask, split), read_preds(csv_path)
    f1 = np.array([_doc_f1(preds[d], gold[d])["f1"]
                   for d in sorted(gold) if clusters[d] == target])
    boot = f1[np.random.default_rng(seed).integers(0, len(f1), size=(n, len(f1)))].mean(axis=1)
    return f1.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(f1)


def samples(clusters, target, split="test", k=2, words=12):
    """First few words of k documents in a cluster -- so clusters get named from their
    text, not guessed at from their length statistics."""
    from datasets import load_dataset
    ds = load_dataset("MBZUAI/AraSeg-2026-Shared-Task-PA", split=split)
    out = []
    for r in ds:
        if clusters[str(r["doc_id"])] == target:
            out.append(" ".join(r["tokens"][:words]))
            if len(out) == k:
                break
    return out


def describe(clusters, split="test"):
    """{cluster: (n_docs, mean tokens, boundary density, mean segment length)} from gold.

    Gold labels alone carry all of it: len() is the document's token count and sum() its
    boundary count, so naming a cluster ("short-unit" vs "long-form prose") costs no extra
    dataset pass.
    """
    gold = read_gold("PA", split)
    acc = defaultdict(list)
    for doc_id, g in gold.items():
        acc[clusters[doc_id]].append((len(g), int(g.sum())))
    out = {}
    for c, rows in acc.items():
        toks = np.array([r[0] for r in rows], dtype=float)
        bnds = np.array([r[1] for r in rows], dtype=float)
        out[c] = (len(rows), toks.mean(), (bnds / toks).mean(),
                  float(toks.sum() / max(bnds.sum(), 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="outputs/genre_proxy/pa_test.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ci", nargs="*", default=None,
                    help="cluster ids to bootstrap a 95%% CI for, on every head")
    ap.add_argument("--samples", nargs="*", default=None,
                    help="cluster ids to print sample text for")
    args = ap.parse_args()

    clusters = json.loads(Path(args.clusters).read_text(encoding="utf-8"))
    desc = describe(clusters, args.split)
    order = sorted(desc, key=lambda c: -desc[c][0])

    print(f"\nGenre-proxy clusters ({args.split}, n={sum(d[0] for d in desc.values())} docs)")
    print(f"{'cluster':>8} {'n':>5} {'mean tok':>9} {'bnd dens':>9} {'mean seg':>9}")
    for c in order:
        n, tok, dens, seg = desc[c]
        print(f"{c:>8} {n:>5} {tok:>9.0f} {dens:>9.4f} {seg:>9.1f}")

    for subtask, csv_path, expected in LOCKS:
        if not Path(csv_path).exists():
            print(f"\n{subtask}: MISSING {csv_path} -- skipped")
            continue
        per_c, overall = cluster_scores(subtask, csv_path, clusters, args.split)
        flag = "" if abs(overall["f1"] - expected) < 5e-4 else "  <-- NOT the expected lock!"
        print(f"\n{subtask}  ({csv_path})")
        print(f"  overall  P {100*overall['precision']:.2f}  R {100*overall['recall']:.2f}"
              f"  F1 {100*overall['f1']:.2f}   [expected {100*expected:.2f}]{flag}")
        print(f"  {'cluster':>8} {'n':>5} {'P':>7} {'R':>7} {'F1':>7} {'dF1':>7}")
        for c in order:
            s = per_c[c]
            print(f"  {c:>8} {desc[c][0]:>5} {100*s['precision']:>7.2f} {100*s['recall']:>7.2f}"
                  f" {100*s['f1']:>7.2f} {100*(s['f1']-overall['f1']):>+7.2f}")
        for c in (args.ci or []):
            m, lo, hi, n = cluster_ci(subtask, csv_path, clusters, c, args.split)
            print(f"    cluster {c} (n={n}): F1 {100*m:.2f}  95% CI [{100*lo:.2f}, {100*hi:.2f}]")

    for c in (args.samples or []):
        print(f"\ncluster {c} sample text:")
        for s in samples(clusters, c, args.split):
            print(f"  {s}")


def _selfcheck():
    """Per-cluster macros must re-aggregate to the known lock score, size-weighted."""
    clusters = json.loads(Path("outputs/genre_proxy/pa_test.json").read_text(encoding="utf-8"))
    for subtask, csv_path, expected in LOCKS:
        per_c, overall = cluster_scores(subtask, csv_path, clusters)
        sizes = {c: sum(1 for v in clusters.values() if v == c) for c in per_c}
        recon = sum(per_c[c]["f1"] * sizes[c] for c in per_c) / sum(sizes.values())
        assert abs(recon - overall["f1"]) < 1e-9, f"{subtask}: {recon} != {overall['f1']}"
        assert abs(overall["f1"] - expected) < 5e-4, \
            f"{subtask}: scored {overall['f1']:.4f}, expected {expected:.4f}"
        print(f"OK {subtask}: clusters re-aggregate to {100*overall['f1']:.2f}")
    print("OK: partition is exhaustive and every CSV is the lock it claims to be")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
