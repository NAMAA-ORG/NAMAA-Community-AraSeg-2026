"""Prove a locked head reproduces off-cluster, from the AraSeg-backup checkpoints.

The blind run (Jul 20-25) will execute on rented GPU, not KISSKI. Nothing has ever
loaded these checkpoints anywhere else. This reproduces a locked head end-to-end and
diffs the CSV against the submitted one -- byte-identical means the blind pipeline is
portable and KISSKI is optional.

Only two things differ off-cluster, and this patches exactly those:
  1. `output_dir` -- where best_<HEAD>.pt lives (KISSKI outputs/ -> backup root)
  2. `model_name` -- e32/e33 point at KISSKI-local base-model dirs, which do not
     exist elsewhere; off-cluster they must resolve to HF repo ids.
Everything else runs the real ensemble.py path, so a pass means the real thing works.

VRAM: members are built one at a time, so peak = largest single member. PA's e33
(Gemma-4-12B, bf16) needs ~24GB of weights + activations -> a >=40GB card. A 16GB
T4 cannot run this, and has no bf16 at all.

Usage (RunPod A100, from experiments/):
    python verify_offcluster.py --subtask PA \
        --ckpt-root /workspace/AraSeg-backup \
        --base-model e32=Qwen/Qwen3.5-9B --base-model e33=google/gemma-4-12b \
        --expect outputs/phaseC_v2_pa_logit/test_predictions_PA.csv
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# The four locks, exactly as recorded in docs/handoff.md. Reproducing a head means
# reproducing ITS combiner and ITS frozen threshold -- not a fresh sweep.
LOCKS = {
    "PA": dict(
        members=["e25_xlmr_single_pa_stride256", "e32_qwen35_9b_pa", "e33_gemma4_12b_pa"],
        combine="logit", threshold=0.25, sat_cache=None,
    ),
    # The other three heads are decoder/stack locks -- they are NOT reproduced by
    # ensemble.py alone (they need fit_decoder.py / the saved stacker weights), so
    # they are deliberately not wired up here. PA is the portability probe: it is
    # the head under threat, the cheapest, and it exercises both the full-checkpoint
    # (e25) and LoRA-adapter (e32/e33) load paths, which is all we need to prove.
}


def patch_config(src: Path, ckpt_root: Path, base_override: str | None, dst_dir: Path) -> Path:
    cfg = yaml.safe_load(src.read_text())
    exp_id = cfg["experiment_id"]

    # Checkpoints: outputs/e25 -> <backup>/e25
    cfg["output_dir"] = str(ckpt_root / exp_id)

    if base_override:
        cfg["model_name"] = base_override
    # Do not use Path(...).is_absolute(): on Windows it is false for Unix absolute
    # paths, so that guard could silently pass and fail deep inside ensemble.py.
    elif cfg["model_name"].startswith("/"):
        sys.exit(
            f"{exp_id}: model_name is a site-local path ({cfg['model_name']}) and no "
            f"--base-model {exp_id}=<hf/repo-id> was given. resolve_model_path() will "
            f"raise FileNotFoundError off-cluster. Pass the HF id."
        )

    ckpt = Path(cfg["output_dir"])
    if not ckpt.is_dir():
        sys.exit(f"{exp_id}: no checkpoint dir at {ckpt}")

    dst = dst_dir / src.name
    dst.write_text(yaml.safe_dump(cfg))
    print(f"  [cfg] {exp_id}: ckpt={cfg['output_dir']}  base={cfg['model_name']}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", default="PA", choices=sorted(LOCKS))
    ap.add_argument("--ckpt-root", required=True, type=Path,
                    help="Backup root holding e25/, e32/, ... (i.e. the mirrored outputs/)")
    ap.add_argument("--base-model", action="append", default=[], metavar="EXPID=HF_ID",
                    help="Override a member's base model, e.g. e32=Qwen/Qwen3.5-9B. Repeatable.")
    ap.add_argument("--expect", type=Path, default=None,
                    help="The submitted CSV this must reproduce byte-for-byte. Omit on the "
                         "blind split (nothing to diff against — the CSV is the deliverable).")
    ap.add_argument("--test-split", dest="test_split", default="test",
                    help="'test' (public, verify) or 'blind' (Testing Phase — writes the "
                         "submission CSV, no diff). Needs HF_TOKEN for the gated -Blind repo.")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/verify_offcluster"))
    args = ap.parse_args()
    if args.test_split == "test" and args.expect is None:
        ap.error("--expect is required when --test-split test (there is a CSV to diff against)")

    lock = LOCKS[args.subtask]
    overrides = dict(kv.split("=", 1) for kv in args.base_model)

    print(f"[verify] {args.subtask}: {lock['members']} combine={lock['combine']} "
          f"thr={lock['threshold']}")

    tmp = Path(tempfile.mkdtemp(prefix="verify_cfg_"))
    try:
        cfgs = [
            patch_config(Path("configs") / f"{m}.yaml", args.ckpt_root,
                         overrides.get(m.split("_")[0]), tmp)
            for m in lock["members"]
        ]

        cmd = [
            sys.executable, "ensemble.py",
            "--subtask", args.subtask,
            "--members", *map(str, cfgs),
            "--combine", lock["combine"],
            "--threshold_policy", "fixed",      # the lock's threshold is FROZEN --
            "--threshold", str(lock["threshold"]),  # re-tuning it here would be a leak
            "--test-split", args.test_split,
            "--output_dir", str(args.output_dir),
        ]
        if lock["sat_cache"]:
            cmd += ["--sat_cache", lock["sat_cache"]]

        print(f"[verify] $ {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    got = args.output_dir / f"test_predictions_{args.subtask.replace('-', '_')}.csv"

    if args.expect is None:   # blind split — nothing to diff, the CSV is the submission
        n = len(got.read_text().strip().splitlines()) - 1
        print(f"\n[verify] {args.test_split} predictions written: {got} ({n} docs).")
        print(f"[verify] Package for Codabench: copy to 'prediction' (no extension), zip → prediction.zip")
        return 0

    a, b = args.expect.read_text().strip(), got.read_text().strip()
    if a == b:
        print(f"\n[verify] PASS -- {got} is byte-identical to {args.expect}.")
        print("[verify] The blind pipeline is portable: these checkpoints reproduce the "
              "lock off-cluster. KISSKI is optional.")
        return 0

    # Not identical -- say exactly how far off, because "a few docs differ" (nondeterminism)
    # and "everything differs" (wrong base weights) need completely different responses.
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        print(f"\n[verify] FAIL -- row count differs: expected {len(la)}, got {len(lb)}")
        return 1
    bad = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    print(f"\n[verify] FAIL -- {len(bad)}/{len(la)-1} documents differ.")
    print("[verify] A handful of docs => GPU nondeterminism; re-score with official_eval.py "
          "and accept if F1 matches to 2dp.")
    print("[verify] Most or all docs => the base-model weights are NOT the ones the adapters "
          "were trained against. Mirror the bases off KISSKI instead of pulling from HF.")
    for i in bad[:3]:
        print(f"    line {i}:\n      expected {la[i][:90]}\n      got      {lb[i][:90]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
