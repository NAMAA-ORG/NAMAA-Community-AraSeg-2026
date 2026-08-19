"""Force a document-final boundary in submission CSVs, and repackage.

A boundary label marks a segment END, so the last segment of a document must close
on its last token. Every one of the 2632 labeled AraSeg documents (4 heads x
train/dev/test) does. Our decoders never enforced it and emit structurally
impossible output -- a document whose final segment never closes -- on 28 of the
624 blind documents.

Setting pred[-1]=1 where gold[-1]=1 converts a false negative to a true positive
and cannot create a false positive, so the change is non-negative by construction.
Verified with the official scorer on PA test: 94.49 -> 94.56.

Legality: derivable from TRAIN alone (174/174), and nothing is fitted -- it is a
property of the annotation scheme, not a learned parameter. Strictly weaker than
the dev-tuned thresholds already inside every lock.

Writes a NEW submission folder; the originals are never touched.

Usage:
    python force_final_boundary.py --src ../submissions/blind-phase-20260720 \
                                   --dst ../submissions/blind-phase-20260723-forced
    python force_final_boundary.py --verify-labeled     # re-check the invariant
"""
import argparse
import csv
import shutil
import zipfile
from pathlib import Path

HEADS = ["PA", "NoPnx-PA", "NP", "NoPnx-NP"]


def verify_labeled():
    """The invariant this whole script rests on. Fails loudly if it ever breaks."""
    from datasets import load_dataset
    bad = 0
    for h in HEADS:
        ds = load_dataset(f"MBZUAI/AraSeg-2026-Shared-Task-{h}")
        for sp in ds:
            n = len(ds[sp])
            ends = sum(1 for x in ds[sp]["labels"] if x and x[-1] == 1)
            flag = "" if ends == n else "  <-- INVARIANT BROKEN"
            bad += n - ends
            print(f"  {h:10} {sp:6} {ends}/{n} end with a gold boundary{flag}")
    assert bad == 0, f"{bad} documents do NOT end with a boundary -- do NOT apply this fix"
    print("\nINVARIANT HOLDS on every labeled document.")


def force_csv(src: Path, dst: Path):
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    changed = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Document ID", "Prediction"])
        for r in rows:
            s = r["Prediction"].strip()
            if s[-1] != "1":
                changed += 1
                s = s[:-1] + "1"
            w.writerow([r["Document ID"], s])
    return len(rows), changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="existing submission folder (<head>/prediction.zip)")
    ap.add_argument("--dst", help="NEW folder to write; must not exist")
    ap.add_argument("--verify-labeled", action="store_true")
    ap.add_argument("--csv", help="single-head mode: a decoder's test_predictions_*.csv")
    ap.add_argument("--out-zip", dest="out_zip", help="single-head mode: prediction.zip to write")
    args = ap.parse_args()

    if args.verify_labeled:
        verify_labeled()
        return

    # Single-head mode. The --src/--dst path re-packages an existing four-head
    # submission; a re-lock on ONE head has no such folder, only a fresh CSV from
    # fit_decoder. Same force_csv, same packaging assertion -- the bare,
    # extensionless `prediction` member is what the dev-phase uploads all failed on.
    if args.csv:
        assert args.out_zip, "--csv needs --out-zip"
        out = Path(args.out_zip)
        assert not out.exists(), f"{out} exists -- refusing to overwrite a submission"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.parent / "prediction"
        n, ch = force_csv(Path(args.csv), tmp)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(tmp, arcname="prediction")
        tmp.unlink()
        with zipfile.ZipFile(out) as z:
            assert z.namelist() == ["prediction"], f"repack broke {out}"
        print(f"  {n} docs, forced {ch} final boundaries -> {out}")
        return

    src, dst = Path(args.src), Path(args.dst)
    assert not dst.exists(), f"{dst} exists -- refusing to overwrite a submission folder"

    total_rows = total_changed = 0
    for head in HEADS:
        zin = src / head / "prediction.zip"
        assert zin.exists(), f"missing {zin}"
        work = dst / head
        work.mkdir(parents=True)

        with zipfile.ZipFile(zin) as z:
            names = z.namelist()
            assert names == ["prediction"], (
                f"{zin} holds {names}, expected exactly ['prediction'] -- that bare, "
                f"extensionless member is the packaging the dev-phase uploads all failed on")
            raw = work / "prediction_in.csv"
            raw.write_bytes(z.read("prediction"))

        out = work / "prediction"
        n, ch = force_csv(raw, out)
        raw.unlink()
        total_rows += n
        total_changed += ch

        # Repack: one member, named `prediction`, no extension, no directory prefix.
        zout = work / "prediction.zip"
        with zipfile.ZipFile(zout, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out, arcname="prediction")
        out.unlink()

        with zipfile.ZipFile(zout) as z:
            assert z.namelist() == ["prediction"], f"repack broke {zout}"
        print(f"  {head:10} {n:4d} docs, forced {ch:3d} final boundaries -> {zout}")

    print(f"\n{total_changed} of {total_rows} documents changed. Originals untouched at {src}")


if __name__ == "__main__":
    main()
