"""
dataset_loader.py
-----------------
Extracts drive_zip.zip, discards empty class folders, and splits every
class into train / valid / test sets.

Output structure:
    hand-sign-drive/
        train/<class>/...
        valid/<class>/...
        test/<class>/...

Usage:
    python dataset_loader.py
    python dataset_loader.py --zip my_file.zip --out my_dataset --train 0.7 --valid 0.15

Splits default to 70 / 15 / 15.
"""

import argparse
import os
import random
import shutil
import tempfile
import zipfile
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Extract zip and create train/valid/test split.")
    p.add_argument("--zip",   default="drive_zip.zip", help="Path to the source zip file.")
    p.add_argument("--out",   default="hand-sign-drive", help="Output dataset directory.")
    p.add_argument("--train", type=float, default=0.70, help="Fraction for training set.")
    p.add_argument("--valid", type=float, default=0.15, help="Fraction for validation set.")
    p.add_argument("--seed",  type=int,   default=42,   help="Random seed for reproducibility.")
    return p.parse_args()


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def split_files(files: list[Path], train_frac: float, valid_frac: float, seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, round(n * train_frac))
    n_valid = max(1, round(n * valid_frac))
    # whatever remains goes to test (at least 1 if possible)
    train = shuffled[:n_train]
    valid = shuffled[n_train: n_train + n_valid]
    test  = shuffled[n_train + n_valid:]
    return train, valid, test


def copy_files(files: list[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)


def main():
    args = parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    out_root = Path(args.out)
    train_frac = args.train
    valid_frac = args.valid
    test_frac  = 1.0 - train_frac - valid_frac
    if test_frac < 0:
        raise ValueError("--train + --valid must be <= 1.0")

    print(f"Extracting {zip_path} …")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        # Find the top-level folder inside the zip (e.g. Hand_gesture/)
        top_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        if len(top_dirs) == 1:
            src_root = top_dirs[0]
        else:
            src_root = tmp_path  # multiple top-level dirs — treat tmp as root

        print(f"Source root inside zip: {src_root.name}")

        # Collect class folders
        class_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
        print(f"Found {len(class_dirs)} class folder(s) in zip.")

        skipped, processed = [], []
        split_summary = []

        for class_dir in class_dirs:
            images = collect_images(class_dir)
            if not images:
                skipped.append(class_dir.name)
                continue

            train_imgs, valid_imgs, test_imgs = split_files(
                images, train_frac, valid_frac, args.seed
            )

            copy_files(train_imgs, out_root / "train" / class_dir.name)
            copy_files(valid_imgs, out_root / "valid" / class_dir.name)
            if test_imgs:
                copy_files(test_imgs, out_root / "test"  / class_dir.name)

            processed.append(class_dir.name)
            split_summary.append((class_dir.name, len(train_imgs), len(valid_imgs), len(test_imgs)))

    # Remove any empty leaf directories that may have been created
    for split in ("train", "valid", "test"):
        split_dir = out_root / split
        if split_dir.exists():
            for d in list(split_dir.iterdir()):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()

    # Summary
    print(f"\nDataset written to: {out_root.resolve()}")
    print(f"Split ratios — train: {train_frac:.0%}  valid: {valid_frac:.0%}  test: {test_frac:.0%}\n")
    print(f"{'Class':<20} {'Train':>6} {'Valid':>6} {'Test':>6}")
    print("-" * 42)
    for cls, tr, va, te in split_summary:
        print(f"{cls:<20} {tr:>6} {va:>6} {te:>6}")
    if skipped:
        print(f"\nSkipped (empty): {', '.join(skipped)}")
    print(f"\nDone. {len(processed)} class(es) ready for training.")


if __name__ == "__main__":
    main()
