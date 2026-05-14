#!/usr/bin/env python3
"""Copy per-case T1 `.nii.gz` files and extract them into a destination folder.

Default behavior matches the UCSF cohort layout on the mounted Windows drive:
- Source: /media/pleb/7AFAFDB8FAFD70AD/UPenn/UCSF/DATA-IMAGE-STRUCTURAL
- Destination: /home/pleb/fastsurfer/data

For each immediate case folder under the source root, the script looks for the
case-level T1 file like `UCSF-PDGM-0004_T1.nii.gz`, copies it into the
destination, extracts it there as `.nii`, and removes the copied `.nii.gz`.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path("/media/pleb/7AFAFDB8FAFD70AD/UPenn/UCSF/DATA-IMAGE-STRUCTURAL")
DEFAULT_DEST = Path("/home/pleb/fastsurfer/data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Folder containing one subfolder per case.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_DEST,
        help="Folder where copied files should be extracted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing `.nii` outputs in the destination.",
    )
    return parser.parse_args()


def find_t1_gz(case_dir: Path) -> Path | None:
    expected = case_dir / f"{case_dir.name}_T1.nii.gz"
    if expected.exists():
        return expected

    matches = sorted(
        path
        for path in case_dir.glob("*_T1.nii.gz")
        if path.is_file() and not path.name.startswith("._")
    )
    return matches[0] if matches else None


def copy_and_extract(src_gz: Path, dest_dir: Path, overwrite: bool) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_gz = dest_dir / src_gz.name
    extracted_nii = dest_dir / src_gz.with_suffix("").name

    if extracted_nii.exists() and not overwrite:
        return f"skip existing {extracted_nii.name}"

    shutil.copy2(src_gz, copied_gz)
    with gzip.open(copied_gz, "rb") as src_handle, extracted_nii.open("wb") as dst_handle:
        shutil.copyfileobj(src_handle, dst_handle)
    copied_gz.unlink()
    return f"copied and extracted {src_gz.name} -> {extracted_nii.name}"


def main() -> int:
    args = parse_args()

    if not args.source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {args.source_root}")

    case_dirs = sorted(
        path for path in args.source_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )

    copied = 0
    skipped_missing = 0
    skipped_existing = 0

    for case_dir in case_dirs:
        src_gz = find_t1_gz(case_dir)
        if src_gz is None:
            print(f"missing T1 file in {case_dir}")
            skipped_missing += 1
            continue

        result = copy_and_extract(src_gz, args.dest_dir, args.overwrite)
        print(result)
        if result.startswith("skip existing"):
            skipped_existing += 1
        else:
            copied += 1

    print(
        f"done: copied={copied}, skipped_existing={skipped_existing}, "
        f"skipped_missing={skipped_missing}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
