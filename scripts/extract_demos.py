"""
Unpack any archives dropped in data/raw/ and flatten the .dem files back into
data/raw/. Safe to run repeatedly — skips demos already extracted.

Handles .rar (unar) and .zip (bsdtar). Just drop HLTV .rar files in data/raw
and run:  python scripts/extract_demos.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

RAW = Path("data/raw")


def extract(archive: Path, dest: Path) -> None:
    if archive.suffix.lower() == ".rar":
        subprocess.run(["unar", "-force-overwrite", "-o", str(dest), str(archive)], check=True)
    else:  # .zip and friends
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(dest)], check=True)


def main() -> None:
    archives = [p for p in RAW.glob("*") if p.suffix.lower() in (".rar", ".zip")]
    if not archives:
        print("No .rar/.zip archives in data/raw/ — nothing to extract.")
        return

    tmp = RAW / "_unpack"
    for arc in archives:
        print(f"→ extracting {arc.name}")
        tmp.mkdir(exist_ok=True)
        extract(arc, tmp)
        # move every .dem found (possibly nested) up into data/raw/
        for dem in tmp.rglob("*.dem"):
            target = RAW / dem.name
            if target.exists():
                print(f"    skip (exists): {dem.name}")
                continue
            shutil.move(str(dem), str(target))
            print(f"    + {dem.name}")
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nDone. {len(list(RAW.glob('*.dem')))} .dem files now in data/raw/")


if __name__ == "__main__":
    sys.exit(main())
