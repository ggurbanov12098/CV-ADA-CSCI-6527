"""Remove all generated outputs and logs."""

import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

DIRS_TO_CLEAR = [
    os.path.join(ROOT, "images", "outputs"),
]

def main():
    removed = 0
    for d in DIRS_TO_CLEAR:
        if os.path.isdir(d):
            count = sum(len(files) for _, _, files in os.walk(d))
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
            removed += count
            print(f"Cleared {d}/ ({count} files)")
        else:
            print(f"Skipped {d}/ (not found)")

    print(f"\nDone — removed {removed} files total.")


if __name__ == "__main__":
    main()
