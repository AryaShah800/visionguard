"""Assemble a Hugging Face Docker Space folder from this repo (does not upload anything).

    python scripts/prepare_space.py <output_dir>

Then follow docs/SERVING.md ("Deploying to Hugging Face Spaces") to push <output_dir> to your Space.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_FILES = ['selection.json', 'cnn.pt', 'frozen.pt', 'finetuned.pt']


def main(out: Path) -> None:
    missing = [f for f in CHECKPOINT_FILES if not (ROOT / 'runs' / 'baseline' / f).is_file()]
    if missing:
        sys.exit(f'Missing runs/baseline files: {missing}. Train or restore the run first.')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'vg', out / 'vg', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (out / 'runs' / 'baseline').mkdir(parents=True, exist_ok=True)
    for f in CHECKPOINT_FILES:
        shutil.copy2(ROOT / 'runs' / 'baseline' / f, out / 'runs' / 'baseline' / f)
    for f in ['Dockerfile', '.dockerignore', 'requirements-api.txt']:
        shutil.copy2(ROOT / f, out / f)
    shutil.copy2(ROOT / 'space' / 'README.md', out / 'README.md')  # Space metadata front matter
    print(f'Space folder ready: {out}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
