"""Grad-CAM galleries from the held-out test set: first 6 correct and first 6 wrong predictions in manifest order.

Usage: python scripts/make_gradcam_gallery.py [--run runs/baseline] [--out reports]
Needs the TrashNet images (python -m vg download) and the trained run. Writes
reports/gradcam_correct.png and reports/gradcam_errors.png (no cherry-picking: manifest order).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vg.common import CLASSES, read_json
from vg.data import load_manifest
from vg.explain import display_crop, explain_image
from vg.predict import load_predictor


def draw(items, title, path):
    fig, axes = plt.subplots(2, len(items), figsize=(2.3 * len(items), 5.2))
    for column, (crop, overlay, truth, predicted, score) in enumerate(items):
        axes[0, column].imshow(crop)
        axes[0, column].set_title(f'true: {truth}', fontsize=9)
        axes[1, column].imshow(overlay)
        axes[1, column].set_title(f'pred: {predicted} ({score:.2f})', fontsize=9)
        for row in range(2):
            axes[row, column].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/baseline')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.json')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/dataset-resized')
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, args.images)
    model, checkpoint = load_predictor(args.run)
    correct, wrong = [], []
    for row in (r for r in manifest['rows'] if r['split'] == 'test'):
        if len(correct) >= 6 and len(wrong) >= 6:
            break
        with Image.open(args.images / row['path']) as image:
            image = image.convert('RGB')
            result = explain_image(image, model, checkpoint)
            item = (display_crop(image), result['overlay'], row['label'], result['prediction'], result['score'])
        hit = row['label'] == result['prediction']
        if hit and len(correct) < 6:
            correct.append(item)
        elif not hit and len(wrong) < 6:
            wrong.append(item)
    args.out.mkdir(parents=True, exist_ok=True)
    draw(correct, 'Grad-CAM, first 6 correct test predictions (manifest order)', args.out / 'gradcam_correct.png')
    draw(wrong, 'Grad-CAM, first 6 test errors (manifest order)', args.out / 'gradcam_errors.png')
    print(f'wrote {len(correct)} correct, {len(wrong)} error panels to {args.out}')


if __name__ == '__main__':
    main()
