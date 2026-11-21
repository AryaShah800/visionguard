"""Percentile bootstrap 95% confidence intervals for the selected model, from reports/predictions.csv.

Usage: python scripts/bootstrap_ci.py [--predictions reports/predictions.csv] [--out reports/bootstrap_ci.json]
Resamples the 379 held-out test images with replacement (2000 draws, seed 0). This captures test-set sampling
uncertainty only; it does not capture training-seed variance (the experiment uses one seed).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
ROOT = Path(__file__).resolve().parents[1]


def f1_scores(actual, predicted):
    scores = []
    for index in range(len(CLASSES)):
        tp = np.sum((actual == index) & (predicted == index))
        fp = np.sum((actual != index) & (predicted == index))
        fn = np.sum((actual == index) & (predicted != index))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, default=ROOT / 'reports/predictions.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/bootstrap_ci.json')
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    with args.predictions.open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    actual = np.array([CLASSES.index(r['actual']) for r in rows])
    predicted = np.array([CLASSES.index(r['predicted']) for r in rows])
    rng = np.random.default_rng(0)
    macro, accuracy, per_class = [], [], []
    for _ in range(args.draws):
        sample = rng.integers(0, len(rows), len(rows))
        f1 = f1_scores(actual[sample], predicted[sample])
        macro.append(f1.mean())
        accuracy.append(np.mean(actual[sample] == predicted[sample]))
        per_class.append(f1)
    per_class = np.array(per_class)

    def interval(values):
        low, high = np.percentile(values, [2.5, 97.5])
        return [round(float(low), 4), round(float(high), 4)]

    point = f1_scores(actual, predicted)
    result = {'images': len(rows), 'draws': args.draws, 'seed': 0,
              'macro_f1': {'point': round(float(point.mean()), 4), 'ci95': interval(macro)},
              'accuracy': {'point': round(float(np.mean(actual == predicted)), 4), 'ci95': interval(accuracy)},
              'per_class_f1': {name: {'point': round(float(point[i]), 4), 'support': int(np.sum(actual == i)),
                                      'ci95': interval(per_class[:, i])} for i, name in enumerate(CLASSES)}}
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
