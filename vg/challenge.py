from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .common import CLASSES, device_for, read_json, save_json, sha256
from .data import EXTENSIONS, WasteDataset, image_hashes
from .evaluate import plot_confusion, verify_selection
from .metrics import classification_metrics, selective_metrics
from .models import load_model
from .train import infer_batches


def lock_challenge(image_root, manifest_path, benchmark_path):
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError('Challenge manifest already locked; choose a new path for a new collection.')
    root = Path(image_root).resolve()
    benchmark = read_json(benchmark_path)
    known_pixels = {r['pixel_sha256'] for r in benchmark['rows']}
    known_phashes = [r['phash'] for r in benchmark['rows']]
    rows, seen = [], set()
    for label in CLASSES:
        for path in sorted((root / label).rglob('*')):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            pixel, phash = image_hashes(path)
            if pixel in known_pixels or any((phash ^ h).bit_count() <= 4 for h in known_phashes):
                raise ValueError(f'Possible benchmark overlap: {path.name}. Review collection before locking.')
            if pixel in seen:
                raise ValueError(f'Duplicate challenge photo: {path.name}')
            seen.add(pixel)
            rows.append({'path': path.relative_to(root).as_posix(), 'label': label,
                         'sha256': sha256(path), 'pixel_sha256': pixel})
    if {r['label'] for r in rows} != set(CLASSES):
        raise ValueError('Collect at least one independent photo for each of the six classes; target 20 per class.')
    save_json(manifest_path, {'classes': CLASSES, 'rows': rows,
                              'benchmark_manifest_sha256': sha256(benchmark_path),
                              'class_counts': dict(Counter(r['label'] for r in rows)),
                              'purpose': 'Independent phone-photo evaluation; do not tune on these images.'})
    print(f'Locked {len(rows)} challenge photos. Verify distinct physical items manually.')


def evaluate_challenge(image_root, manifest_path, run_dir, device='auto'):
    run_dir, root = Path(run_dir), Path(image_root).resolve()
    destination = run_dir / 'challenge-evaluation'
    if destination.exists():
        raise FileExistsError('Challenge results already exist; do not overwrite held-out evaluation.')
    selection = verify_selection(run_dir)
    manifest = read_json(manifest_path)
    if manifest['classes'] != CLASSES or manifest['benchmark_manifest_sha256'] != selection['manifest_sha256']:
        raise ValueError('Challenge class mapping or benchmark provenance mismatch')
    for row in manifest['rows']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or sha256(path) != row['sha256']:
            raise ValueError('Challenge image changed or path escapes dataset root')
    device = device_for(device)
    model, checkpoint = load_model(run_dir / f"{selection['selected_model']}.pt", device)
    loader = DataLoader(WasteDataset(manifest['rows'], root), batch_size=32)
    targets, scores = infer_batches(model, loader, device)
    report = classification_metrics(targets, scores.argmax(axis=1))
    report['selective'] = selective_metrics(targets, scores, checkpoint['threshold'])
    report['challenge_manifest_sha256'] = sha256(manifest_path)
    report['selected_model'] = selection['selected_model']
    benchmark_report = run_dir / 'evaluation/metrics.json'
    if benchmark_report.exists():
        prior = read_json(benchmark_report)['models'][selection['selected_model']]['macro_f1']
        report['macro_f1_change_percentage_points'] = (report['macro_f1'] - prior) * 100
    destination.mkdir()
    save_json(destination / 'metrics.json', report)
    plot_confusion(report, destination / 'confusion_matrix.png', 'Independent phone photos')
    print({'macro_f1': report['macro_f1'], 'accuracy': report['accuracy'], 'count': report['count']})
    return report
