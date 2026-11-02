import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import CLASSES, device_for, read_json, save_json, seed_everything, sha256
from .data import WasteDataset, load_manifest, open_rgb, transform_for
from .metrics import classification_metrics, selective_metrics
from .models import load_model
from .train import infer_batches


def verify_selection(run_dir, manifest_path=None):
    run_dir = Path(run_dir)
    selection = read_json(run_dir / 'selection.json')
    if manifest_path and selection['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Manifest differs from the one used for model selection')
    for kind, expected in selection['checkpoint_sha256'].items():
        if sha256(run_dir / f'{kind}.pt') != expected:
            raise ValueError(f'{kind} checkpoint changed after selection')
    return selection


def plot_confusion(metrics, destination, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.array(metrics['confusion_matrix'])
    ax.imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center', color='white' if value > matrix.max() / 2 else 'black')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted class', ylabel='True class', title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def error_gallery(rows, predictions, root, destination):
    errors = [(row, prediction) for row, prediction in zip(rows, predictions) if row['label'] != CLASSES[prediction]][:12]
    if not errors:
        return
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, (row, prediction) in zip(axes.flat, errors):
        ax.imshow(open_rgb(Path(root) / row['path']))
        ax.set_title(f"True: {row['label']}\nPredicted: {CLASSES[prediction]}", fontsize=10)
    fig.suptitle('First 12 test errors in manifest order')
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def training_curves(run_dir, destination):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, kind in zip(axes, ['cnn', 'frozen', 'finetuned']):
        history = read_json(Path(run_dir) / f'{kind}-history.json')
        epochs = [row['epoch'] for row in history]
        ax.plot(epochs, [row['train_loss'] for row in history], label='Training loss')
        ax.plot(epochs, [row['validation_loss'] for row in history], label='Validation loss')
        ax.set(title=kind, xlabel='Epoch', ylabel='Weighted cross entropy')
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def measure_latency(model, image_path, device, repeats=30):
    transform = transform_for()
    tensor = transform(open_rgb(image_path)).unsqueeze(0).to(device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for _ in range(5):
        model(tensor)
    sync()
    model_times, total_times = [], []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        model(tensor)
        sync()
        model_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        batch = transform(open_rgb(image_path)).unsqueeze(0).to(device)
        model(batch).softmax(dim=1).cpu().numpy()
        sync()
        total_times.append((time.perf_counter() - start) * 1000)
    return {'device': str(device), 'batch_size': 1, 'input_size': 224, 'repeats': repeats,
            'model_only_median_ms': float(np.median(model_times)),
            'decode_preprocess_transfer_infer_median_ms': float(np.median(total_times)),
            'decode_preprocess_transfer_infer_p95_ms': float(np.percentile(total_times, 95)),
            'note': 'Warm model; repeated image and OS file cache; model loading excluded.'}


def evaluate(manifest_path, image_root, run_dir, device='auto'):
    run_dir = Path(run_dir)
    output = run_dir / 'evaluation'
    if output.exists():
        raise FileExistsError('Test evaluation already exists; inspect the saved report instead of overwriting it.')
    selection = verify_selection(run_dir, manifest_path)
    manifest = load_manifest(manifest_path, image_root)
    seed_everything(selection['seed'])
    device = device_for(device)
    test_rows = [r for r in manifest['rows'] if r['split'] == 'test']
    loader = DataLoader(WasteDataset(test_rows, image_root), batch_size=32, num_workers=0)
    output.mkdir()
    targets = np.array([CLASSES.index(r['label']) for r in test_rows])
    baseline = selection['models']['majority']['majority_class']
    reports = {'majority': classification_metrics(targets, np.full(len(targets), baseline))}
    for kind in ['cnn', 'frozen', 'finetuned']:
        model, checkpoint = load_model(run_dir / f'{kind}.pt', device)
        actual, scores = infer_batches(model, loader, device)
        predictions = scores.argmax(axis=1)
        reports[kind] = classification_metrics(actual, predictions)
        reports[kind]['selective'] = selective_metrics(actual, scores, checkpoint['threshold'])
        if kind == selection['selected_model']:
            plot_confusion(reports[kind], output / 'confusion_matrix.png', f'{kind}: held-out test set')
            error_gallery(test_rows, predictions, image_root, output / 'errors.png')
            with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['path', 'actual', 'predicted', 'model_score'])
                writer.writerows((row['path'], row['label'], CLASSES[int(pred)], float(score.max()))
                                 for row, pred, score in zip(test_rows, predictions, scores))
            latency = measure_latency(model, Path(image_root) / test_rows[0]['path'], device)
        del model
    training_curves(run_dir, output / 'training_curves.png')
    report = {'selected_model': selection['selected_model'], 'selection_basis': 'validation macro-F1 only',
              'manifest_sha256': sha256(manifest_path), 'models': reports, 'latency': latency,
              'hardware': read_json(run_dir / 'config.json')['hardware'],
              'challenge_status': 'Not evaluated: independent phone photos have not been provided.',
              'limitations': ['Single training seed.', 'Small historical dataset with simple backgrounds.',
                             'Perceptual grouping is incomplete physical-object identification.',
                             'Scores are not calibrated probabilities.']}
    save_json(output / 'metrics.json', report)
    lines = ['# VisionGuard measured results', '',
             f"Selected using validation macro-F1: **{selection['selected_model']}**.", '',
             '| Model | Validation macro-F1 | Test macro-F1 | Test accuracy |',
             '|---|---:|---:|---:|']
    for kind, metrics in reports.items():
        lines.append(f"| {kind} | {selection['models'][kind]['validation']['macro_f1']:.4f} | "
                     f"{metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} |")
    lines += ['', f'Test images: {len(test_rows)}. Single seed: {selection["seed"]}.', '',
              'Independent phone-photo evaluation: **not yet performed**.', '',
              '![Confusion matrix](confusion_matrix.png)', '', '![Training curves](training_curves.png)', '',
              '![Error gallery](errors.png)', '', 'Source images: Gary Thung and Mindy Yang, TrashNet.']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print({kind: {'macro_f1': m['macro_f1'], 'accuracy': m['accuracy']} for kind, m in reports.items()}, flush=True)
    return report
