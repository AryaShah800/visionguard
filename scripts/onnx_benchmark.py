"""Export the selected model to ONNX, quantize to int8, and measure parity, accuracy and CPU latency.

Usage (from repo root, TrashNet downloaded, trained run present):
    python scripts/onnx_benchmark.py --run runs/baseline --out reports/onnx_benchmark.json
Calibration uses 100 TRAINING images (fixed stride); accuracy is measured on the 379 held-out test images.
Latency: batch size 1, preprocessed tensor already in memory (model only, no image decoding), single process,
median and p95 of N timed runs after warm-up, repeated in 3 rounds; reported values are medians over rounds.
The machine is shared and not isolated, so treat latency as indicative.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vg.common import CLASSES, read_json
from vg.data import load_manifest
from vg.models import load_model
from vg.onnx_tools import OnnxPredictor, export_onnx, preprocess, quantize_static


def timed(function, runs, warmup=20):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def latency(function, runs, rounds=3):
    medians, p95s = [], []
    for _ in range(rounds):
        samples = timed(function, runs)
        medians.append(statistics.median(samples))
        p95s.append(float(np.percentile(samples, 95)))
    return {'median_ms': round(statistics.median(medians), 2), 'p95_ms': round(statistics.median(p95s), 2),
            'rounds': rounds, 'runs_per_round': runs}


def cpu_name():
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/baseline')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/onnx_benchmark.json')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'runs/onnx')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.json')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/dataset-resized')
    parser.add_argument('--runs', type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    selected = read_json(args.run / 'selection.json')['selected_model']
    manifest = load_manifest(args.manifest, args.images)
    test = [r for r in manifest['rows'] if r['split'] == 'test']
    train = [r for r in manifest['rows'] if r['split'] == 'train'][::17][:100]

    fp32, int8 = args.model_dir / 'model_fp32.onnx', args.model_dir / 'model_int8.onnx'
    export_onnx(args.run / f'{selected}.pt', fp32)
    calibration = []
    for row in train:
        with Image.open(args.images / row['path']) as image:
            calibration.append(preprocess(image))
    quantize_static(fp32, int8, calibration)

    torch_model, _ = load_model(args.run / f'{selected}.pt')
    tensors = []
    for row in test:
        with Image.open(args.images / row['path']) as image:
            tensors.append(preprocess(image))
    batch = np.stack(tensors)
    labels = np.array([CLASSES.index(r['label']) for r in test])
    with torch.inference_mode():
        torch_logits = np.concatenate([torch_model(torch.from_numpy(batch[i:i + 32])).numpy()
                                       for i in range(0, len(batch), 32)])
    import onnxruntime
    result = {'selected_model': selected, 'test_images': len(test), 'calibration_images': len(calibration),
              'torch_threads': 4, 'onnxruntime_threads': 'default', 'cpu': cpu_name(),
              'python': platform.python_version(), 'torch': torch.__version__, 'onnxruntime': onnxruntime.__version__}
    sizes = {'pytorch_checkpoint': (args.run / f'{selected}.pt').stat().st_size,
             'onnx_fp32': fp32.stat().st_size, 'onnx_int8': int8.stat().st_size}
    result['size_mb'] = {k: round(v / 1e6, 2) for k, v in sizes.items()}

    variants = {'pytorch_fp32': None, 'onnx_fp32': fp32, 'onnx_int8': int8}
    one = batch[:1]
    for name, path in variants.items():
        predictor = None if path is None else OnnxPredictor(path)
        logits = torch_logits if predictor is None else np.concatenate(
            [predictor.logits(batch[i:i + 32]) for i in range(0, len(batch), 32)])
        prediction = logits.argmax(1)
        entry = {'accuracy': round(float(accuracy_score(labels, prediction)), 4),
                 'macro_f1': round(float(f1_score(labels, prediction, average='macro')), 4),
                 'top1_agreement_with_pytorch': round(float((prediction == torch_logits.argmax(1)).mean()), 4),
                 'max_abs_logit_diff_vs_pytorch': round(float(np.abs(logits - torch_logits).max()), 5)}
        if predictor is None:
            tensor = torch.from_numpy(one)

            def run():
                with torch.inference_mode():
                    torch_model(tensor)
        else:
            def run():
                predictor.logits(one)
        entry['latency_batch1'] = latency(run, args.runs)
        result[name] = entry
        print(name, entry, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
