import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vg.common import CLASSES, save_json
from vg.data import grouped_split, validate_partitions, load_manifest, transform_for, image_hashes
from vg.metrics import select_threshold, selective_metrics
from vg.models import build_model, training_mode, load_model
from vg.predict import predict_image
from vg.challenge import lock_challenge
from vg.evaluate import verify_selection


def synthetic_rows():
    return [{'path': f'{label}/{i}.png', 'label': label, 'group': f'{label}-{i // 2}',
             'sha256': f'{label}-{i}', 'pixel_sha256': f'{label}-{i}'}
            for label in CLASSES for i in range(40)]


def test_group_split_is_repeatable_and_never_leaks():
    first = grouped_split(synthetic_rows(), 42)
    second = grouped_split(synthetic_rows(), 42)
    assert first == second
    validate_partitions(first)
    for name in ['train', 'validation', 'test']:
        assert {r['label'] for r in first if r['split'] == name} == set(CLASSES)


def test_leakage_is_rejected():
    rows = grouped_split(synthetic_rows())
    other = next(r for r in rows if r['split'] != rows[0]['split'])
    other['group'] = rows[0]['group']
    with pytest.raises(ValueError, match='leakage'):
        validate_partitions(rows)


def test_manifest_rejects_changed_images(tmp_path):
    rows = grouped_split(synthetic_rows())
    for row in rows:
        path = tmp_path / row['path']
        path.parent.mkdir(exist_ok=True)
        path.write_text('changed')
    manifest = tmp_path / 'manifest.json'
    save_json(manifest, {'classes': CLASSES, 'rows': rows})
    with pytest.raises(ValueError, match='changed'):
        load_manifest(manifest, tmp_path)


def test_frozen_backbone_has_no_gradients_or_batchnorm_updates():
    torch.set_num_threads(2)
    model = build_model('frozen')
    training_mode(model, 'frozen')
    bn = next(m for m in model.features.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    model(torch.rand(2, 3, 64, 64)).sum().backward()
    assert torch.equal(before, bn.running_mean)
    assert all(p.grad is None for p in model.features.parameters())
    assert model.classifier[1].weight.grad is not None


def test_finetuning_only_unfreezes_final_blocks():
    model = build_model('finetuned')
    training_mode(model, 'finetuned')
    assert all(not p.requires_grad for p in model.features[:14].parameters())
    assert all(p.requires_grad for p in model.features[14:].parameters())
    assert not model.features[0].training
    assert model.features[14].training


def test_checkpoint_roundtrip_and_predict(tmp_path):
    model = build_model('cnn').eval()
    checkpoint = {'kind': 'cnn', 'classes': CLASSES, 'threshold': .99, 'state_dict': model.state_dict()}
    path = tmp_path / 'model.pt'
    torch.save(checkpoint, path)
    restored, metadata = load_model(path)
    image = Image.new('RGB', (256, 256), 'green')
    result = predict_image(image, restored, metadata)
    assert result['prediction'] in CLASSES
    assert len(result['top_three']) == 3
    assert result['uncertain']
    assert 0 <= result['top_three'][0]['score'] <= 1
    with pytest.raises(ValueError, match='16'):
        predict_image(Image.new('RGB', (2, 2)), restored, metadata)


def test_evaluation_transform_is_deterministic():
    image = Image.new('RGB', (512, 384), 'red')
    transform = transform_for()
    assert torch.equal(transform(image), transform(image))
    assert transform(image).shape == (3, 224, 224)


def test_threshold_selection_and_empty_coverage():
    scores = np.array([[.95, .05], [.1, .9], [.55, .45]])
    actual = np.array([0, 1, 1])
    threshold = select_threshold(actual, scores)
    report = selective_metrics(actual, scores, threshold)
    assert report['coverage'] == pytest.approx(2 / 3)
    assert report['accepted_accuracy'] == 1
    assert selective_metrics(actual, scores, .999)['accepted_accuracy'] is None
    assert select_threshold(np.array([1, 0, 1]), scores) is None


def test_challenge_requires_all_classes(tmp_path):
    benchmark = tmp_path / 'benchmark.json'
    save_json(benchmark, {'rows': []})
    with pytest.raises(ValueError, match='six classes'):
        lock_challenge(tmp_path, tmp_path / 'challenge.json', benchmark)


def test_challenge_rejects_benchmark_images(tmp_path):
    folder = tmp_path / 'cardboard'
    folder.mkdir()
    image = folder / 'copied.png'
    Image.new('RGB', (64, 64), 'orange').save(image)
    pixel, phash = image_hashes(image)
    benchmark = tmp_path / 'benchmark.json'
    save_json(benchmark, {'rows': [{'pixel_sha256': pixel, 'phash': phash}]})
    with pytest.raises(ValueError, match='overlap'):
        lock_challenge(tmp_path, tmp_path / 'challenge.json', benchmark)


def test_selected_checkpoint_tampering_is_rejected(tmp_path):
    (tmp_path / 'cnn.pt').write_bytes(b'changed checkpoint')
    save_json(tmp_path / 'selection.json', {'checkpoint_sha256': {'cnn': 'original-hash'}})
    with pytest.raises(ValueError, match='changed after selection'):
        verify_selection(tmp_path)
