import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vg.common import CLASSES
from vg.explain import display_crop, explain_image, gradcam, overlay
from vg.models import build_model


def sample():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))


@pytest.mark.parametrize('kind', ['cnn', 'frozen', 'finetuned'])
def test_gradcam_shape_range_and_target(kind):
    model = build_model(kind)
    cam, predicted, index, scores = gradcam(model, sample())
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert predicted == index and abs(scores.sum() - 1) < 1e-5
    if kind != 'cnn':  # an untrained small CNN can legitimately yield an all-negative (zero after ReLU) map
        assert cam.max() > 0


def test_gradcam_explains_requested_class_and_overlay_matches_crop():
    model = build_model('frozen')
    cam, predicted, index, _ = gradcam(model, sample(), target=3)
    assert index == 3
    result = explain_image(sample(), model, {}, target=3)
    assert result['explained_class'] == CLASSES[3]
    assert result['overlay'].size == display_crop(sample()).size == (224, 224)
    assert overlay(sample(), cam).mode == 'RGB'


def test_gradcam_leaves_model_params_without_grads_when_frozen():
    model = build_model('frozen')
    gradcam(model, sample())
    assert all(p.grad is None for p in model.features.parameters())
