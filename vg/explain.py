"""Grad-CAM explanations for the VisionGuard classifiers.

Grad-CAM (Selvaraju et al., 2017) weights the last convolutional feature maps by the
average gradient of the target class score and keeps the positive part. It shows which
image regions drove the score; it does not prove the model reasons correctly.
"""
import io

import numpy as np
import torch
from PIL import Image, ImageOps

from .common import CLASSES
from .data import transform_for


def target_layer(model):
    """Last convolutional feature map: MobileNetV2 ends features with a conv block; the small CNN with a pool."""
    features = model.features
    return features[-2] if isinstance(features[-1], torch.nn.AdaptiveAvgPool2d) else features[-1]


def gradcam(model, image, target=None):
    """Returns (heatmap in [0, 1] shaped like the input crop, predicted index, target index, scores)."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    tensor = transform_for()(image).unsqueeze(0)
    store = {}

    def forward_hook(_module, _inputs, output):
        store['activations'] = output
        output.register_hook(lambda grad: store.__setitem__('gradients', grad))

    handle = target_layer(model).register_forward_hook(forward_hook)
    try:
        model.eval()
        with torch.enable_grad():
            # Frozen backbones have no trainable parameters; the input needs a grad path.
            tensor.requires_grad_(True)
            logits = model(tensor)
            scores = logits.softmax(dim=1)[0].detach().numpy()
            predicted = int(logits.argmax(dim=1))
            index = predicted if target is None else int(target)
            model.zero_grad(set_to_none=True)
            logits[0, index].backward()
    finally:
        handle.remove()
    weights = store['gradients'].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * store['activations']).sum(dim=1, keepdim=True)).detach()
    cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode='bilinear', align_corners=False)[0, 0].numpy()
    peak = cam.max()
    cam = cam / peak if peak > 0 else np.zeros_like(cam)
    return cam, predicted, index, scores


def display_crop(image):
    """The same resize + center crop the model sees, as a PIL image, so overlays line up."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    resize, crop = 232, 224
    scale = resize / min(image.size)
    image = image.resize((max(crop, round(image.width * scale)), max(crop, round(image.height * scale))), Image.BILINEAR)
    left, top = (image.width - crop) // 2, (image.height - crop) // 2
    return image.crop((left, top, left + crop, top + crop))


def colorize(cam):
    """Blue-to-red ramp without a matplotlib dependency at serve time."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image, cam, alpha=0.45):
    base = display_crop(image)
    heat = Image.fromarray(colorize(cam))
    return Image.blend(base, heat, alpha)


def explain_image(image, model, checkpoint, target=None):
    cam, predicted, index, scores = gradcam(model, image, target)
    return {'overlay': overlay(image, cam), 'heatmap': cam, 'prediction': CLASSES[predicted],
            'explained_class': CLASSES[index], 'score': float(scores[index])}


def overlay_png_bytes(image, model, checkpoint, target=None):
    result = explain_image(image, model, checkpoint, target)
    buffer = io.BytesIO()
    result['overlay'].save(buffer, format='PNG')
    return buffer.getvalue(), result
