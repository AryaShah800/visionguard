from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .common import CLASSES
from .data import transform_for
from .evaluate import verify_selection
from .models import load_model


def load_predictor(run_dir):
    selection = verify_selection(run_dir)
    model, checkpoint = load_model(Path(run_dir) / f"{selection['selected_model']}.pt")
    torch.set_num_threads(4)
    return model, checkpoint


@torch.inference_mode()
def predict_image(image, model, checkpoint):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if image.width < 16 or image.height < 16:
        raise ValueError('Use an image at least 16 by 16 pixels.')
    device = next(model.parameters()).device
    scores = model(transform_for()(image).unsqueeze(0).to(device)).softmax(dim=1)[0].cpu().numpy()
    ranked = np.argsort(scores)[::-1][:3]
    threshold = checkpoint['threshold']
    uncertain = threshold is not None and float(scores[ranked[0]]) < threshold
    return {'prediction': CLASSES[int(ranked[0])], 'uncertain': uncertain,
            'threshold': threshold, 'model': checkpoint['kind'],
            'top_three': [{'class': CLASSES[int(i)], 'score': float(scores[i])} for i in ranked],
            'note': 'Model scores are not calibrated correctness probabilities. One prominent item per photo.'}


def predict_file(path, run_dir):
    model, checkpoint = load_predictor(run_dir)
    with Image.open(path) as image:
        return predict_image(image, model, checkpoint)
