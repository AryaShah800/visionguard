"""ONNX export, static int8 quantization and an onnxruntime predictor for the selected checkpoint."""
from pathlib import Path

import numpy as np
import torch

from .common import CLASSES
from .data import transform_for
from .models import load_model


def export_onnx(checkpoint_path, destination, opset=17):
    model, checkpoint = load_model(checkpoint_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, dummy, str(destination), input_names=['image'], output_names=['logits'],
                      dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=opset,
                      dynamo=False)
    return checkpoint


def quantize_static(fp32_path, int8_path, calibration_tensors):
    """Static QDQ int8 quantization calibrated on preprocessed (3, 224, 224) float arrays."""
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType
    from onnxruntime.quantization import quantize_static as ort_quantize

    class Reader(CalibrationDataReader):
        def __init__(self, tensors):
            self.iterator = iter([{'image': t[None].astype(np.float32)} for t in tensors])

        def get_next(self):
            return next(self.iterator, None)

    ort_quantize(str(fp32_path), str(int8_path), Reader(calibration_tensors),
                 quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                 per_channel=True, calibrate_method=CalibrationMethod.MinMax)


def preprocess(image):
    return transform_for()(image.convert('RGB')).numpy()


class OnnxPredictor:
    def __init__(self, path, threads=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    def logits(self, batch):
        return self.session.run(None, {'image': np.asarray(batch, dtype=np.float32)})[0]

    def predict(self, image):
        scores = torch.softmax(torch.from_numpy(self.logits(preprocess(image)[None])), dim=1)[0].numpy()
        return CLASSES[int(scores.argmax())], scores


def load_onnx_predictor(run_dir, model_path=None):
    """Loader for vg.api: (OnnxPredictor, checkpoint metadata) using the verified run and an exported ONNX file."""
    from .evaluate import verify_selection
    selection = verify_selection(run_dir)
    _, checkpoint = load_model(Path(run_dir) / f"{selection['selected_model']}.pt")
    path = Path(model_path) if model_path else Path(run_dir).parent / 'onnx' / 'model_int8.onnx'
    checkpoint = {'kind': f"{checkpoint['kind']}-onnx", 'threshold': checkpoint['threshold']}
    return OnnxPredictor(path), checkpoint


def predict_image_onnx(image, predictor, checkpoint):
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image).convert('RGB')
    if image.width < 16 or image.height < 16:
        raise ValueError('Use an image at least 16 by 16 pixels.')
    _, scores = predictor.predict(image)
    ranked = np.argsort(scores)[::-1][:3]
    threshold = checkpoint['threshold']
    return {'prediction': CLASSES[int(ranked[0])], 'uncertain': threshold is not None and float(scores[ranked[0]]) < threshold,
            'threshold': threshold, 'model': checkpoint['kind'],
            'top_three': [{'class': CLASSES[int(i)], 'score': float(scores[i])} for i in ranked],
            'note': 'Model scores are not calibrated correctness probabilities. One prominent item per photo.'}
