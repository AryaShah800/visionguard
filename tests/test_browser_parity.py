"""Parity: the in-browser int8 ONNX demo (docs/index.html) vs Python onnxruntime on the same file.

Python path = vg.onnx_tools.OnnxPredictor (torchvision preprocessing + onnxruntime).
Browser path = docs/index.html (JS preprocessing + onnxruntime-web). Needs playwright + Edge and network
access to jsDelivr (the page pins onnxruntime-web), so it is skipped when playwright is not installed.
"""
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DOCS = Path(__file__).resolve().parents[1] / 'docs'
MODEL = DOCS / 'model_int8.onnx'
PROB_TOLERANCE = 0.001
pytest.importorskip('onnxruntime')
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason='docs/model_int8.onnx missing')


def python_result(path):
    from vg.onnx_tools import OnnxPredictor
    from PIL import ImageOps
    with Image.open(path) as image:
        label, scores = OnnxPredictor(MODEL).predict(ImageOps.exif_transpose(image).convert('RGB'))
    return label, scores


def sample_files(tmp_path):
    files = sorted((DOCS / 'samples').glob('*.jpg'))
    assert len(files) == 6
    # non-square inputs exercise the short-side resize and centre-crop on both axes
    with Image.open(files[0]) as image:
        portrait = tmp_path / 'portrait.png'
        image.crop((100, 0, 300, 384)).save(portrait)
    with Image.open(files[1]) as image:
        landscape = tmp_path / 'wide.png'
        image.resize((700, 300)).save(landscape)
    return files + [portrait, landscape]


def test_python_side_is_sane(tmp_path):
    for path in sample_files(tmp_path):
        label, scores = python_result(path)
        assert abs(float(scores.sum()) - 1) < 1e-4 and label in ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')


def test_browser_matches_python(tmp_path):
    sync_api = pytest.importorskip('playwright.sync_api')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}/index.html'
    worst = 0.0
    try:
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel='msedge', headless=True)
            except Exception:
                browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(base)
            page.wait_for_selector('body[data-ready="1"]', timeout=60000)
            for path in sample_files(tmp_path):
                page.evaluate("document.getElementById('result').hidden = true; delete document.getElementById('result').dataset.json")
                page.set_input_files('#file', str(path))
                page.wait_for_function("!!document.getElementById('result').dataset.json", timeout=30000)
                got = json.loads(page.get_attribute('#result', 'data-json'))
                label, scores = python_result(path)
                diff = float(np.abs(np.array(got['probabilities']) - scores).max())
                worst = max(worst, diff)
                assert got['top_three'][0]['class'] == label, path.name
                assert diff < PROB_TOLERANCE, f'{path.name}: max prob diff {diff:.4f}'
                assert got['uncertain'] == (float(scores.max()) < 0.4)
            browser.close()
            assert not errors, errors
    finally:
        server.shutdown()
    print(f'worst max-abs probability diff across 8 images: {worst:.5f}')
