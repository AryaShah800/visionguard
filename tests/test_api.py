"""API tests: synthetic images and a fake predictor; no trained checkpoint or network needed."""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
try:
    import python_multipart  # noqa: F401  (python-multipart >= 0.0.13)
except ImportError:
    pytest.importorskip('multipart', reason='API tests need requirements-api.txt')
from fastapi.testclient import TestClient

from vg import __version__
from vg.api import MAX_UPLOAD_BYTES, NOT_READY, create_app
from vg.common import CLASSES
from vg.models import build_model
from vg.predict import load_predictor


def image_bytes(size=(64, 64), color='green', fmt='PNG', mode='RGB'):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def truncated_png():
    buffer = io.BytesIO()
    Image.effect_noise((256, 256), 64).save(buffer, format='PNG')  # noise does not compress, so the cut lands in pixel data
    data = buffer.getvalue()
    return data[:len(data) // 2]


def upload(client, data, name='item.png', content_type='image/png'):
    return client.post('/predict', files={'file': (name, data, content_type)})


class FakeLoader:
    def __init__(self, threshold=0.5):
        self.threshold, self.calls = threshold, []

    def __call__(self, run_dir):
        self.calls.append(Path(run_dir))
        return object(), {'kind': 'fake', 'threshold': self.threshold}


def fake_predict(scores):
    def predict(image, model, checkpoint):
        assert image.size[0] >= 16 and image.size[1] >= 16
        ranked = sorted(zip(CLASSES, scores), key=lambda pair: -pair[1])[:3]
        threshold = checkpoint['threshold']
        return {'prediction': ranked[0][0], 'uncertain': threshold is not None and ranked[0][1] < threshold,
                'threshold': threshold, 'model': checkpoint['kind'],
                'top_three': [{'class': name, 'score': score} for name, score in ranked], 'note': 'synthetic'}
    return predict


CONFIDENT = [0.05, 0.05, 0.70, 0.10, 0.05, 0.05]
UNSURE = [0.20, 0.15, 0.30, 0.15, 0.10, 0.10]


@pytest.fixture
def loader():
    return FakeLoader()


@pytest.fixture
def client(loader, tmp_path):
    return TestClient(create_app(loader=loader, predict=fake_predict(CONFIDENT), run_dir=tmp_path))


def test_health_reports_version_without_loading_model(client, loader):
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body == {'status': 'ok', 'version': __version__, 'ready': False, 'model_loaded': False, 'model': None}
    assert loader.calls == []


def test_health_ready_when_run_artifacts_exist(client, loader, tmp_path):
    (tmp_path / 'selection.json').write_text('{}', encoding='utf-8')
    assert client.get('/health').json()['ready'] is True
    assert loader.calls == []


def test_predict_returns_top_three_and_loads_model_once(client, loader, tmp_path):
    first = upload(client, image_bytes())
    second = upload(client, image_bytes(color='blue'))
    assert first.status_code == second.status_code == 200
    body = first.json()
    assert body['prediction'] == 'metal'
    assert body['abstained'] is False
    assert [item['class'] for item in body['top_three']] == ['metal', 'paper', 'cardboard']
    scores = [item['score'] for item in body['top_three']]
    assert scores == sorted(scores, reverse=True)
    assert body['model'] == 'fake' and body['threshold'] == 0.5
    assert loader.calls == [tmp_path]
    health = client.get('/health').json()
    assert health['ready'] is True and health['model_loaded'] is True and health['model'] == 'fake'


def test_predict_abstains_below_threshold(loader, tmp_path):
    client = TestClient(create_app(loader=loader, predict=fake_predict(UNSURE), run_dir=tmp_path))
    body = upload(client, image_bytes()).json()
    assert body['abstained'] is True
    assert body['prediction'] == 'metal'


def test_run_dir_is_read_from_env_lazily(loader, tmp_path, monkeypatch):
    monkeypatch.delenv('VISIONGUARD_RUN', raising=False)
    monkeypatch.delenv('VG_RUN_DIR', raising=False)
    client = TestClient(create_app(loader=loader, predict=fake_predict(CONFIDENT)))
    monkeypatch.setenv('VG_RUN_DIR', str(tmp_path / 'configured'))
    assert upload(client, image_bytes()).status_code == 200
    assert loader.calls == [tmp_path / 'configured']


@pytest.mark.parametrize('fmt,name,content_type', [('JPEG', 'item.jpg', 'image/jpeg'), ('WEBP', 'item.webp', 'image/webp'),
                                                   ('PNG', 'item.bin', 'application/octet-stream')])
def test_accepts_supported_formats(client, fmt, name, content_type):
    response = upload(client, image_bytes(fmt=fmt), name, content_type)
    assert response.status_code == 200, response.text


def test_rejects_file_just_over_limit(client, loader):
    response = upload(client, b'\0' * (MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 413
    assert '5 MiB' in response.json()['detail']
    assert loader.calls == []


def test_rejects_large_body_by_content_length(client, loader):
    response = upload(client, b'\0' * (6 * 1024 * 1024))
    assert response.status_code == 413
    assert loader.calls == []


def test_rejects_large_streamed_body_without_content_length(client, loader):
    boundary = 'visionguardtest'
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\n'
            'Content-Type: image/png\r\n\r\n').encode()
    body = head + b'\0' * (6 * 1024 * 1024) + f'\r\n--{boundary}--\r\n'.encode()
    response = client.post('/predict', content=iter([body]),
                           headers={'content-type': f'multipart/form-data; boundary={boundary}'})
    assert response.status_code == 413
    assert loader.calls == []


@pytest.mark.parametrize('data,name,content_type,status', [
    (b'', 'empty.png', 'image/png', 400),
    (b'definitely not an image', 'fake.png', 'image/png', 415),
    (b'hello', 'note.txt', 'text/plain', 415),
    (image_bytes(fmt='GIF', mode='L', color=128), 'item.gif', 'image/gif', 415),
    (image_bytes(fmt='GIF', mode='L', color=128), 'item.bin', 'application/octet-stream', 415),
    (truncated_png(), 'cut.png', 'image/png', 400),
    (image_bytes(size=(8, 8)), 'tiny.png', 'image/png', 422),
], ids=['empty', 'fake', 'text', 'gif', 'disguised-gif', 'truncated-png', 'tiny'])
def test_rejects_invalid_uploads(client, loader, data, name, content_type, status):
    response = upload(client, data, name, content_type)
    assert response.status_code == status, response.text
    assert loader.calls == []


def test_rejects_animated_images(client, loader):
    buffer = io.BytesIO()
    frames = [Image.new('RGB', (32, 32), color) for color in ['red', 'blue']]
    frames[0].save(buffer, format='PNG', save_all=True, append_images=frames[1:])
    assert upload(client, buffer.getvalue()).status_code == 415
    assert loader.calls == []


def test_rejects_decompression_bomb_before_decoding(client, loader):
    # 7000 x 7000 1-bit PNG: a few KB on disk, 49 MP decoded (over the 40 MP cap).
    bomb = image_bytes(size=(7000, 7000), color=0, mode='1')
    assert len(bomb) < MAX_UPLOAD_BYTES
    response = upload(client, bomb)
    assert response.status_code == 413
    assert 'dimensions' in response.json()['detail']
    assert loader.calls == []


def test_pixel_cap_is_configurable(loader, tmp_path):
    client = TestClient(create_app(loader=loader, predict=fake_predict(CONFIDENT), run_dir=tmp_path, max_pixels=10_000))
    assert upload(client, image_bytes(size=(200, 200))).status_code == 413
    assert upload(client, image_bytes(size=(64, 64))).status_code == 200


def test_missing_model_returns_503_without_local_paths(tmp_path):
    run_dir = tmp_path / 'private-run-folder'
    client = TestClient(create_app(loader=load_predictor, run_dir=run_dir))
    response = upload(client, image_bytes())
    assert response.status_code == 503
    assert response.json() == {'detail': NOT_READY}
    assert str(tmp_path) not in response.text and 'private-run-folder' not in response.text
    assert client.get('/health').json()['ready'] is False


def test_unexpected_errors_hide_details(loader, tmp_path):
    def broken(image, model, checkpoint):
        raise RuntimeError(f'failure reading {tmp_path / "secret.pt"}')
    client = TestClient(create_app(loader=loader, predict=broken, run_dir=tmp_path), raise_server_exceptions=False)
    response = upload(client, image_bytes())
    assert response.status_code == 500
    assert response.json() == {'detail': 'Internal server error.'}
    assert 'secret.pt' not in response.text


def test_real_predict_image_with_untrained_cnn(tmp_path):
    # Exercises vg.predict.predict_image end to end with random weights; checks shape, not accuracy.
    def loader(run_dir):
        return build_model('cnn').eval(), {'kind': 'cnn', 'classes': CLASSES, 'threshold': None}
    client = TestClient(create_app(loader=loader, run_dir=tmp_path))
    response = upload(client, image_bytes(size=(300, 200), fmt='JPEG'), 'item.jpg', 'image/jpeg')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['prediction'] in CLASSES
    assert body['abstained'] is False
    assert len(body['top_three']) == 3
    assert all(0 <= item['score'] <= 1 for item in body['top_three'])
    assert body['prediction'] == body['top_three'][0]['class']
