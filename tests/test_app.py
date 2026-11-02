from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_handles_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv('VISIONGUARD_RUN', str(tmp_path / 'missing-run'))
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception
    assert any('not ready yet' in item.value for item in app.info)


@pytest.mark.skipif(not (ROOT / 'runs/baseline/selection.json').exists(), reason='Requires a trained local experiment')
def test_app_predicts_training_sample(monkeypatch):
    monkeypatch.setenv('VISIONGUARD_RUN', str(ROOT / 'runs/baseline'))
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception
    app.selectbox[0].select('Cardboard — training example').run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any(metric.label == 'Top predicted category' for metric in app.metric)
