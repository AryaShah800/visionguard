"""Local image demo; all predictions use the validation-selected checkpoint."""
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from vg.common import CLASSES, ROOT, read_json
from vg.predict import load_predictor, predict_image

st.set_page_config(page_title='VisionGuard | Waste image lab', page_icon='♻', layout='wide')
RUN = Path(os.environ.get('VISIONGUARD_RUN', str(ROOT / 'runs/baseline')))
st.markdown('### ♻ VisionGuard')
st.title('A closer look at waste.')
st.write('Explore what a trained vision model sees—and where it gets things wrong.')
st.caption('Six material categories · One prominent item per photo · Local inference')


@st.cache_resource
def predictor(run_path, checkpoint_version):
    return load_predictor(Path(run_path))


with st.sidebar:
    st.header('The experiment')
    st.write('Small CNN → frozen MobileNetV2 → fine-tuned MobileNetV2')
    st.caption('The final model is selected by validation macro-F1 before the test set is evaluated.')
    st.divider()
    st.write('**Recognized categories**')
    st.write('Cardboard · Glass · Metal · Paper · Plastic · Trash')
    st.info('Predicts visual categories. Local recycling rules, mixed materials and unfamiliar objects require human judgment.')

classify, experiment = st.tabs(['Classify an image', 'Results & limitations'])

with classify:
    left, right = st.columns([1.1, 1], gap='large')
    with left:
        uploaded = st.file_uploader('Upload a photo', type=['jpg', 'jpeg', 'png', 'webp'],
                                    help='Use one clearly visible item. Maximum file size: 10 MB.')
        samples = {}
        manifest_path = ROOT / 'data/manifest.json'
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            for label in CLASSES:
                row = next((r for r in manifest['rows'] if r['label'] == label and r['split'] == 'train'), None)
                if row:
                    samples[f"{label.title()} — training example"] = ROOT / 'data/dataset-resized' / row['path']
        sample = st.selectbox('Or explore a training example', ['Choose an example'] + list(samples))
        st.caption('Examples are from training data and do not demonstrate generalization. An upload takes priority.')
        image = None
        try:
            if uploaded is not None:
                if uploaded.size > 10 * 1024 * 1024:
                    st.error('Please upload a file smaller than 10 MB.')
                else:
                    image = Image.open(io.BytesIO(uploaded.getvalue()))
                    image.load()
            elif sample in samples:
                with Image.open(samples[sample]) as source:
                    image = source.copy()
            if image is not None:
                st.image(image, caption='Your image', width='stretch')
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            image = None
            st.error(f'Could not read this image: {error}')
    with right:
        st.subheader('Model prediction')
        if not (RUN / 'selection.json').exists():
            st.info('The model is not ready yet. Finish training with python -m vg train, then refresh this page.')
        elif image is None:
            st.info('Upload a photo or select an example to get started.')
        elif st.button('Classify image', type='primary', use_container_width=True):
            try:
                version = (RUN / 'selection.json').stat().st_mtime_ns
                model, checkpoint = predictor(str(RUN.resolve()), version)
                with st.spinner('Examining the image…'):
                    result = predict_image(image, model, checkpoint)
                if result['uncertain']:
                    st.warning('Uncertain — review this image manually.')
                st.metric('Top predicted category', result['prediction'].title())
                st.caption(f"Model: {result['model']} · Softmax scores, not calibrated probabilities of correctness")
                scores = pd.DataFrame(result['top_three']).rename(columns={'class': 'Category', 'score': 'Model score'})
                st.bar_chart(scores.set_index('Category'), color='#26A77A', horizontal=True)
                st.dataframe(scores, hide_index=True, use_container_width=True)
                st.caption('A high model score can still be wrong, especially on cluttered scenes or unseen materials.')
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f'Prediction could not complete: {error}')

with experiment:
    metrics_path = RUN / 'evaluation/metrics.json'
    if metrics_path.exists():
        report = read_json(metrics_path)
        selected = report['selected_model']
        measured = report['models'][selected]
        first, second, third = st.columns(3)
        first.metric('Test macro-F1', f"{measured['macro_f1']:.3f}")
        second.metric('Test accuracy', f"{measured['accuracy']:.1%}")
        third.metric('Held-out images', measured['count'])
        st.caption(f'Selected using validation only: {selected}. One training seed; scores are specific to this split.')
        table = [{'Model': name, 'Test macro-F1': values['macro_f1'], 'Test accuracy': values['accuracy']}
                 for name, values in report['models'].items()]
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        st.image(str(RUN / 'evaluation/confusion_matrix.png'), caption='Held-out test confusion matrix', width=650)
        with st.expander('Training curves and example mistakes'):
            st.image(str(RUN / 'evaluation/training_curves.png'))
            if (RUN / 'evaluation/errors.png').exists():
                st.image(str(RUN / 'evaluation/errors.png'))
        st.download_button('Download measured metrics', metrics_path.read_text(encoding='utf-8'),
                           file_name='visionguard-metrics.json', mime='application/json')
    else:
        st.info('The experiment report will appear after python -m vg evaluate completes.')
    st.subheader('What this experiment does—and does not establish')
    st.write('TrashNet images generally show a single item on a simple background. '
             'Performance on this benchmark does not establish reliability on everyday phone photos.')
    challenge = RUN / 'challenge-evaluation/metrics.json'
    if challenge.exists():
        independent = read_json(challenge)
        st.success(f"Independent photo evaluation: {independent['count']} images, macro-F1 {independent['macro_f1']:.3f}.")
    else:
        st.warning('Independent phone-photo evaluation is pending. No real-world performance claim is made.')
    st.caption('Dataset: Gary Thung and Mindy Yang, TrashNet. https://github.com/garythung/trashnet')
