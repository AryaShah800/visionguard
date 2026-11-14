"""HTTP inference API for the validation-selected VisionGuard checkpoint.

Local run (from the repository root):  uvicorn vg.api:app --port 8000
The model loads on the first /predict call (or at startup when VG_PRELOAD=1).
The run directory comes from VG_RUN_DIR (fallback VISIONGUARD_RUN, then runs/baseline).
Responses never include local file paths; details of load failures go to the server log only.
"""
import io
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from . import __version__
from .common import ROOT
from .predict import load_predictor, predict_image

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_PIXELS = 40_000_000  # well below Pillow's own 89 MP bomb warning; ~120 MB as decoded RGB
MIN_SIDE = 16
ALLOWED_FORMATS = ('JPEG', 'PNG', 'WEBP')
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/jpg', 'image/pjpeg', 'image/png', 'image/webp', 'application/octet-stream'}
DEFAULT_RUN_DIR = ROOT / 'runs' / 'baseline'

NOT_READY = 'Model is not ready. Point VG_RUN_DIR at a trained, verified run directory.'
UNSUPPORTED = 'Upload a JPEG, PNG or WebP image.'
INTERNAL = 'Internal server error.'

log = logging.getLogger('visionguard.api')


def too_large_message(limit):
    return f'Image must be {limit / (1024 * 1024):g} MiB or smaller.'


def run_dir_from_env():
    value = os.environ.get('VG_RUN_DIR') or os.environ.get('VISIONGUARD_RUN')
    return Path(value) if value else DEFAULT_RUN_DIR


class ModelUnavailable(Exception):
    """Raised when the checkpoint cannot be loaded; the message is safe to show clients."""


class LazyPredictor:
    """Loads the model once, on first use, and keeps it for the life of the process."""

    def __init__(self, loader, run_dir=None):
        self._loader, self._run_dir = loader, run_dir
        self._lock = threading.Lock()
        self._loaded = None
        self.failed = False

    @property
    def run_dir(self):
        return Path(self._run_dir) if self._run_dir is not None else run_dir_from_env()

    @property
    def loaded(self):
        return self._loaded is not None

    @property
    def kind(self):
        if self._loaded is None:
            return None
        checkpoint = self._loaded[1]
        return checkpoint.get('kind') if isinstance(checkpoint, dict) else None

    def ready(self):
        if self.loaded:
            return True
        try:
            return not self.failed and (self.run_dir / 'selection.json').is_file()
        except OSError:
            return False

    def get(self):
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    try:
                        self._loaded = self._loader(self.run_dir)
                    except Exception:
                        self.failed = True
                        log.exception('VisionGuard model failed to load')  # server log only
                        raise ModelUnavailable(NOT_READY) from None
                    self.failed = False
        return self._loaded


class BodyLimitMiddleware:
    """Rejects request bodies over the limit before they are parsed or spooled to disk.

    Checks Content-Length up front and also counts streamed bytes, so chunked uploads
    without a Content-Length header cannot bypass the cap.
    """

    def __init__(self, app, max_bytes, message):
        self.app, self.max_bytes, self.message = app, max_bytes, message

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        length = dict(scope.get('headers') or []).get(b'content-length')
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                await self._reject(send, 400, 'Invalid Content-Length header.')
                return
            if declared > self.max_bytes:
                await self._reject(send, 413, self.message)
                return
        received, started = 0, False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.max_bytes:
                    raise HTTPException(413, self.message)
            return message

        async def tracking_send(message):
            nonlocal started
            if message['type'] == 'http.response.start':
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except HTTPException as error:
            if error.status_code != 413 or started:
                raise
            await self._reject(send, 413, self.message)

    @staticmethod
    async def _reject(send, status, detail):
        body = json.dumps({'detail': detail}).encode()
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-type', b'application/json'), (b'content-length', str(len(body)).encode())]})
        await send({'type': 'http.response.body', 'body': body})


def decode_image(data, max_pixels):
    """Opens an upload safely: allowed formats only, size checked before pixels are decoded."""
    try:
        image = Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS)
    except UnidentifiedImageError:
        raise HTTPException(415, UNSUPPORTED) from None
    except Image.DecompressionBombError:
        raise HTTPException(413, 'Image dimensions are too large.') from None
    except (OSError, ValueError, SyntaxError, EOFError):
        raise HTTPException(400, 'The image file is corrupt or truncated.') from None
    try:
        width, height = image.size
        if width * height > max_pixels:
            raise HTTPException(413, 'Image dimensions are too large.')
        if min(width, height) < MIN_SIDE:
            raise HTTPException(422, f'Use an image at least {MIN_SIDE} by {MIN_SIDE} pixels.')
        # Phone JPEGs can carry extra MPO frames (previews/depth); only the first is used, so allow them.
        if image.format in ('PNG', 'WEBP') and getattr(image, 'is_animated', False):
            raise HTTPException(415, 'Animated images are not supported.')
        image.load()
    except HTTPException:
        image.close()
        raise
    except Image.DecompressionBombError:
        image.close()
        raise HTTPException(413, 'Image dimensions are too large.') from None
    except (OSError, ValueError, SyntaxError, EOFError):
        image.close()
        raise HTTPException(400, 'The image file is corrupt or truncated.') from None
    return image


def create_app(loader=load_predictor, predict=predict_image, run_dir=None,
               max_upload_bytes=MAX_UPLOAD_BYTES, max_pixels=MAX_PIXELS, preload=None):
    """Builds the API. `loader(run_dir) -> (model, checkpoint)` and `predict(image, model, checkpoint)`
    default to vg.predict; tests inject fakes. `run_dir=None` reads VG_RUN_DIR at load time."""
    holder = LazyPredictor(loader, run_dir)
    too_large = too_large_message(max_upload_bytes)

    @asynccontextmanager
    async def lifespan(app):
        wanted = preload if preload is not None else os.environ.get('VG_PRELOAD', '').lower() in {'1', 'true', 'yes'}
        if wanted:
            try:
                await run_in_threadpool(holder.get)
            except ModelUnavailable:
                log.warning('Model not ready at startup; /health reports ready=false.')
        yield

    app = FastAPI(title='VisionGuard API', version=__version__, lifespan=lifespan,
                  description='Waste-image classification (six TrashNet categories). '
                              'Scores are softmax outputs, not calibrated probabilities of correctness.')
    app.state.predictor = holder
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_upload_bytes + MULTIPART_OVERHEAD_BYTES, message=too_large)

    @app.exception_handler(ModelUnavailable)
    async def model_unavailable(request: Request, error: ModelUnavailable):
        return JSONResponse({'detail': NOT_READY}, status_code=503)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        log.exception('Unhandled error on %s', request.url.path)
        return JSONResponse({'detail': INTERNAL}, status_code=500)

    @app.get('/health')
    def health():
        return {'status': 'ok', 'version': __version__, 'ready': holder.ready(),
                'model_loaded': holder.loaded, 'model': holder.kind}

    def classify(data):
        image = decode_image(data, max_pixels)
        try:
            model, checkpoint = holder.get()
            try:
                return predict(image, model, checkpoint)
            except ValueError:
                raise HTTPException(422, 'The image could not be classified.') from None
        finally:
            image.close()

    @app.post('/predict')
    async def predict_endpoint(file: UploadFile = File(...)):
        content_type = (file.content_type or '').split(';')[0].strip().lower()
        if content_type and content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(415, UNSUPPORTED)
        data = await file.read(max_upload_bytes + 1)
        await file.close()
        if len(data) > max_upload_bytes:
            raise HTTPException(413, too_large)
        if not data:
            raise HTTPException(400, 'The uploaded file is empty.')
        result = await run_in_threadpool(classify, data)
        return {'prediction': result['prediction'], 'abstained': bool(result['uncertain']),
                'top_three': [{'class': item['class'], 'score': float(item['score'])} for item in result['top_three'][:3]],
                'threshold': result.get('threshold'), 'model': result.get('model'), 'note': result.get('note')}

    return app


app = create_app()
