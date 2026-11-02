"""Audit images and split duplicate groups before applying augmentation."""
import csv
import hashlib
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights

from .common import CLASSES, ROOT, read_json, save_json, sha256

DATA_URL = 'https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip'
ARCHIVE_SHA256 = '0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def download():
    directory = ROOT / 'data'
    directory.mkdir(exist_ok=True)
    archive = directory / 'dataset-resized.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(DATA_URL, timeout=120) as response, temporary.open('wb') as file:
            while block := response.read(1024 * 1024):
                file.write(block)
        temporary.replace(archive)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Dataset archive differs from the verified experiment version; review it before use.')
    with zipfile.ZipFile(archive) as bundle:
        # Extract only expected images, never arbitrary archive paths.
        for member in bundle.infolist():
            parts = Path(member.filename).parts
            if (len(parts) == 3 and parts[0] == 'dataset-resized'
                    and parts[1] in CLASSES and Path(parts[2]).suffix.lower() in EXTENSIONS):
                destination = directory.joinpath(*parts)
                destination.parent.mkdir(exist_ok=True, parents=True)
                if not destination.exists():
                    destination.write_bytes(bundle.read(member))
    save_json(directory / 'source.json', {'url': DATA_URL, 'archive_sha256': sha256(archive),
              'attribution': 'Gary Thung and Mindy Yang, TrashNet, https://github.com/garythung/trashnet'})
    return directory / 'dataset-resized'


def open_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def image_hashes(path):
    image = open_rgb(path)
    pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    gray = np.asarray(image.convert('L').resize((32, 32)), dtype=float)
    low = dctn(gray, norm='ortho')[:8, :8].flatten()[1:]
    bits = low > np.median(low)
    perceptual = sum(int(bit) << i for i, bit in enumerate(bits))
    return pixels, perceptual


def grouped_split(rows, seed=42):
    """Choose approximately stratified group partitions without using model scores."""
    y = np.array([CLASSES.index(row['label']) for row in rows])
    groups = np.array([row['group'] for row in rows])

    def choose(indices, fraction, random_state):
        splitter = GroupShuffleSplit(n_splits=200, test_size=fraction, random_state=random_state)
        target = np.bincount(y[indices], minlength=len(CLASSES)) * fraction
        best = None
        for train, hold in splitter.split(indices, y[indices], groups[indices]):
            counts = np.bincount(y[indices[hold]], minlength=len(CLASSES))
            remaining = np.bincount(y[indices[train]], minlength=len(CLASSES))
            if min(counts) == 0 or min(remaining) == 0:
                continue
            score = np.mean(np.abs(counts - target) / np.maximum(target, 1))
            if best is None or score < best[0]:
                best = (score, indices[train], indices[hold])
        if best is None:
            raise ValueError('Too few independent groups per class to create all partitions.')
        return best[1], best[2]

    development, test = choose(np.arange(len(rows)), .15, seed)
    train, validation = choose(development, .15 / .85, seed + 1)
    for name, indices in [('train', train), ('validation', validation), ('test', test)]:
        for index in indices:
            rows[index]['split'] = name
    validate_partitions(rows)
    return rows


def validate_partitions(rows):
    for field in ['path', 'sha256', 'pixel_sha256', 'group']:
        seen = {}
        for row in rows:
            key = row[field]
            if key in seen and seen[key] != row['split']:
                raise ValueError(f'{field} leakage across splits: {key}')
            seen[key] = row['split']
    for split in ['train', 'validation', 'test']:
        if {r['label'] for r in rows if r['split'] == split} != set(CLASSES):
            raise ValueError(f'{split} must contain all six classes')


def prepare(image_root, output, seed=42, groups_file=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists. Use another path to preserve the locked split.')
    image_root = Path(image_root).resolve()
    rows, excluded = [], []
    for label in CLASSES:
        for path in sorted((image_root / label).rglob('*')):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                pixel, perceptual = image_hashes(path)
                rows.append({'path': path.relative_to(image_root).as_posix(), 'label': label,
                             'sha256': sha256(path), 'pixel_sha256': pixel, 'phash': perceptual})
            except (OSError, ValueError) as error:
                excluded.append({'path': str(path), 'reason': str(error)})
    if not rows:
        raise ValueError(f'No class images found in {image_root}')
    raw_count = len(rows)
    by_pixel = {}
    for row in rows:
        by_pixel.setdefault(row['pixel_sha256'], []).append(row)
    rows = []
    for matches in by_pixel.values():
        if len({r['label'] for r in matches}) > 1:
            excluded.extend({'path': r['path'], 'reason': 'conflicting exact duplicate'} for r in matches)
        else:
            rows.append(matches[0])
            excluded.extend({'path': r['path'], 'reason': 'exact duplicate'} for r in matches[1:])
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    near_pairs = []
    for i, row in enumerate(rows):
        for j in range(i):
            distance = (row['phash'] ^ rows[j]['phash']).bit_count()
            if distance <= 4:
                union(i, j)
                near_pairs.append({'first': rows[j]['path'], 'second': row['path'], 'distance': distance})
    if groups_file:
        with Path(groups_file).open(encoding='utf-8-sig', newline='') as file:
            identities = {r['path']: r['item_id'] for r in csv.DictReader(file)}
        unknown = set(identities) - {r['path'] for r in rows}
        if unknown:
            raise ValueError(f'Unknown group paths: {sorted(unknown)[:3]}')
        seen = {}
        for i, row in enumerate(rows):
            identity = identities.get(row['path'])
            if identity:
                if identity in seen:
                    union(seen[identity], i)
                seen[identity] = i
    for i, row in enumerate(rows):
        row['group'] = f'g{find(i):05d}'
    rows = grouped_split(rows, seed)
    payload = {'schema': 1, 'classes': CLASSES, 'seed': seed, 'rows': rows,
               'grouping': 'Exact pixel duplicates removed; pHash distance <=4 grouped; optional item IDs.',
               'limitation': 'No complete physical-object identity metadata; pHash cannot catch every near-duplicate.'}
    save_json(output, payload)
    audit = {'raw_valid_images': raw_count, 'kept_images': len(rows), 'excluded': excluded,
             'near_duplicate_pairs': near_pairs, 'manifest_sha256': sha256(output),
             'splits': {name: dict(Counter(r['label'] for r in rows if r['split'] == name))
                        for name in ['train', 'validation', 'test']}}
    save_json(output.with_name(output.stem + '-audit.json'), audit)
    print({key: value for key, value in audit.items() if key not in ['excluded', 'near_duplicate_pairs']})
    return payload


def load_manifest(path, image_root, verify=True):
    manifest = read_json(path)
    if manifest['classes'] != CLASSES:
        raise ValueError('Class mapping mismatch')
    validate_partitions(manifest['rows'])
    root = Path(image_root).resolve()
    for row in manifest['rows']:
        target = (root / row['path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Image path escapes dataset root')
        if verify and sha256(target) != row['sha256']:
            raise ValueError(f"Image changed since split was locked: {row['path']}")
    return manifest


def transform_for(training=False):
    if not training:
        return MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.70, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class WasteDataset(Dataset):
    def __init__(self, rows, root, training=False):
        self.rows, self.root = rows, Path(root)
        self.transform = transform_for(training)
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if index not in self.cache:
            image = open_rgb(self.root / row['path'])
            image.thumbnail((512, 512))
            self.cache[index] = image
        return self.transform(self.cache[index]), CLASSES.index(row['label'])
