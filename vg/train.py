import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .common import CLASSES, device_for, save_json, seed_everything, sha256
from .data import WasteDataset, load_manifest
from .metrics import classification_metrics, select_threshold, selective_metrics
from .models import build_model, load_model, training_mode


@torch.inference_mode()
def infer_batches(model, loader, device):
    model.eval()
    targets, scores = [], []
    for images, labels in loader:
        output = model(images.to(device))
        scores.append(output.softmax(dim=1).cpu().numpy())
        targets.extend(labels.tolist())
    return np.array(targets), np.concatenate(scores)


def train_experiment(manifest_path, image_root, run_dir, device='auto', seed=42,
                     cnn_epochs=10, head_epochs=10, tune_epochs=8, batch_size=32, patience=3):
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('Use a new run directory; existing experiments are immutable.')
    if min(cnn_epochs, head_epochs, tune_epochs, batch_size, patience) < 1:
        raise ValueError('Epochs, batch size and patience must be positive')
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = device_for(device)
    manifest = load_manifest(manifest_path, image_root)
    training = [r for r in manifest['rows'] if r['split'] == 'train']
    validation = [r for r in manifest['rows'] if r['split'] == 'validation']
    config = {'seed': seed, 'cnn_epochs': cnn_epochs, 'head_epochs': head_epochs,
              'tune_epochs': tune_epochs, 'patience': patience, 'batch_size': batch_size,
              'device': str(device), 'hardware': torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor(),
              'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'torchvision': str(torchvision.__version__), 'numpy': np.__version__},
              'manifest_sha256': sha256(manifest_path), 'classes': CLASSES,
              'selection': 'validation macro-F1', 'input_size': 224,
              'weights': 'MobileNet_V2_Weights.IMAGENET1K_V2',
              'loss': 'class-weighted cross entropy; weights from training only',
              'optimizer': 'AdamW, weight_decay=0.0001',
              'learning_rates': {'cnn': .001, 'head': .001, 'finetune_backbone': .0001, 'finetune_head': .0005},
              'augmentation': 'RandomResizedCrop(224, scale=.7..1), horizontal flip, rotation 15, color jitter .15/.15/.1',
              'determinism': 'Seeded; deterministic cuDNN; exact equality across hardware/versions is not guaranteed.'}
    save_json(run_dir / 'config.json', config)
    val_loader = DataLoader(WasteDataset(validation, image_root), batch_size=batch_size, num_workers=0)
    counts = np.bincount([CLASSES.index(r['label']) for r in training], minlength=len(CLASSES))
    weights = torch.tensor(len(training) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    majority = int(np.argmax(counts))
    truth = np.array([CLASSES.index(r['label']) for r in validation])
    comparisons = {'majority': {'validation': classification_metrics(truth, np.full(len(truth), majority)),
                                'majority_class': majority}}
    for kind, epochs in [('cnn', cnn_epochs), ('frozen', head_epochs), ('finetuned', tune_epochs)]:
        seed_everything(seed)
        model = build_model(kind, pretrained=(kind == 'frozen'))
        if kind == 'finetuned':
            previous = torch.load(run_dir / 'frozen.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(previous['state_dict'])
        model.to(device)
        if kind == 'finetuned':
            parameters = [{'params': model.features[14:].parameters(), 'lr': .0001},
                          {'params': model.classifier.parameters(), 'lr': .0005}]
        else:
            parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
        loader = DataLoader(WasteDataset(training, image_root, training=True), batch_size=batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        best, waiting, history = -1, 0, []
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            training_mode(model, kind)
            total_loss, total_weight, train_labels, train_predictions = 0.0, 0.0, [], []
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_weight = weights[labels].sum().item()
                total_loss += loss.item() * batch_weight
                total_weight += batch_weight
                train_labels.extend(labels.cpu().tolist())
                train_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            actual, scores = infer_batches(model, val_loader, device)
            metrics = classification_metrics(actual, scores.argmax(axis=1))
            # Validation loss uses the same class weights as training.
            per_example = -np.log(np.clip(scores[np.arange(len(actual)), actual], 1e-12, 1))
            val_weights = weights.detach().cpu().numpy()[actual]
            val_loss = float(np.sum(per_example * val_weights) / np.sum(val_weights))
            entry = {'epoch': epoch, 'train_loss': total_loss / total_weight, 'validation_loss': val_loss,
                     'train_accuracy': float(np.mean(np.array(train_labels) == train_predictions)),
                     'validation_accuracy': metrics['accuracy'], 'validation_macro_f1': metrics['macro_f1'],
                     'seconds': time.perf_counter() - start}
            history.append(entry)
            save_json(run_dir / f'{kind}-history.json', history)
            print(f"{kind} epoch {epoch}/{epochs}: val macro-F1={metrics['macro_f1']:.4f} "
                  f"accuracy={metrics['accuracy']:.4f} ({entry['seconds']:.1f}s)", flush=True)
            if metrics['macro_f1'] > best:
                best, waiting = metrics['macro_f1'], 0
                threshold = select_threshold(actual, scores)
                checkpoint = {'kind': kind, 'classes': CLASSES, 'epoch': epoch,
                              'manifest_sha256': config['manifest_sha256'], 'threshold': threshold,
                              'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
                torch.save(checkpoint, run_dir / f'{kind}.pt')
                comparisons[kind] = {'validation': metrics, 'best_epoch': epoch,
                                     'threshold': threshold, 'selective_validation': selective_metrics(actual, scores, threshold)}
            else:
                waiting += 1
            if waiting >= patience:
                break
        del model, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    selected = max(['cnn', 'frozen', 'finetuned'], key=lambda k: comparisons[k]['validation']['macro_f1'])
    summary = {'selected_model': selected, 'models': comparisons,
               'manifest_sha256': config['manifest_sha256'], 'seed': seed,
               'checkpoint_sha256': {kind: sha256(run_dir / f'{kind}.pt') for kind in ['cnn', 'frozen', 'finetuned']},
               'test_evaluated': False}
    save_json(run_dir / 'selection.json', summary)
    print(f'Selected on validation: {selected}. Test set has not been evaluated.', flush=True)
    return summary
