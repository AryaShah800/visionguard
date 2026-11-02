import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .common import CLASSES, ROOT


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, channels = [], 3
        for output in [16, 32, 64, 128]:
            blocks.extend([nn.Conv2d(channels, output, 3, padding=1), nn.BatchNorm2d(output),
                           nn.ReLU(), nn.MaxPool2d(2)])
            channels = output
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(),
                                        nn.Dropout(.3), nn.Linear(128, len(CLASSES)))

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(kind, pretrained=False):
    if kind == 'cnn':
        return SmallCNN()
    if kind not in ['frozen', 'finetuned']:
        raise ValueError(f'Unknown model {kind}')
    torch.hub.set_dir(str(ROOT / 'data' / 'torch-cache'))
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if kind == 'finetuned':
        for parameter in model.features[14:].parameters():
            parameter.requires_grad = True
    return model


def training_mode(model, kind):
    model.train()
    if kind in ['frozen', 'finetuned']:
        # Frozen BatchNorm statistics must not drift during head training.
        model.features.eval()
        if kind == 'finetuned':
            model.features[14:].train()


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES:
        raise ValueError('Checkpoint class mapping mismatch')
    model = build_model(checkpoint['kind'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint
