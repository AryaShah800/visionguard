import argparse
import json
from pathlib import Path

from .common import ROOT


def main():
    parser = argparse.ArgumentParser(description='VisionGuard training, evaluation and inference')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('download')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--images', type=Path, default=ROOT / 'data/dataset-resized')
    prepare.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.json')
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--groups', type=Path)
    train = commands.add_parser('train')
    train.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.json')
    train.add_argument('--images', type=Path, default=ROOT / 'data/dataset-resized')
    train.add_argument('--run', type=Path, default=ROOT / 'runs/baseline')
    train.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--cnn-epochs', type=int, default=10)
    train.add_argument('--head-epochs', type=int, default=10)
    train.add_argument('--tune-epochs', type=int, default=8)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--patience', type=int, default=3)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.json')
    evaluate.add_argument('--images', type=Path, default=ROOT / 'data/dataset-resized')
    evaluate.add_argument('--run', type=Path, default=ROOT / 'runs/baseline')
    evaluate.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    predict = commands.add_parser('predict')
    predict.add_argument('image', type=Path)
    predict.add_argument('--run', type=Path, default=ROOT / 'runs/baseline')
    lock = commands.add_parser('lock-challenge')
    lock.add_argument('--images', type=Path, required=True)
    lock.add_argument('--manifest', type=Path, default=ROOT / 'data/challenge.json')
    lock.add_argument('--benchmark', type=Path, default=ROOT / 'data/manifest.json')
    challenge = commands.add_parser('evaluate-challenge')
    challenge.add_argument('--images', type=Path, required=True)
    challenge.add_argument('--manifest', type=Path, default=ROOT / 'data/challenge.json')
    challenge.add_argument('--run', type=Path, default=ROOT / 'runs/baseline')
    challenge.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.command == 'download':
        from .data import download
        print(download())
    elif args.command == 'prepare':
        from .data import prepare
        prepare(args.images, args.manifest, args.seed, args.groups)
    elif args.command == 'train':
        from .train import train_experiment
        train_experiment(args.manifest, args.images, args.run, args.device, args.seed,
                         args.cnn_epochs, args.head_epochs, args.tune_epochs, args.batch_size, args.patience)
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        evaluate(args.manifest, args.images, args.run, args.device)
    elif args.command == 'predict':
        from .predict import predict_file
        print(json.dumps(predict_file(args.image, args.run), indent=2))
    elif args.command == 'lock-challenge':
        from .challenge import lock_challenge
        lock_challenge(args.images, args.manifest, args.benchmark)
    else:
        from .challenge import evaluate_challenge
        evaluate_challenge(args.images, args.manifest, args.run, args.device)


if __name__ == '__main__':
    main()
