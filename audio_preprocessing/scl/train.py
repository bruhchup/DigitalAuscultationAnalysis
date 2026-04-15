"""Main training entry point for SCL respiratory sound classification.

Usage:
    python -m audio_preprocessing.scl.train --method hybrid --backbone cnn6 --epochs 400

Methods:
    sl      - Standard cross-entropy supervised learning
    scl     - Supervised contrastive learning + linear eval
    mscl    - Multi-head supervised contrastive (with metadata) + linear eval
    hybrid  - Joint contrastive + cross-entropy
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torchaudio import transforms as T

from .args import parser
from .models import CNN6, CNN10, CNN14, Projector, LinearClassifier, model_dict
from .dataset import ICBHI
from .augmentations import SpecAugment
from .utils import Normalize, Standardize
from .losses import SupConLoss, SupConCELoss
from .ce import train_ce
from .hybrid import train_supconce
from .scl import train_scl, linear_scl
from .mscl import train_mscl, linear_mscl


def main():
    args = parser.parse_args()

    print(f"Method: {args.method}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.bs}")
    print(f"Learning rate: {args.lr}")

    # Check for metadata CSV
    csv_path = os.path.join(args.datapath, args.metadata)
    if not os.path.isfile(csv_path):
        print(f"ERROR: Metadata CSV not found at {csv_path}")
        print(f"Run: python -m audio_preprocessing.scl.generate_metadata --datadir {args.datapath}")
        sys.exit(1)

    # Constants
    NUM_CLASSES = 4
    OUT_DIM = 128
    NFFT = 1024
    NMELS = 64
    WIN_LENGTH = 1024
    HOP_LENGTH = 512
    FMIN = 50
    FMAX = 2000

    # Model definition
    embed_only = args.method != 'sl'

    if args.backbone == 'cnn6':
        path_to_weights = os.path.join(args.weightspath, 'Cnn6_mAP=0.343.pth')
        model = CNN6(num_classes=NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only,
                     from_scratch=args.scratch, path_to_weights=path_to_weights, device=args.device)
    elif args.backbone == 'cnn10':
        path_to_weights = os.path.join(args.weightspath, 'Cnn10_mAP=0.380.pth')
        model = CNN10(num_classes=NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only,
                      from_scratch=args.scratch, path_to_weights=path_to_weights, device=args.device)
    elif args.backbone == 'cnn14':
        path_to_weights = os.path.join(args.weightspath, 'Cnn14_mAP=0.431.pth')
        model = CNN14(num_classes=NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only,
                      from_scratch=args.scratch, path_to_weights=path_to_weights, device=args.device)
    else:
        print(f"Unknown backbone: {args.backbone}")
        sys.exit(1)

    if embed_only:
        projector = Projector(name=args.backbone, out_dim=OUT_DIM, device=args.device)
        if args.method == 'mscl':
            projector2 = Projector(name=args.backbone, out_dim=OUT_DIM, device=args.device)
        classifier = LinearClassifier(name=args.backbone, num_classes=NUM_CLASSES, device=args.device)

    # Spectrogram transforms
    melspec = T.MelSpectrogram(
        n_fft=NFFT, n_mels=NMELS, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, f_min=FMIN, f_max=FMAX
    ).to(args.device)
    normalize = Normalize()
    melspec_norm = torch.nn.Sequential(melspec, normalize)
    standardize = Standardize(device=args.device)

    specaug = SpecAugment(
        freq_mask=args.freqmask, time_mask=args.timemask,
        freq_stripes=args.freqstripes, time_stripes=args.timestripes
    ).to(args.device)

    train_transform = nn.Sequential(melspec_norm, specaug, standardize)
    val_transform = nn.Sequential(melspec_norm, standardize)

    # Dataset and dataloaders
    print(f"\nLoading ICBHI dataset from {args.datapath}...")
    train_ds = ICBHI(
        data_path=args.datapath, metadatafile=args.metadata, duration=args.duration,
        split='train', device=args.device, samplerate=args.samplerate,
        pad_type=args.pad, meta_label=args.metalabel
    )
    val_ds = ICBHI(
        data_path=args.datapath, metadatafile=args.metadata, duration=args.duration,
        split='test', device=args.device, samplerate=args.samplerate,
        pad_type=args.pad, meta_label=args.metalabel
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Optimizer
    if args.method == 'sl':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.method == 'scl':
        optimizer = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()),
                                     lr=args.lr, weight_decay=args.wd)
        optimizer2 = torch.optim.Adam(classifier.parameters(), lr=args.lr2, weight_decay=args.wd)
    elif args.method == 'mscl':
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(projector.parameters()) + list(projector2.parameters()),
            lr=args.lr, weight_decay=args.wd)
        optimizer2 = torch.optim.Adam(classifier.parameters(), lr=args.lr2, weight_decay=args.wd)
    elif args.method == 'hybrid':
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
            lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

    # Class weights for cross-entropy
    if args.noweights:
        criterion_ce = nn.CrossEntropyLoss()
    else:
        weights = torch.tensor([2063, 1215, 501, 363], dtype=torch.float32)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        weights = weights.to(args.device)
        criterion_ce = nn.CrossEntropyLoss(weight=weights)

    # Train
    print(f"\n{'='*60}")
    print(f"Starting {args.method.upper()} training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    if args.method == 'sl':
        best_state, best_score, best_se, best_sp = train_ce(
            model, train_loader, val_loader, train_transform, val_transform,
            criterion_ce, optimizer, args.epochs, scheduler, args.device, NUM_CLASSES)

    elif args.method == 'scl':
        criterion = SupConLoss(temperature=args.tau, device=args.device)
        _, model, checkpoint = train_scl(
            model, projector, train_loader, train_transform,
            criterion, optimizer, scheduler, args.epochs, args.device)
        print(f"\n{'='*60}")
        print(f"Starting linear evaluation for {args.epochs2} epochs")
        print(f"{'='*60}\n")
        best_state, best_score, best_se, best_sp = linear_scl(
            model, checkpoint, classifier, train_loader, val_loader,
            val_transform, criterion_ce, optimizer2, args.epochs2, args.device, NUM_CLASSES)

    elif args.method == 'mscl':
        criterion = SupConLoss(temperature=args.tau, device=args.device)
        _, model, checkpoint = train_mscl(
            model, projector, projector2, train_loader, train_transform,
            criterion, optimizer, scheduler, args.epochs, args.lam, args.device)
        print(f"\n{'='*60}")
        print(f"Starting linear evaluation for {args.epochs2} epochs")
        print(f"{'='*60}\n")
        best_state, best_score, best_se, best_sp = linear_mscl(
            model, checkpoint, classifier, train_loader, val_loader,
            val_transform, criterion_ce, optimizer2, args.epochs2, args.device, NUM_CLASSES)

    elif args.method == 'hybrid':
        criterion = SupConCELoss(temperature=args.tau, weights=weights, alpha=args.alpha, device=args.device)
        best_state, best_score, best_se, best_sp = train_supconce(
            model, projector, classifier, train_loader, val_loader,
            train_transform, val_transform, criterion, criterion_ce,
            optimizer, args.epochs, scheduler, args.device, NUM_CLASSES)

    # Save model and config
    os.makedirs(args.save_dir, exist_ok=True)

    # Save model weights
    torch.save(best_state, os.path.join(args.save_dir, 'model_state.pth'))
    print(f"\nModel saved to {args.save_dir}/model_state.pth")

    # Save preprocessing config for inference
    config = {
        "model_type": "pytorch",
        "model_name": f"{args.backbone}_{args.method}",
        "backbone": args.backbone,
        "method": args.method,
        "TARGET_SR": args.samplerate,
        "TARGET_DURATION": args.duration,
        "TARGET_LENGTH": args.samplerate * args.duration,
        "FILTER_LOWCUT": FMIN,
        "FILTER_HIGHCUT": FMAX,
        "N_MELS": NMELS,
        "N_FFT": NFFT,
        "HOP_LENGTH": HOP_LENGTH,
        "WIN_LENGTH": WIN_LENGTH,
        "CLASS_NAMES": ["Normal", "Crackle", "Wheeze", "Both"],
        "NUM_CLASSES": NUM_CLASSES,
        "embed_only": embed_only,
    }
    config_path = os.path.join(args.save_dir, 'preprocessing_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Save results
    results = {
        "method": args.method,
        "backbone": args.backbone,
        "best_icbhi_score": best_score,
        "best_sensitivity": best_se,
        "best_specificity": best_sp,
        "epochs": args.epochs,
        "batch_size": args.bs,
        "learning_rate": args.lr,
    }
    results_path = os.path.join(args.save_dir, 'model_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best ICBHI Score: {best_score:.4f}")
    print(f"Sensitivity: {best_se:.4f}")
    print(f"Specificity: {best_sp:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
