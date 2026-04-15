import argparse

parser = argparse.ArgumentParser(description="SCL Training for Respiratory Sound Classification")

# Generic
parser.add_argument("--method", type=str, default='hybrid',
                    help="Training method: sl, scl, mscl, hybrid")
parser.add_argument("--device", type=str, default="cuda:0",
                    help="Device to train on")
parser.add_argument("--workers", type=int, default=3,
                    help="Number of data loader workers")
parser.add_argument("--bs", type=int, default=128,
                    help="Batch size")
parser.add_argument("--epochs", type=int, default=400,
                    help="Number of training epochs")
parser.add_argument("--epochs2", type=int, default=100,
                    help="Epochs for linear eval (scl/mscl)")

# Model
parser.add_argument("--backbone", type=str, default='cnn6',
                    help="Backbone: cnn6, cnn10, cnn14")
parser.add_argument("--dropout", action='store_true',
                    help="Enable dropout in backbone")
parser.add_argument("--scratch", action='store_true',
                    help="Train from scratch (no pretrained weights)")
parser.add_argument("--weightspath", type=str, default='models/panns_weights',
                    help="Path to pretrained PANNs weights directory")

# Data
parser.add_argument("--dataset", type=str, default='ICBHI',
                    help="Dataset: ICBHI")
parser.add_argument("--datapath", type=str, default='data/ICBHI_final_database',
                    help="Path to dataset audio files")
parser.add_argument("--metadata", type=str, default='metadata.csv',
                    help="Metadata CSV filename (inside datapath)")
parser.add_argument("--metalabel", type=str, default='loc',
                    help="Metadata label for mscl: loc (chest location), equip (equipment)")
parser.add_argument("--samplerate", type=int, default=16000,
                    help="Target sampling rate")
parser.add_argument("--duration", type=int, default=8,
                    help="Max duration of audio clips in seconds")
parser.add_argument("--pad", type=str, default='circular',
                    help="Padding type: circular, zero")
parser.add_argument("--noweights", action='store_true',
                    help="Disable class weights for cross entropy")

# Optimizer
parser.add_argument("--wd", type=float, default=1e-4,
                    help="Weight decay")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate")
parser.add_argument("--lr2", type=float, default=1e-1,
                    help="Learning rate for linear eval")

# Data Augmentation
parser.add_argument("--freqmask", type=int, default=20,
                    help="Frequency mask size for SpecAugment")
parser.add_argument("--freqstripes", type=int, default=2,
                    help="Number of frequency masks")
parser.add_argument("--timemask", type=int, default=50,
                    help="Time mask size for SpecAugment")
parser.add_argument("--timestripes", type=int, default=2,
                    help="Number of time masks")

# Loss parameters
parser.add_argument("--tau", type=float, default=0.06,
                    help="Temperature for contrastive loss")
parser.add_argument("--alpha", type=float, default=0.5,
                    help="Tradeoff between CE and contrastive loss (hybrid)")
parser.add_argument("--lam", type=float, default=0.75,
                    help="Tradeoff between class SCL and metadata SCL (mscl)")

# Output
parser.add_argument("--save_dir", type=str, default='models/scl_model',
                    help="Directory to save trained model")
