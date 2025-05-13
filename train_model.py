"""
Train the FathomNetClassifier.

Author: Ben Jafek
2025/04/28
"""

import os
import json
import random

from PIL import Image
import pandas as pd
import torch
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from benj_prac.fathomnet.create_subset import create_subset_json
from benj_prac.fathomnet.fathomnet_dataset import FathomNetDataset
from benj_prac.fathomnet.classifier import FathomNetClassifier
from benj_prac.fathomnet import preprocessing

FULL_JSON = '/home/bjafek/Nuro/benj_prac/fathomnet/data/dataset_train.json'
SUBSET_JSON = '/home/bjafek/Nuro/benj_prac/fathomnet/data/dataset_train_subset.json'
SUBSET_SIZE = 200

ANNOTATIONS = "/home/bjafek/Nuro/benj_prac/fathomnet/data/train/annotations.csv"
DOWNSAMPLE = False

TRAIN_OUT_DIR = "/home/bjafek/Nuro/benj_prac/fathomnet/output/sixth"

BATCH_SIZE = 16  # maxes out my VDI


def main():
    # Enable downsampling training data to be downloaded to speed up experimentation and testing
    if DOWNSAMPLE:
        create_subset_json(FULL_JSON, SUBSET_JSON, SUBSET_SIZE)

    # Load labels
    train_annotations_df = pd.read_csv(ANNOTATIONS)
    label_encoder = LabelEncoder().fit(train_annotations_df["label"].dropna())
    print(train_annotations_df.head())
    num_classes = len(label_encoder.classes_)

    # Load and transform the training dataset
    full_dataset = FathomNetDataset(
        csv_path=ANNOTATIONS,
        transform=preprocessing.TRAIN_TRANSFORMS,
        label_encoder=label_encoder,
    )

    # Use encoded labels (self.label_ids) for stratification
    targets = full_dataset.label_ids

    # Perform stratified split (80% train, 20% val)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X=targets, y=targets))

    # Create PyTorch subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset.dataset.transform = preprocessing.VAL_TRANSFORMS
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FathomNetClassifier(num_classes=num_classes)

    # Setup PyTorch Lightning Trainer with Early Stopping and Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}',
        dirpath=TRAIN_OUT_DIR,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        min_delta=0.001,
    )
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback,
        ]
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
