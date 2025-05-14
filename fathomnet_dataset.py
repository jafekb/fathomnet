import random

import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class FathomNetDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images and labels from a CSV file for training or testing.

    Args:
        csv_path (str): Path to the CSV file containing image paths and labels.
        transform (callable, optional): Optional image transformations to apply.
        label_encoder (LabelEncoder, optional): Optional pre-fitted LabelEncoder for consistent label mapping.
        is_test (bool): Flag indicating whether the dataset is for testing (no labels).

    CSV Requirements:
        - Must contain a "path" column with image file paths.
        - If not in test mode, must contain a "label" column with class labels.

    Returns:
        - If `is_test` is False: a tuple (image, encoded_label).
        - If `is_test` is True: a tuple (image, image_path).
    """

    def __init__(self, csv_path, transform=None, label_encoder=None, is_test=False, n_classes_subset=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test = is_test

        if n_classes_subset is not None:
            all_classes = set(self.data["label"])
            assert n_classes_subset <= len(all_classes)
            random_classes = random.sample(all_classes, n_classes_subset)
            self.data = self.data[self.data["label"].isin(random_classes)]
            print (f"Sampling data to {n_classes_subset} classes: {sorted(random_classes)}")

        self.image_paths = self.data["path"].tolist()

        if not is_test:
            self.labels = self.data["label"].tolist()
            # Use provided label encoder or fit one
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_ids = self.label_encoder.fit_transform(self.labels)
            else:
                self.label_encoder = label_encoder
                self.label_ids = self.label_encoder.transform(self.labels)
        else:
            self.labels = None
            self.label_ids = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.image_paths[idx]
        else:
            label = self.label_ids[idx]
            return image, label
