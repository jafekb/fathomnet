"""
Evaluate the model that was trained with train_model.py

Author: Ben Jafek
2025/04/28
"""
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader, Subset


from benj_prac.fathomnet import preprocessing
from benj_prac.fathomnet.classifier import FathomNetClassifier
from benj_prac.fathomnet.fathomnet_dataset import FathomNetDataset


TEST_ANNOTATIONS = "/home/bjafek/Nuro/benj_prac/fathomnet/data/test/annotations.csv"
TRAIN_ANNOTATIONS = "/home/bjafek/Nuro/benj_prac/fathomnet/data/train/annotations.csv"
BATCH_SIZE = 16

BEST_MODEL_PATH = "/home/bjafek/Nuro/benj_prac/fathomnet/output/fifth/epoch=13-val_loss=1.26-train_loss=0.85.ckpt"

OUT_DIR = "/home/bjafek/Nuro/benj_prac/fathomnet/output/fifth"

def main():
    train_annotations_df = pd.read_csv(TRAIN_ANNOTATIONS)
    label_encoder = LabelEncoder().fit(train_annotations_df["label"].dropna())

    # Load and transform the test dataset
    test_dataset = FathomNetDataset(
        csv_path=TEST_ANNOTATIONS,
        transform=preprocessing.TEST_TRANSFORMS,
        label_encoder=label_encoder,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=35,
        shuffle=False,
    )

    # Set device to GPU if available, otherwise CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Load the best model and move it to the correct device
    best_model = FathomNetClassifier.load_from_checkpoint(
        BEST_MODEL_PATH,
        num_classes=len(label_encoder.classes_),
    )
    best_model = best_model.to(device)

    # Generate predictions for the test dataset
    predictions = []
    ids = []

    # Set the model to evaluation mode
    best_model.eval()

    # No gradients needed for inference
    with torch.no_grad():
        for batch in test_loader:
            images, image_ids = batch
            images = images.to(device)

            logits = best_model(images)
            preds = torch.argmax(logits, dim=1).tolist()
            predictions.extend(preds)
            ids.extend(image_ids)

    # Decode numeric predictions to original string labels
    decoded_predictions = label_encoder.inverse_transform(predictions)

    # Save predictions to the test annotations CSV file
    submission_df = pd.read_csv(TEST_ANNOTATIONS)
    submission_df["annotation_id"] = range(1, len(submission_df) + 1)
    submission_df["concept_name"] = decoded_predictions
    submission_df = submission_df.drop(["path", "label"], axis=1)
    submission_df.to_csv(
        os.path.join(OUT_DIR, "submission.csv"),
        index=False,
    )
    print(submission_df.head())

if __name__ == "__main__":
    main()
