"""
Use BioCLIP embeddings to try to split it out.

Visualize in 2D or 3D first.

Author: Ben Jafek
2025/05/13
"""
import os
import time

import open_clip
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
import torch

from benj_prac.fathomnet.fathomnet_dataset import FathomNetDataset

FULL_JSON = '/home/bjafek/Nuro/benj_prac/fathomnet/data/dataset_train.json'
ANNOTATIONS = "/home/bjafek/Nuro/benj_prac/fathomnet/data/train/annotations.csv"

EMBEDDINGS_NAME = "/home/bjafek/Nuro/benj_prac/fathomnet/data/bioclip/embeddings.pt"
IMAGES_NAME = "/home/bjafek/Nuro/benj_prac/fathomnet/data/bioclip/images.pt"

# N_IMAGES = len(dataset)
N_IMAGES = 100


def get_dataset():

    # Load labels
    train_annotations_df = pd.read_csv(ANNOTATIONS)
    label_encoder = LabelEncoder().fit(train_annotations_df["label"].dropna())
    print(train_annotations_df.head())
    num_classes = len(label_encoder.classes_)

    # Load and transform the training dataset
    full_dataset = FathomNetDataset(
        csv_path=ANNOTATIONS,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNetV2-M expects 224x224 input
        ]),
        label_encoder=label_encoder,
    )

    return full_dataset


def get_images(dataset, preprocess):
    if os.path.isfile(IMAGES_NAME):
        print("Loading existing images")
        return torch.load(IMAGES_NAME)

    start = time.time()
    print ("Loading images...", end=" ")
    images = torch.zeros((N_IMAGES, 3, 224, 224))
    for idx in trange(N_IMAGES):
        img, _ = dataset[idx]
        images[idx] = preprocess(img)
    torch.save(images, IMAGES_NAME)
    print (f"done! Took {time.time() - start:.2f}s")
    return images


def create_embeddings(dataset):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

    images = get_images(dataset, preprocess_val)

    preprocessed = torch.zeros((N_IMAGES, 512))

    start = time.time()
    print ("Preprocessing...", end=" ")
    with torch.no_grad():
        preprocessed = model.encode_image(images)
    print (f"done! Took {time.time() - start:.2f}s")

    out = {
        "embeddings": preprocessed,
        "labels": dataset.labels,
    }
    torch.save(out, EMBEDDINGS_NAME)


def main():
    dataset = get_dataset()
    embeddings = create_embeddings(dataset)


if __name__ == "__main__":
    main()
