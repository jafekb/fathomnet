"""
Use BioCLIP embeddings to try to split it out.

Visualize in 2D or 3D first.

Author: Ben Jafek
2025/05/13
"""
import os
import time

from matplotlib import pyplot as plt
import open_clip
import pandas as pd
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
import torch
# import umap  # TODO(bjafek)
from sklearn.manifold import TSNE


from benj_prac.fathomnet.fathomnet_dataset import FathomNetDataset

FULL_JSON = '/home/bjafek/Nuro/benj_prac/fathomnet/data/dataset_train.json'
TRAIN_ANNOTATIONS = "/home/bjafek/Nuro/benj_prac/fathomnet/data/train/annotations.csv"
TEST_ANNOTATIONS = "/home/bjafek/Nuro/benj_prac/fathomnet/data/test/annotations.csv"


EMBEDDINGS_NAME = "/home/bjafek/Nuro/benj_prac/fathomnet/data/bioclip/embeddings.pt"
TEST_EMBEDDINGS_NAME = "/home/bjafek/Nuro/benj_prac/fathomnet/data/bioclip/test_embeddings.pt"
EMBEDDINGS_DIR = "/home/bjafek/Nuro/benj_prac/fathomnet/data/bioclip/final_embeddings"
IMAGES_NAME = "/home/bjafek/Nuro/benj_prac/fathomnet/data/bioclip/images.pt"

OUT_DIR = "/home/bjafek/Nuro/benj_prac/fathomnet/output/eleventh"

# N_IMAGES = len(dataset)
N_IMAGES = None


def get_dataset():
    global N_IMAGES

    # Load labels
    train_annotations_df = pd.read_csv(TRAIN_ANNOTATIONS)
    label_encoder = LabelEncoder().fit(train_annotations_df["label"].dropna())

    # Load and transform the training dataset
    full_dataset = FathomNetDataset(
        csv_path=TRAIN_ANNOTATIONS,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
        ]),
        label_encoder=label_encoder,
        n_classes_subset=None,
    )

    if N_IMAGES is None:
        N_IMAGES = len(full_dataset)

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

def get_embeddings_dir():
    if os.path.isdir(EMBEDDINGS_DIR):
        print ("All embeddings already calculated!")
        return

    start = time.time()
    print ("Preprocessing...", end=" ")
    with torch.no_grad():
        for idx in trange(N_IMAGES):
            preprocessed_row = model.encode_image(images[idx].unsqueeze(0))
            out_name = os.path.join(EMBEDDINGS_DIR, f"emb_{idx:05d}.pt")
            torch.save(preprocessed_row, out_name)
    print (f"done! Took {time.time() - start:.2f}s")

def get_embeddings_mat(dataset):
    if os.path.isfile(EMBEDDINGS_NAME):
        print ("Loading existing embeddings!")
        return torch.load(EMBEDDINGS_NAME)

    preprocessed = torch.zeros((N_IMAGES, 512))

    filenames = sorted(os.listdir(EMBEDDINGS_DIR))
    n_fns = len(filenames)
    for idx in trange(n_fns):
        row = torch.load(os.path.join(EMBEDDINGS_DIR, filenames[idx]))
        preprocessed[idx] = row

    out = {
        "embeddings": preprocessed,
        "labels": dataset.labels[:N_IMAGES],
    }
    torch.save(out, EMBEDDINGS_NAME)
    return preprocessed


def create_embeddings(dataset):
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip")
    # tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")

    if os.path.isfile(EMBEDDINGS_NAME):
        print ("Loading existing embeddings")
        return torch.load(EMBEDDINGS_NAME), model, preprocess_val

    print ("Creating new embeddings...")

    images = get_images(dataset, preprocess_val)
    get_embeddings_dir()
    preprocessed = get_embeddings_mat(dataset)

    return preprocessed, model, preprocess_val


def visualize_2d(embeddings):
    """
    Reduces the dimensionality of CLIP embeddings to 2D using T-SNE
    and plots the result.
    """
    data = embeddings["embeddings"]
    labels = embeddings["labels"]

    # Reduce dimensionality to 2D using T-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(data)  # Fit and transform

    available_colors = "rgbcmyk"
    color_conv = {}
    for idx, lab in enumerate(set(labels)):
        color_conv[lab] = available_colors[idx]
    colors = [color_conv[lab] for lab in labels]

    # Create a scatter plot of the reduced embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors)
    plt.title("T-SNE Visualization of CLIP Embeddings")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.show()

def reduce_dimension(d1, d2):
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=5,
        metric="correlation",
    )
    d1_shape = len(d1)
    combo = torch.concat([d1, d2])
    transformed = tsne.fit_transform(combo)
    out1 = transformed[:d1_shape]
    out2 = transformed[d1_shape:]

    return out1, out2


def get_predictions(data, model, preprocess):
    """
    Get the predictions from the BioCLIP
    """
    train_annotations_df = pd.read_csv(TRAIN_ANNOTATIONS)
    label_encoder = LabelEncoder().fit(train_annotations_df["label"].dropna())

    # Load and transform the test dataset
    test_dataset = FathomNetDataset(
        csv_path=TEST_ANNOTATIONS,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
        ]),
        label_encoder=label_encoder,
        is_test=True,
    )

    def get_test_embeddings(test_dataset):
        if os.path.isfile(TEST_EMBEDDINGS_NAME):
            return torch.load(TEST_EMBEDDINGS_NAME)

        n_test_images = len(test_dataset)

        test_images = torch.zeros((n_test_images, 3, 224, 224))
        for idx in range(n_test_images):
            img, _ = test_dataset[idx]
            test_images[idx] = preprocess(img)
        print (test_images.shape)

        test_embeddings = torch.zeros((n_test_images, 512))
        with torch.no_grad():
            for idx in trange(n_test_images):
                preprocessed_row = model.encode_image(test_images[idx].unsqueeze(0))
                test_embeddings[idx] = preprocessed_row
        torch.save(test_embeddings, TEST_EMBEDDINGS_NAME)
        return test_embeddings

    test_embeddings = get_test_embeddings(test_dataset)
    print (test_embeddings.shape)

    k_neighbors = 3
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(data["labels"])
    x_train = data["embeddings"]

    x_train, test_embeddings = reduce_dimension(x_train, test_embeddings)
    knn_model.fit(x_train, y_train)

    y_pred = knn_model.predict(test_embeddings)
    decoded_predictions = label_encoder.inverse_transform(y_pred).tolist()

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


def main():
    dataset = get_dataset()
    train_embeddings, model, preprocess = create_embeddings(dataset)
    # visualize_2d(embeddings)
    get_predictions(train_embeddings, model, preprocess)


if __name__ == "__main__":
    main()
