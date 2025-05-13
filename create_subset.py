import json
import random

def create_subset_json(input_json_path, output_json_path, subset_size):
    """
    Create a subset of a COCO-style JSON dataset by randomly sampling a specified number of images.

    Args:
        input_json_path (str): Path to the input JSON file in COCO format.
        output_json_path (str): Path where the output subset JSON file will be saved.
        subset_size (int): Number of images to include in the subset.

    The function preserves the original structure of the COCO JSON file,
    including 'info', 'licenses', 'categories', and filters annotations to only those
    that correspond to the sampled images.
    """
    # Load the full dataset
    with open(input_json_path, "r") as f:
        full_data = json.load(f)

    # Sample images
    sampled_images = random.sample(full_data["images"], subset_size)
    sampled_image_ids = {img["id"] for img in sampled_images}

    # Filter annotations
    sampled_annotations = [
        ann for ann in full_data["annotations"] if ann["image_id"] in sampled_image_ids
    ]

    # Build new dataset with original structure
    subset_data = {
        "info": full_data.get("info", {}),
        "licenses": full_data.get("licenses", []),
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": full_data["categories"]
    }

    # Save it
    with open(output_json_path, "w") as f:
        json.dump(subset_data, f, indent=4)
