from torchvision import transforms

# Define data transformations for preprocessing
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNetV2-M expects 224x224 input
    transforms.RandomHorizontalFlip(p=0.5),  # Marine life often has left/right symmetry
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # Simulates light variation underwater
    transforms.RandomRotation(degrees=10),  # Slight rotation
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Mild zoom/pan
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225]),
])

# For validation/test (no augmentations, just resizing and normalization)
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
