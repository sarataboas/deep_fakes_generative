"""
Images preprocessing by applying the following steps: 
    - Image resize (256)
    - Image Cropping with 3 methods: 
        1. Original Image
        2. RandomCrop
        3. Face Crop
    - Data Augmentation
        1. Horizontal Flip (mirror)  (0.5 probability)
        2. Color Modifications
    - Conversion to tensors --> small parameters to avoid big modifications
    - Normalization

The test set will not be subjected to the augmentation steps
"""

from torchvision import transforms

# ------------ CONFIGURATION ----------
IMG_SIZE = 224

# ImageNet dataset statistics for the color channels - CNN trained convention
IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])