from torchvision import transforms

# ImageNet statistics — standard for any model pre-trained on ImageNet (ResNet, EfficientNet, etc.)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """Training transforms for the classifier with light augmentation."""
    return transforms.Compose([
        # Resize slightly larger than target so RandomCrop has room to shift
        transforms.Resize(int(img_size * 1.14)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms(img_size: int = 224):
    """Deterministic transforms for validation and test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_vae_transforms(img_size: int = 64):
    """
    Training transforms for VAE/GAN models.

    Normalises to [-1, 1] (mean=0.5, std=0.5) instead of ImageNet statistics
    because VAE decoders and GAN generators output Tanh activations in [-1, 1].
    Horizontal flip is the only augmentation — spatial structure must be preserved.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def get_vae_val_transforms(img_size: int = 64):
    """Deterministic transforms for VAE/GAN validation and test (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
