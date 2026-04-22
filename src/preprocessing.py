from torchvision import transforms

# Estatísticas do ImageNet mantêm-se aqui pois são o padrão para modelos pré-treinados
IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(img_size: int = 224):
    """
    Cria as transformações de treino baseadas no img_size do config.
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), # Resize ligeiramente maior para o RandomCrop ter margem
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

def get_test_transforms(img_size: int = 224):
    """
    Cria as transformações de teste/validação.
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])