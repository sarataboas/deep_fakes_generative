# CHECK THIS: 
# https://medium.com/@kumudtraveldiaries/step-by-step-preprocessing-guide-for-images-in-both-cnn-and-dense-layer-pipelines-1994c3ad3e87
# https://www.kaggle.com/code/vesuvius13/how-to-preprocess-and-train-a-cnn-step-by-step


from torchvision import transforms

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5), # data augmentation
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02
        ), # data augmentation
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