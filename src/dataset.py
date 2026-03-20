"""
Creates the Dataset Class
"""


from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class DeepFakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake image classification using a metadata dataframe.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df: Metadata dataframe.
            transform: torchvision transform pipeline to apply to each image.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Loads one sample from the dataset.

        Args:
            idx: Index of the sample to load.

        Returns:
            dict containing:
                - image: preprocessed image tensor
                - label: integer label
                - image_id: unique image identifier
                - source_type: source category
                - filepath: original image path
        """
        row = self.df.iloc[idx]

        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": label,
            "image_id": row["image_id"],
            "source_type": row["source_type"],
            "filepath": row["filepath"],
        }

        return sample