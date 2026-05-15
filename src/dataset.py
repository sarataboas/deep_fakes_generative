"""
PyTorch Dataset for the deepfake detection project.
Loads images on demand from a metadata DataFrame produced by build_metadata.py.
"""

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class DeepFakeDataset(Dataset):
    """
    Map-style Dataset that reads images lazily from disk.

    Each sample is a dict so that training loops can access metadata
    (source_type, filepath) alongside the image tensor without needing
    a separate lookup structure.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        # reset_index ensures iloc indexing is contiguous after any prior filtering
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
            "image_id": row.get("image_id", ""),
            "source_type": row["source_type"],
            "filepath": row["filepath"],
        }

        return sample