"""
Module for Pytorch dataset representations
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx: int) -> dict:
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments:
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        # You could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used

        # Device-agnostic tensor creation (works on both CPU and CUDA)
        sample["image"] = torch.from_numpy(self.data[slc[0]]["image"][slc[1]]).float().unsqueeze(0)
        sample["seg"] = torch.from_numpy(self.data[slc[0]]["seg"][slc[1]]).float().unsqueeze(0)

        return sample

    def __len__(self) -> int:
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
