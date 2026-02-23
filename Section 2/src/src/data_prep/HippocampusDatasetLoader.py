"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

from utils.utils import med_reshape


def LoadHippocampusData(root_dir: str, y_shape: int, z_shape: int) -> np.ndarray:
    """Load hippocampus dataset from disk into memory, reshaping to common size.

    Arguments:
        root_dir {str} -- path to the root directory containing 'images' and 'labels' subdirectories
        y_shape {int} -- target size for the coronal (Y) dimension
        z_shape {int} -- target size for the sagittal (Z) dimension

    Returns:
        Numpy array of dictionaries with 'image', 'seg', and 'filename' keys
    """

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header
        # since we will not use it
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        # Normalize all images so that values are in [0..1] range
        max_pixel_value = image.max()
        if max_pixel_value == 0:
            raise ValueError(f"Image {f} has max pixel value of 0; cannot normalize.")
        image = image / max_pixel_value

        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # Labels are cast to int because cross-entropy loss expects integer class indices
        out.append({"image": image, "seg": label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)
