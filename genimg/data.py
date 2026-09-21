"""Dataset helpers for data the library cannot fetch for you.

The named datasets in ``BaseModel``'s registry all download themselves. Large
image corpora do not: FFHQ, CelebA-HQ and friends have to be obtained from
their own sources and unpacked by hand, so what a library can usefully offer is
a way to point at the folder once they are on disk.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .base import _Rescale

# Extensions worth globbing for. Matched case-insensitively.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class FolderImages(Dataset):
    """Every image under a directory, resized to a square and normalised.

    Built for corpora you download yourself -- FFHQ's ``thumbnails128x128``
    being the motivating case -- where there are no class subdirectories and no
    labels, just files. (torchvision's ``ImageFolder`` wants one directory per
    class, which such a corpus does not have.)

    Each item is ``(image_tensor, 0)``: the dummy label keeps the
    ``(image, label)`` unpacking in ``train`` and ``compute_fid`` working.

    Args:
        root:        Directory searched recursively for images.
        image_size:  Output side length. Images are resized on their short side
                     and centre-cropped, so nothing is squashed.
        channels:    3 for RGB, 1 for greyscale. Must match the model's
                     ``channels`` config.
        pixel_range: Range to normalise into. The default suits the DDPM; pass
                     ``(0.0, 1.0)`` for the VAE. ``set_dataset`` warns if this
                     disagrees with the model.
        max_images:  Keep only this many, for quick experiments on a subset
                     (say 5000 of FFHQ's 70000).
        seed:        How that subset is chosen. ``None`` takes the first
                     ``max_images`` by sorted path; an int samples that many at
                     random, reproducibly.

    Raises:
        FileNotFoundError: if ``root`` holds no images.
    """

    def __init__(self, root: str, image_size: int = 128, channels: int = 3,
                 pixel_range: Tuple[float, float] = (-1.0, 1.0),
                 max_images: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {channels!r}")

        root = os.path.expanduser(root)
        paths = sorted(
            p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if p.lower().endswith(IMAGE_SUFFIXES)
        )
        if not paths:
            raise FileNotFoundError(
                f"no images under {root!r} (looked recursively for "
                f"{', '.join(IMAGE_SUFFIXES)}). This dataset reads images you "
                f"have already downloaded -- it fetches nothing itself.")

        if max_images is not None and max_images < len(paths):
            paths = (random.Random(seed).sample(paths, max_images)
                     if seed is not None else paths[:max_images])

        self.paths = paths
        self.mode = "RGB" if channels == 3 else "L"
        lo, hi = pixel_range
        steps = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),              # -> [0, 1]
        ]
        if (lo, hi) != (0.0, 1.0):
            steps.append(_Rescale(lo, hi))
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert(self.mode)), 0
