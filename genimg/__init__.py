"""gen — a small library for generative image models on MNIST-style data.

Public API
----------
    import genimg

    vae = genimg.VAE(latent_dim=64, epochs=50)
    vae.train()
    imgs = vae.generate(16)

    ddpm = genimg.DDPM(timesteps=500)
    ddpm.train()
    imgs = ddpm.sample(16)

Both models expose the same configuration-driven interface inherited from
``genimg.BaseModel`` (``get`` / ``set`` / ``set_config`` / ``build`` / ``save``
/ ``load``), so a third model can be added by subclassing ``BaseModel`` and
implementing ``_build_model`` / ``_build_data`` / ``train`` / ``generate``.
"""

from .base import BaseModel, REBUILD_MODEL, REBUILD_OPTIM, REBUILD_DATA
from .data import FolderImages
from .vae import VAE
from .ddpm import DDPM

__all__ = [
    "BaseModel",
    "VAE",
    "DDPM",
    "FolderImages",
    "REBUILD_MODEL",
    "REBUILD_OPTIM",
    "REBUILD_DATA",
]

__version__ = "0.1.0"
