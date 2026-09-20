"""Variational autoencoder model for the ``gen`` library.

Only VAE-specific code lives here; all of the config / build / save plumbing
is inherited from :class:`gen.base.BaseModel`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .base import BaseModel, REBUILD_MODEL, REBUILD_OPTIM, REBUILD_DATA, SchemaEntry
from .nn.vae_modules import Encoder, Decoder, VAENet


class VAE(BaseModel):
    """Trainer + generator for a variational autoencoder.

    >>> vae = gen.VAE(latent_dim=64, epochs=50)
    >>> vae.train()
    >>> imgs = vae.generate(16)
    """

    # Reuse the common rows, add only the VAE-specific architecture keys.
    _SCHEMA: Dict[str, SchemaEntry] = {
        **BaseModel._BASE_SCHEMA,
        "x_dim":      (int, lambda v: v > 0, (REBUILD_MODEL,)),
        "hidden_dim": (int, lambda v: v > 0, (REBUILD_MODEL,)),
        "latent_dim": (int, lambda v: v > 0, (REBUILD_MODEL,)),
    }

    _DEFAULTS: Dict[str, Any] = {
        "dataset_path": "~/datasets",
        "batch_size": 100,
        "num_workers": 1,
        "epochs": 30,
        "lr": 1e-3,
        "save_dir": "vae_outputs",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "x_dim": 784,
        "hidden_dim": 400,
        "latent_dim": 200,
    }

    # ------------------------------------------------------------------ #
    # Component construction (model-specific bits only)
    # ------------------------------------------------------------------ #
    def _build_model(self) -> None:
        enc = Encoder(self._config["x_dim"], self._config["hidden_dim"],
                      self._config["latent_dim"])
        dec = Decoder(self._config["latent_dim"], self._config["hidden_dim"],
                      self._config["x_dim"])
        self._model = VAENet(enc, dec).to(self._device())
        self._stale.discard(REBUILD_MODEL)
        self._stale.add(REBUILD_OPTIM)        # fresh model -> fresh optimizer

    def _build_data(self) -> None:
        # Use a caller-supplied dataset if one was registered via set_dataset();
        # otherwise fall back to MNIST. Either way the pixels must be in [0, 1]:
        # the decoder emits Bernoulli probabilities and the loss is a BCE against
        # them. A custom dataset also has to match x_dim once flattened, and the
        # visualisation helpers assume x_dim is a perfect square.
        train = self._custom_dataset
        if train is None:
            tf = transforms.Compose([transforms.ToTensor()])
            train = MNIST(self._config["dataset_path"], train=True,
                          download=True, transform=tf)
        self._train_loader = DataLoader(
            train, batch_size=self._config["batch_size"], shuffle=True,
            num_workers=self._config["num_workers"],
            pin_memory=(self._config["device"] == "cuda"))
        self._stale.discard(REBUILD_DATA)

    # ------------------------------------------------------------------ #
    # VAE-specific public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def _loss_function(x_hat, x, mu, logvar):
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld

    def train(self, epochs: Optional[int] = None, verbose: bool = True) -> list:
        """Train the VAE (BCE + KLD). Returns per-epoch average -ELBO / sample."""
        self.build()
        epochs = epochs if epochs is not None else self._config["epochs"]
        device = self._device()
        os.makedirs(self._config["save_dir"], exist_ok=True)

        epoch_losses: list = []
        for epoch in range(epochs):
            self._model.train()
            total_loss, n_seen = 0.0, 0
            for x, _ in self._train_loader:
                x = x.view(x.size(0), -1).to(device)      # flatten to (B, x_dim)
                x_hat, mu, logvar = self._model(x)
                loss = self._loss_function(x_hat, x, mu, logvar)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()
                n_seen += x.size(0)

            avg = total_loss / max(n_seen, 1)
            epoch_losses.append(avg)
            self.loss_history.append(avg)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} finished — avg -ELBO/sample = {avg:.4f}")
        return epoch_losses

    @torch.no_grad()
    def generate(self, n: int = 16, **kwargs: Any) -> torch.Tensor:
        """Sample latents from p(z)=N(0, I) and decode to flat images in [0, 1]."""
        self.build(skip=(REBUILD_DATA,))
        self._model.eval()
        z = torch.randn(n, self._config["latent_dim"], device=self._device())
        return self._model.Decoder(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode-reparameterize-decode `x` (flat or (B,1,H,W)); return (B, x_dim)."""
        self.build(skip=(REBUILD_DATA,))
        self._model.eval()
        if x.dim() > 2:
            x = x.view(x.size(0), self._config["x_dim"])
        x = x.to(self._device())
        x_hat, _, _ = self._model(x)
        return x_hat

    # ------------------------------------------------------------------ #
    # Visualisation: only the VAE-specific pieces; the grid/loss plotting
    # itself is inherited from BaseModel.
    # ------------------------------------------------------------------ #
    def _grid_images(self, n: int, **kwargs: Any) -> torch.Tensor:
        """VAE.generate returns flat (n, x_dim) vectors; reshape to images."""
        imgs = self.generate(n, **kwargs).detach().cpu()
        side = int(self._config["x_dim"] ** 0.5)      # 784 -> 28
        return imgs.view(n, 1, side, side)

    def _loss_label(self) -> tuple:
        return ("Average -ELBO", "VAE Training Loss Curve")

    def show_reconstruction(self, n: int = 8,
                            save_path: str = "reconstruction.png",
                            show: bool = True) -> str:
        """Save a 2-row grid: originals (top) vs their reconstructions (bottom)."""
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        self.build()
        x, _ = next(iter(self._train_loader))
        x = x[:n]
        x_hat = self.reconstruct(x).cpu()

        side = int(self._config["x_dim"] ** 0.5)          # 784 -> 28
        orig = x.view(n, 1, side, side).cpu()
        recon = x_hat.view(n, 1, side, side)
        grid = make_grid(torch.cat([orig, recon], dim=0), nrow=n)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.figure(figsize=(n, 2.4))
        plt.imshow(grid.permute(1, 2, 0).numpy().squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("top: original    bottom: reconstruction")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[show_reconstruction] saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()
        return save_path
