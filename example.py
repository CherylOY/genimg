"""Runnable examples for the ``genimg`` library.

Run from the project root (after ``pip install -e .`` or with the repo on
PYTHONPATH):

    python example.py vae            # train a VAE on MNIST, sample + reconstruct
    python example.py ddpm           # train a small DDPM on MNIST, sample + DDIM
    python example.py ddpm --attention   # the deeper attention U-Net instead
    python example.py all --epochs 5

Everything is intentionally short (a few epochs) so it finishes quickly on CPU;
bump --epochs for real results. Outputs (PNG grids) are written to --outdir.
"""

from __future__ import annotations

import argparse
import os

import torch

import genimg


def _device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_vae(epochs: int, device: str, outdir: str) -> None:
    print("\n=== VAE on MNIST ===")
    vae = genimg.VAE(latent_dim=64, epochs=epochs, device=device, save_dir=outdir)
    vae.train()                                    # prints per-epoch -ELBO

    # Sample new digits from the prior p(z) = N(0, I).
    vae.show_samples(n=64, nrow=8, save_path=os.path.join(outdir, "vae_samples.png"),
                     show=False)
    # Compare originals vs their reconstructions.
    vae.show_reconstruction(n=8, save_path=os.path.join(outdir, "vae_recon.png"),
                            show=False)
    vae.show_loss(save_path=os.path.join(outdir, "vae_loss.png"), show=False)
    vae.save(os.path.join(outdir, "vae.pt"))
    print("VAE done — see", outdir)


def run_ddpm(epochs: int, device: str, outdir: str, attention: bool) -> None:
    print(f"\n=== DDPM on MNIST (arch={'attention' if attention else 'small'}) ===")
    ddpm = genimg.DDPM(
        arch="attention" if attention else "small",
        timesteps=200,                             # small for a quick demo
        epochs=epochs,
        device=device,
        save_dir=outdir,
    )
    ddpm.train()                                   # prints per-epoch MSE loss

    # Full ancestral sampling and the faster (reproducible) DDIM sampler.
    ddpm.plot_samples(n=16, nrow=4, title="DDPM samples",
                      save_path=os.path.join(outdir, "ddpm_samples.png"), show=False)
    ddpm.plot_samples(n=16, nrow=4, use_ddim=True, ddim_steps=50, seed=0,
                      title="DDIM (50 steps)",
                      save_path=os.path.join(outdir, "ddpm_ddim.png"), show=False)

    # Memorization check: each generated digit next to its nearest training image.
    ddpm.show_nearest_train(n=4, metric="nuclear", max_train=5000, ddim_steps=50,
                            seed=0, save_path=os.path.join(outdir, "ddpm_nearest.png"),
                            show=False)

    ddpm.show_loss(save_path=os.path.join(outdir, "ddpm_loss.png"), show=False)
    ddpm.save(os.path.join(outdir, "ddpm.pt"))
    print("DDPM done — see", outdir)


def main() -> None:
    p = argparse.ArgumentParser(description="gen library examples")
    p.add_argument("model", choices=["vae", "ddpm", "all"],
                   help="which example to run")
    p.add_argument("--epochs", type=int, default=3, help="training epochs (default 3)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--attention", action="store_true",
                   help="use the attention U-Net for the DDPM")
    p.add_argument("--outdir", default="example_outputs")
    args = p.parse_args()

    device = _device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"device: {device}  |  epochs: {args.epochs}  |  outdir: {args.outdir}")

    if args.model in ("vae", "all"):
        run_vae(args.epochs, device, args.outdir)
    if args.model in ("ddpm", "all"):
        run_ddpm(args.epochs, device, args.outdir, args.attention)


if __name__ == "__main__":
    main()
