# =============================================================================
# vae_model.py — Variational AutoEncoder (VAE) Architecture
# =============================================================================
# Reference: Stanley H. Chan, "Tutorial on Diffusion Models for Imaging and
#            Vision", arXiv:2403.18103, 2025. Section 1 (VAE), pp. 2-17.
#
# This file defines three nn.Module classes:
#   1. Encoder  — maps input x to the parameters (µ, log σ²) of qϕ(z|x)
#   2. Decoder  — maps latent vector z to reconstructed image fθ(z)
#   3. VAE      — combines Encoder + Decoder and implements the
#                 reparameterization trick to make the latent sampling
#                 differentiable during training
#
# Key distributions (Chan 2025, Section 1.1):
#   p(z)       : Prior over latent space.  Chosen to be N(0, I).
#   qϕ(z|x)   : Encoder distribution (proxy for the true p(z|x)).
#                Parameterised as N(z | µϕ(x), σ²ϕ(x)·I).
#   pθ(x|z)   : Decoder distribution (proxy for the true p(x|z)).
#                Parameterised as a Bernoulli (for binary/normalised images).
# =============================================================================

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Step 1: Encoder  —  qϕ(z|x) = N(z | µϕ(x), σ²ϕ(x)·I)
# -----------------------------------------------------------------------------
# The encoder is a neural network that takes an image x ∈ R^{input_dim} and
# outputs the mean µ and log-variance log σ² of the approximate posterior
# qϕ(z|x).  (Chan 2025, Eq. 1.2 and Figure 1.5)
#
# Why log σ² instead of σ²?
#   - The network can output any real number for log σ², whereas σ² must be
#     positive.  Taking exp() later guarantees positivity.
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Args:
            input_dim  : Dimension of flattened input image (e.g. 784 for MNIST).
            hidden_dim : Width of the two hidden layers.
            latent_dim : Dimension of the latent space z.
        """
        super().__init__()

        # Two shared hidden layers that extract features from x
        self.FC_input  = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)

        # Two separate output heads for µ and log σ²
        # These are the parameters of qϕ(z|x) = N(z | µ, σ²I)
        self.FC_mean   = nn.Linear(hidden_dim, latent_dim)   # outputs µϕ(x)
        self.FC_var    = nn.Linear(hidden_dim, latent_dim)   # outputs log σ²ϕ(x)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Flattened input image, shape (B, input_dim).
        Returns:
            mean    : µϕ(x),       shape (B, latent_dim).
            log_var : log σ²ϕ(x),  shape (B, latent_dim).
        """
        # Shared feature extraction
        h = self.LeakyReLU(self.FC_input(x))
        h = self.LeakyReLU(self.FC_input2(h))

        # Produce the two parameters of the Gaussian qϕ(z|x)
        mean    = self.FC_mean(h)   # µϕ(x)
        log_var = self.FC_var(h)    # log σ²ϕ(x)  — can be any real number

        return mean, log_var


# -----------------------------------------------------------------------------
# Step 2: Decoder  —  pθ(x|z)
# -----------------------------------------------------------------------------
# The decoder is a neural network that maps a latent vector z ∈ R^{latent_dim}
# to a reconstructed image fθ(z) ∈ R^{output_dim}.  (Chan 2025, Eq. 1.28-1.29
# and Figure 1.6)
#
# For MNIST pixels normalised to [0, 1] we use a Bernoulli decoder:
#   pθ(x|z) = Bernoulli(fθ(z))
# which corresponds to a binary cross-entropy reconstruction loss.
# The final Sigmoid squashes the output to (0, 1) pixel probabilities.
# -----------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            latent_dim : Dimension of the latent vector z.
            hidden_dim : Width of the two hidden layers.
            output_dim : Dimension of the reconstructed image (e.g. 784).
        """
        super().__init__()

        self.FC_hidden  = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output  = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : Latent vector, shape (B, latent_dim).
        Returns:
            x_hat : Reconstructed image probabilities fθ(z), shape (B, output_dim).
        """
        h = self.LeakyReLU(self.FC_hidden(z))
        h = self.LeakyReLU(self.FC_hidden2(h))

        # Sigmoid maps network output to (0, 1) — pixel-wise Bernoulli probability
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


# -----------------------------------------------------------------------------
# Step 3: VAE — Encoder + Reparameterization Trick + Decoder
# -----------------------------------------------------------------------------
# The VAE wraps the encoder and decoder and provides the reparameterization
# trick which makes the stochastic sampling node differentiable so gradients
# can flow back through z to the encoder parameters ϕ.
#
# Reparameterization Trick (Chan 2025, Eq. 1.22 and Example 1.7):
#   Instead of sampling z ~ N(µ, σ²I) directly (non-differentiable w.r.t. µ, σ),
#   we write:
#       z = µ  +  σ ⊙ ε,    ε ~ N(0, I)
#   where ε is sampled independently of ϕ.  Now ∂z/∂µ = 1 and ∂z/∂σ = ε,
#   so gradients flow normally through µ and σ to the encoder network.
# -----------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """
        Args:
            encoder : An Encoder instance.
            decoder : A Decoder instance.
        """
        super().__init__()
        self.Encoder = encoder
        self.Decoder = decoder

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample z from qϕ(z|x) = N(µ, σ²I) using the reparameterization trick.

        Given log σ² from the encoder, we recover σ = exp(0.5 * log σ²).
        Then we sample:
            z = µ  +  σ ⊙ ε,    ε ~ N(0, I)

        This keeps the sampling operation outside the computational graph,
        so gradients w.r.t. µ and log σ² can be computed normally.
        (Chan 2025, Eq. 1.22)

        Args:
            mean    : µϕ(x),       shape (B, latent_dim).
            log_var : log σ²ϕ(x),  shape (B, latent_dim).
        Returns:
            z : Sampled latent vector, shape (B, latent_dim).
        """
        # σ = exp(0.5 · log σ²)  — standard deviation, always positive
        std = torch.exp(0.5 * log_var)
        # ε ~ N(0, I),  same shape as std
        eps = torch.randn_like(std)
        # z = µ + σ ⊙ ε
        return mean + std * eps

    def forward(self, x: torch.Tensor):
        """
        Full VAE forward pass: encode → reparameterize → decode.

        Args:
            x : Flattened input image, shape (B, input_dim).
        Returns:
            x_hat   : Reconstructed image, shape (B, input_dim).
            mean    : µϕ(x),       shape (B, latent_dim).
            log_var : log σ²ϕ(x),  shape (B, latent_dim).
        """
        # Encode: x → (µ, log σ²)  —  parameters of qϕ(z|x)
        mean, log_var = self.Encoder(x)

        # Sample: z ~ qϕ(z|x) via the reparameterization trick
        z = self.reparameterize(mean, log_var)

        # Decode: z → x̂  —  the reconstructed image fθ(z)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


# The gen library imports the wrapper under the name VAENet (gen.vae.VAE is
# the trainer class, so the network cannot also be called VAE there).
VAENet = VAE
