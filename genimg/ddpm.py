"""Denoising diffusion probabilistic model for the ``genimg`` library.

The DDPM adds one component the base class does not know about — a precomputed
noise schedule — so it declares an extra ``REBUILD_SCHEDULE`` tag, registers a
builder for it, and persists the cached buffers via the ``_extra_state`` hooks.
Everything else (config, optimizer, save/load, visualisation) is inherited.

Implementations are ported from the tutorial ``ddpm.py`` / ``model.py`` and
adapted to the BaseModel API (``self._train_loader``, the ``REBUILD_*`` tags,
``build(skip=...)``). Includes the tutorial fixes: per-step x0 clipping in the
sampler, gradient clipping during training, and the conditional 1->3 channel
repeat in FID so RGB models score correctly.
"""

from __future__ import annotations

import itertools
import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .base import BaseModel, REBUILD_MODEL, REBUILD_OPTIM, REBUILD_DATA, SchemaEntry
from .nn.unet import SmallUNet, UNet

REBUILD_SCHEDULE = "schedule"


class DDPM(BaseModel):
    """Trainer + sampler for a denoising diffusion probabilistic model.

    >>> ddpm = genimg.DDPM(timesteps=500)
    >>> ddpm.train()
    >>> imgs = ddpm.sample(16)
    """

    # Schedule is built first so the model/sampler can rely on its buffers.
    _BUILD_ORDER = (REBUILD_SCHEDULE, REBUILD_MODEL, REBUILD_OPTIM, REBUILD_DATA)

    _SCHEMA: Dict[str, SchemaEntry] = {
        **BaseModel._BASE_SCHEMA,
        "image_size": (int, lambda v: v > 0, (REBUILD_MODEL,)),
        "channels":   (int, lambda v: v > 0, (REBUILD_MODEL,)),
        "arch":       (str, lambda v: v in ("small", "attention"), (REBUILD_MODEL,)),
        # Network capacity. base_channels / time_emb_dim apply to both archs;
        # channel_mults / attn_resolutions / num_heads only affect "attention".
        "base_channels":    (int, lambda v: v > 0 and v % 8 == 0, (REBUILD_MODEL,),
                             "must be a positive multiple of 8: ConvBlock and "
                             "AttentionBlock normalise with nn.GroupNorm(8, channels), "
                             "and every level is a multiple of base_channels"),
        "time_emb_dim":     (int, lambda v: v > 0 and v % 2 == 0, (REBUILD_MODEL,),
                             "must be a positive even number: the sinusoidal embedding "
                             "emits dim//2 sine and dim//2 cosine components"),
        "channel_mults":    (tuple, lambda v: len(v) >= 1 and all(isinstance(m, int) and m > 0 for m in v),
                             (REBUILD_MODEL,)),
        "attn_resolutions": (tuple, lambda v: all(isinstance(r, int) and r > 0 for r in v),
                             (REBUILD_MODEL,)),
        "num_heads":        (int, lambda v: v > 0, (REBUILD_MODEL,)),
        "timesteps":  (int, lambda v: v >= 2, (REBUILD_SCHEDULE,)),
        "beta_start": (float, lambda v: 0 < v < 1, (REBUILD_SCHEDULE,)),
        "beta_end":   (float, lambda v: 0 < v < 1, (REBUILD_SCHEDULE,)),
        # device additionally invalidates the schedule (device-resident buffers)
        "device":     (str, lambda v: v in ("cuda", "cpu", "mps"),
                       (REBUILD_MODEL, REBUILD_OPTIM, REBUILD_SCHEDULE)),
    }

    # The sampler ends on (x + 1) / 2, so training data lives in [-1, 1].
    _DATA_RANGE = (-1.0, 1.0)

    _DEFAULTS: Dict[str, Any] = {
        "dataset_path": "~/datasets",
        "batch_size": 128,
        "num_workers": 1,
        "epochs": 50,
        "lr": 1e-3,
        "save_dir": "ddpm_outputs",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_size": 28,
        "channels": 1,
        "arch": "small",
        "base_channels": 32,
        "time_emb_dim": 128,
        "channel_mults": (1, 2, 4, 8),
        "attn_resolutions": (16, 8),
        "num_heads": 4,
        "timesteps": 500,
        "beta_start": 1e-4,
        "beta_end": 0.06,
    }

    # Register the extra builder so base.build() picks it up.
    def _builders(self):
        b = super()._builders()
        b[REBUILD_SCHEDULE] = self._build_schedule
        return b

    # ------------------------------------------------------------------ #
    # Cross-key validation (a _SCHEMA validator only sees one value at a time)
    # ------------------------------------------------------------------ #
    def _validate_config(self) -> None:
        # The forward process has to add noise as t grows, so betas must rise.
        beta_start, beta_end = self._config["beta_start"], self._config["beta_end"]
        if beta_start >= beta_end:
            raise ValueError(
                f"beta_start={beta_start} must be smaller than beta_end={beta_end}: "
                f"the schedule interpolates from beta_start up to beta_end, and a "
                f"flat or descending one leaves the data under-noised at t=T")

        size = self._config["image_size"]

        if self._config["arch"] == "small":
            # SmallUNet pools twice, so 4px is the smallest input that survives
            # the round trip down to the bottleneck and back.
            if size < 4:
                raise ValueError(
                    f"image_size={size} is too small for arch='small': its two "
                    f"pooling stages need at least 4 pixels")
            return

        base  = self._config["base_channels"]
        mults = tuple(self._config["channel_mults"])
        heads = self._config["num_heads"]

        # One pooling stage per level; the bottleneck must keep at least one
        # pixel. Repeated halving is floor(size / 2**levels), so the bound is
        # levels <= floor(log2(size)).
        max_levels = max(size.bit_length() - 1, 0)
        if len(mults) > max_levels:
            raise ValueError(
                f"channel_mults has {len(mults)} levels, which downsamples "
                f"image_size={size} past a single pixel; use at most "
                f"{max_levels} level(s) at this image_size "
                f"(or raise image_size to {2 ** len(mults)})")

        # AttentionBlock splits a level's channels across the heads. Every level
        # is base_channels * mult wide, and the decoder's last one is exactly
        # base_channels, so dividing base_channels covers all of them.
        if base % heads:
            raise ValueError(
                f"num_heads={heads} must divide base_channels={base}: the "
                f"attention blocks split each level's channels across the heads")

        # Attention is inserted only where a level's resolution is listed, so a
        # non-matching list quietly yields a plain conv U-Net plus the
        # bottleneck -- the kind of thing you discover after a day of training.
        attn_res = tuple(self._config["attn_resolutions"])
        if attn_res:
            visited = [size // (2 ** i) for i in range(len(mults))]
            if not set(attn_res) & set(visited):
                warnings.warn(
                    f"attn_resolutions={attn_res} matches none of the resolutions "
                    f"this network visits ({visited}), so self-attention is applied "
                    f"at the bottleneck only. Pick values from {visited}, or pass "
                    f"attn_resolutions=() to say you meant that.")

    # ------------------------------------------------------------------ #
    # Component construction (model-specific bits only)
    # ------------------------------------------------------------------ #
    def _build_schedule(self) -> None:
        T = self._config["timesteps"]
        device = self._device()
        betas = torch.linspace(
            self._config["beta_start"], self._config["beta_end"], T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self._schedule = {
            "betas":                     betas,
            "alphas":                    alphas,
            "alpha_bars":                alpha_bars,
            "sqrt_alpha_bars":           torch.sqrt(alpha_bars),
            "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
            "sqrt_recip_alphas":         torch.sqrt(1.0 / alphas),
            "posterior_variance":        betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars),
            # For reconstructing / clipping the predicted clean image x0:
            "sqrt_recip_alpha_bars":     torch.sqrt(1.0 / alpha_bars),
            "sqrt_recipm1_alpha_bars":   torch.sqrt(1.0 / alpha_bars - 1.0),
            # Posterior-mean coefficients: mean = coef1 * x0 + coef2 * x_t
            "posterior_mean_coef1":      betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars),
            "posterior_mean_coef2":      (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars),
        }
        self._stale.discard(REBUILD_SCHEDULE)

    def _build_model(self) -> None:
        if self._config["arch"] == "small":
            self._model = SmallUNet(
                in_channels   = self._config["channels"],
                base_channels = self._config["base_channels"],
                time_emb_dim  = self._config["time_emb_dim"],
            ).to(self._device())
        else:  # "attention"
            self._model = UNet(
                in_channels      = self._config["channels"],
                base_channels    = self._config["base_channels"],
                channel_mults    = tuple(self._config["channel_mults"]),
                time_emb_dim     = self._config["time_emb_dim"],
                attn_resolutions = tuple(self._config["attn_resolutions"]),
                image_size       = self._config["image_size"],
                num_heads        = self._config["num_heads"],
            ).to(self._device())
        self._stale.discard(REBUILD_MODEL)
        self._stale.add(REBUILD_OPTIM)

    def _build_data(self) -> None:
        # Use a caller-supplied dataset if one was registered via set_dataset();
        # otherwise fall back to MNIST (rescaled to [-1, 1]).
        dataset = self._custom_dataset
        if dataset is None:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),   # [0,1] -> [-1,1]
            ])
            dataset = MNIST(self._config["dataset_path"], train=True,
                            download=True, transform=tf)
        self._train_loader = DataLoader(
            dataset, batch_size=self._config["batch_size"], shuffle=True,
            num_workers=self._config["num_workers"],
            pin_memory=(self._config["device"] == "cuda"))
        self._stale.discard(REBUILD_DATA)

    # Persist the cached schedule buffers alongside the checkpoint.
    def _extra_state(self) -> Dict[str, Any]:
        return {"schedule": getattr(self, "_schedule", None)}

    def _load_extra_state(self, state: Dict[str, Any]) -> None:
        if state.get("schedule") is not None:
            self._schedule = state["schedule"]
            self._stale.discard(REBUILD_SCHEDULE)

    # ------------------------------------------------------------------ #
    # Diffusion math (private helpers)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
        """A private RNG for a seeded draw, or None to use the global stream.

        Seeding locally rather than calling ``torch.manual_seed`` keeps the
        caller's own random stream intact: a sampler has no business resetting
        the RNG of the training loop that called it.
        """
        if seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab  = self._extract(self._schedule["sqrt_alpha_bars"],           t, x_start.shape)
        sqrt_omb = self._extract(self._schedule["sqrt_one_minus_alpha_bars"], t, x_start.shape)
        return sqrt_ab * x_start + sqrt_omb * noise

    def _p_losses(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self._q_sample(x_start, t, noise)
        predicted = self._model(x_noisy, t)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def _p_sample(self, x: torch.Tensor, t: torch.Tensor,
                  clip_denoised: bool = True,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        predicted_noise = self._model(x, t)
        # Reconstruct x0, optionally clip to [-1, 1] to avoid over-exposure.
        sqrt_recip_ab   = self._extract(self._schedule["sqrt_recip_alpha_bars"],   t, x.shape)
        sqrt_recipm1_ab = self._extract(self._schedule["sqrt_recipm1_alpha_bars"], t, x.shape)
        x0 = sqrt_recip_ab * x - sqrt_recipm1_ab * predicted_noise
        if clip_denoised:
            x0 = torch.clamp(x0, -1.0, 1.0)

        coef1 = self._extract(self._schedule["posterior_mean_coef1"], t, x.shape)
        coef2 = self._extract(self._schedule["posterior_mean_coef2"], t, x.shape)
        model_mean = coef1 * x0 + coef2 * x

        post_var_t   = self._extract(self._schedule["posterior_variance"], t, x.shape)
        # torch.randn rather than randn_like: only the former takes a generator
        # across the whole torch>=2.0 range this package supports.
        noise        = torch.randn(x.shape, dtype=x.dtype, device=x.device,
                                   generator=generator)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.sqrt(torch.clamp(post_var_t, min=1e-20)) * noise

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(self, epochs: Optional[int] = None, verbose: bool = True) -> list:
        """Train the noise-prediction network. Returns per-epoch avg losses.

        Gradient clipping (max-norm 1.0) is applied every step to prevent the
        occasional bad batch from blowing the loss up.
        """
        self.build()
        epochs = epochs if epochs is not None else self._config["epochs"]
        device = self._device()
        os.makedirs(self._config["save_dir"], exist_ok=True)
        T = self._config["timesteps"]

        epoch_losses: list = []
        for epoch in range(epochs):
            self._model.train()
            total_loss, n_batches = 0.0, 0
            for step, (x, _) in enumerate(self._train_loader):
                x = x.to(device)
                t = torch.randint(0, T, (x.shape[0],), device=device, dtype=torch.long)
                loss = self._p_losses(x, t)

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                if verbose and step % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}  Step {step:4d}  Loss: {loss.item():.4f}")

            avg = total_loss / max(n_batches, 1)
            epoch_losses.append(avg)
            self.loss_history.append(avg)
            if verbose:
                print(f"Epoch {epoch+1} finished — avg loss = {avg:.4f}")
        return epoch_losses

    @torch.no_grad()
    def sample(self, n: int = 16, clip_denoised: bool = True,
               seed: Optional[int] = None) -> torch.Tensor:
        """Full DDPM ancestral sampling over all timesteps. Returns [0,1] images.

        ``seed`` makes the draw reproducible without disturbing the caller's
        global RNG -- see :meth:`_generator`.
        """
        self.build(skip=(REBUILD_DATA,))
        self._model.eval()
        device = self._device()
        C = self._config["channels"]
        H = W = self._config["image_size"]
        T = self._config["timesteps"]

        g = self._generator(seed, device)
        x = torch.randn((n, C, H, W), device=device, generator=g)
        for i in reversed(range(T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x = self._p_sample(x, t, clip_denoised=clip_denoised, generator=g)
        x = (x + 1) / 2
        return torch.clamp(x, 0, 1)

    def _ddim_timesteps(self, ddim_steps: int) -> list:
        T = self._config["timesteps"]
        # linspace with a single point returns its start, so the general branch
        # would hand back [0] and "denoise" a pure-noise image from t=0. One
        # step means one jump from the noisiest step straight to the image.
        if ddim_steps == 1:
            return [T - 1]
        idx = torch.linspace(0, T - 1, ddim_steps).round().long()
        idx = torch.unique(idx)
        return idx.flip(0).tolist()

    @torch.no_grad()
    def ddim_sample(self, n: int = 16, ddim_steps: int = 50,
                    eta: float = 0.0, clip_denoised: bool = True,
                    seed: Optional[int] = None) -> torch.Tensor:
        """Faster deterministic (eta=0) DDIM sampling. Returns [0,1] images."""
        T = self._config["timesteps"]
        if ddim_steps < 1:
            raise ValueError(f"ddim_steps must be >= 1, got {ddim_steps!r}")
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta!r}")

        self.build(skip=(REBUILD_DATA,))
        self._model.eval()
        device = self._device()
        C = self._config["channels"]
        H = W = self._config["image_size"]

        g = self._generator(seed, device)
        alpha_bars = self._schedule["alpha_bars"]
        times = self._ddim_timesteps(min(ddim_steps, T))
        x = torch.randn((n, C, H, W), device=device, generator=g)

        for i, tau in enumerate(times):
            t    = torch.full((n,), int(tau), device=device, dtype=torch.long)
            ab_t = self._extract(alpha_bars, t, x.shape)
            predicted_noise = self._model(x, t)

            x0 = (x - torch.sqrt(1.0 - ab_t) * predicted_noise) / torch.sqrt(ab_t)
            if clip_denoised:
                x0 = torch.clamp(x0, -1.0, 1.0)

            tau_prev = times[i + 1] if i + 1 < len(times) else -1
            if tau_prev < 0:
                ab_prev = torch.ones_like(ab_t)
            else:
                t_prev  = torch.full((n,), int(tau_prev), device=device, dtype=torch.long)
                ab_prev = self._extract(alpha_bars, t_prev, x.shape)

            sigma = eta * torch.sqrt(
                (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev))
            dir_xt = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0)) * predicted_noise
            noise = (torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=g)
                     if eta > 0 else torch.zeros_like(x))
            x = torch.sqrt(ab_prev) * x0 + dir_xt + sigma * noise

        x = (x + 1) / 2
        return torch.clamp(x, 0, 1)

    # Sampler options that only ddim_sample() understands. sample() takes none
    # of them, so without this check they reach it as an unexpected keyword.
    _DDIM_ONLY_KWARGS = ("ddim_steps", "eta")

    # Unify the entry point: generate() delegates to the chosen sampler.
    # ``kwargs`` also arrives here from the inherited show_samples() /
    # plot_samples(), which forward theirs straight through.
    def generate(self, n: int = 16, use_ddim: bool = False, **kwargs: Any) -> torch.Tensor:
        if use_ddim:
            return self.ddim_sample(n, **kwargs)
        misrouted = sorted(k for k in kwargs if k in self._DDIM_ONLY_KWARGS)
        if misrouted:
            raise TypeError(
                f"{', '.join(repr(k) for k in misrouted)} only applies to the DDIM "
                f"sampler; pass use_ddim=True as well, or call ddim_sample(...) "
                f"directly")
        return self.sample(n, **kwargs)

    def compute_fid(self, n_samples: int = 1000, sample_batch: int = 64,
                    feature: int = 2048, **kwargs: Any) -> float:
        """Frechet Inception Distance between training and generated images.

        Grayscale (1-channel) images are repeated to 3 channels for Inception;
        RGB images pass through unchanged. Lower is better.

        ``n_samples`` should comfortably exceed ``feature``: FID estimates a
        ``feature x feature`` covariance per image set, so with fewer samples
        than features the estimate is singular and the score mostly reports
        that bias. The defaults do not satisfy this -- they are sized for a
        quick sanity check, not for a number worth publishing -- so a warning
        is emitted whenever the ratio is off.
        """
        # Warn before sampling rather than after: generating n_samples images
        # can take minutes, and the ratio is knowable up front.
        # <=, not <: a covariance estimated from n samples has rank n-1 at
        # best, so n == feature is still singular.
        if n_samples <= feature:
            warnings.warn(
                f"FID with feature={feature} estimates a {feature}x{feature} "
                f"covariance from only {n_samples} samples, so the score is "
                f"dominated by that bias rather than by sample quality and is "
                f"not comparable across runs. Raise n_samples above {feature} "
                f"(10k or more is the usual choice), or lower feature "
                f"(torchmetrics offers 64, 192, 768 and 2048).",
                stacklevel=2,
            )

        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError as e:
            raise ImportError(
                "FID needs torchmetrics and the torch-fidelity Inception feature "
                "extractor. Install: pip install 'genimg[fid]'") from e

        self.build()
        self._model.eval()
        device = self._device()
        fid = FrechetInceptionDistance(feature=feature, normalize=True).to(device)

        n_real = 0
        with torch.no_grad():
            for x, _ in self._train_loader:
                x = (x + 1) / 2
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)          # 1ch -> 3ch; RGB passes through
                x = x.to(device)
                fid.update(x, real=True)
                n_real += x.size(0)
                if n_real >= n_samples:
                    break

        n_gen = 0
        while n_gen < n_samples:
            k = min(sample_batch, n_samples - n_gen)
            imgs = self.sample(n=k)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            imgs = imgs.to(device)
            fid.update(imgs, real=False)
            n_gen += k

        return float(fid.compute())

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning (both routines rank trials by FID)
    # ------------------------------------------------------------------ #
    def hp_search(self, search_space: Dict[str, list],
                  epochs_per_trial: Optional[int] = None,
                  fid_n_samples: int = 1000, fid_feature: int = 2048,
                  max_trials: Optional[int] = None, shuffle: bool = False,
                  seed: Optional[int] = None, save_best: Optional[str] = None,
                  verbose: bool = True) -> list:
        """Grid-search the cartesian product of `search_space`, ranking by FID.

        For each combination: reset to the pre-search config snapshot, apply
        the trial's values, force a fresh network + optimizer, train from
        scratch for `epochs_per_trial`, and score with compute_fid(). Returns
        a list of {'params', 'fid'} sorted by FID ascending (best first).
        """
        for k, vs in search_space.items():
            if k not in self._SCHEMA:
                raise KeyError(f"unknown hp key: {k!r}. valid keys: {sorted(self._SCHEMA)}")
            if not isinstance(vs, (list, tuple)) or len(vs) == 0:
                raise ValueError(f"search_space[{k!r}] must be a non-empty list")
            # Vet every candidate now: a value rejected seven trials in would
            # otherwise throw away everything trained up to that point.
            for v in vs:
                self._coerce_and_check(k, v)
        if max_trials is not None and max_trials <= 0:
            raise ValueError(f"max_trials must be > 0 if given, got {max_trials!r}")

        snapshot = self.get_config()
        keys   = list(search_space.keys())
        combos = list(itertools.product(*[search_space[k] for k in keys]))
        total  = len(combos)
        if shuffle:
            random.Random(seed).shuffle(combos)
        if max_trials is not None:
            combos = combos[:max_trials]
        if verbose:
            print(f"[hp_search] running {len(combos)} of {total} combinations")

        results: list = []
        best_fid = float("inf")
        for i, values in enumerate(combos):
            params = dict(zip(keys, values))
            if verbose:
                print(f"\n[hp_search] trial {i+1}/{len(combos)}: {params}")
            self.set_config(**snapshot)
            for k, v in params.items():
                self.set(k, v)
            # Force a fresh model + optimizer even if no model-affecting key changed.
            self._stale.add(REBUILD_MODEL)
            self._stale.add(REBUILD_OPTIM)
            self.loss_history = []

            self.train(epochs=epochs_per_trial, verbose=verbose)
            fid = self.compute_fid(n_samples=fid_n_samples, feature=fid_feature)
            if verbose:
                print(f"[hp_search] trial {i+1} FID = {fid:.4f}")
            results.append({"params": params, "fid": fid})
            if fid < best_fid:
                best_fid = fid
                if save_best is not None:
                    self.save(save_best)
                    if verbose:
                        print(f"[hp_search] new best — saved to {save_best}")

        self.set_config(**snapshot)
        results.sort(key=lambda r: r["fid"])
        if verbose:
            print("\n[hp_search] summary (sorted by FID ascending):")
            for r in results:
                print(f"  FID={r['fid']:.4f}   params={r['params']}")
        return results

    def tune_configs(self, configs: list,
                     epochs_per_trial: Optional[int] = None,
                     fid_n_samples: int = 1000, fid_feature: int = 2048,
                     save_best: Optional[str] = None,
                     verbose: bool = True) -> list:
        """Train and FID-score an explicit list of configurations.

        Like hp_search() but runs only the user-supplied `configs` (a list of
        dicts) rather than expanding a full grid. Same from-scratch guarantee
        per trial; returns {'params', 'fid'} sorted by FID ascending.
        """
        if not isinstance(configs, list) or not configs:
            raise ValueError("configs must be a non-empty list of dicts")
        for i, cfg in enumerate(configs):
            if not isinstance(cfg, dict):
                raise TypeError(f"configs[{i}] must be a dict, got {type(cfg).__name__}")
            for k in cfg:
                if k not in self._SCHEMA:
                    raise KeyError(f"unknown config key {k!r} in configs[{i}]")
                self._coerce_and_check(k, cfg[k])

        snapshot = self.get_config()
        if verbose:
            print(f"[tune_configs] running {len(configs)} user-specified trials")

        results: list = []
        best_fid = float("inf")
        for i, params in enumerate(configs):
            if verbose:
                print(f"\n[tune_configs] trial {i+1}/{len(configs)}: {params}")
            self.set_config(**snapshot)
            for k, v in params.items():
                self.set(k, v)
            self._stale.add(REBUILD_MODEL)
            self._stale.add(REBUILD_OPTIM)
            self.loss_history = []

            self.train(epochs=epochs_per_trial, verbose=verbose)
            fid = self.compute_fid(n_samples=fid_n_samples, feature=fid_feature)
            if verbose:
                print(f"[tune_configs] trial {i+1} FID = {fid:.4f}")
            results.append({"params": dict(params), "fid": fid})
            if fid < best_fid:
                best_fid = fid
                if save_best is not None:
                    self.save(save_best)
                    if verbose:
                        print(f"[tune_configs] new best — saved to {save_best}")

        self.set_config(**snapshot)
        results.sort(key=lambda r: r["fid"])
        if verbose:
            print("\n[tune_configs] summary (sorted by FID ascending):")
            for r in results:
                print(f"  FID={r['fid']:.4f}   params={r['params']}")
        return results

    # ------------------------------------------------------------------ #
    # Memorization check: generated vs nearest training image
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def show_nearest_train(self, n: int = 4, metric: str = "nuclear",
                           chunk: int = 256, max_train: Optional[int] = None,
                           use_ddim: bool = True, ddim_steps: int = 250,
                           eta: float = 0.0, seed: Optional[int] = 0,
                           save_path: Optional[str] = None, show: bool = True) -> list:
        """Check for memorization: for each generated image, find and display the
        most similar image in the training set.

        Interpretation:
            * near-identical pair (same pixels/background) -> memorization
            * clearly different face / digit               -> generalizing

        Args:
            metric:    "nuclear" -> ||gen - train||_* (sum of singular values of
                       the per-channel difference, summed over channels);
                       "l2" -> pixel L2 (much faster).
            chunk:     Training images compared per GPU batch.
            max_train: Cap on how many training images to search (None = all).
            use_ddim / ddim_steps / eta / seed: passed to the sampler; seed fixes
                       the generated images so the check is reproducible.
        Returns:
            List of (train_index, distance) — the nearest match per generated image.

        Note: searches whatever dataset is currently registered / built, so this
        is only meaningful against the set the model was actually trained on.
        """
        if metric not in ("nuclear", "l2"):
            raise ValueError(f"metric must be 'nuclear' or 'l2', got {metric!r}")

        self.build()
        dataset = self._train_loader.dataset
        device  = self._device()
        N = len(dataset) if max_train is None else min(max_train, len(dataset))

        if use_ddim:
            gen = self.generate(n, use_ddim=True, ddim_steps=ddim_steps,
                                eta=eta, seed=seed).cpu()
        else:
            gen = self.generate(n, seed=seed).cpu()

        train = torch.stack([(dataset[i][0] + 1) / 2 for i in range(N)])  # -> [0,1]

        def _nearest(g: torch.Tensor) -> Tuple[int, float]:
            g = g.to(device)
            dists = []
            for s in range(0, N, chunk):
                tb   = train[s:s + chunk].to(device)
                diff = g.unsqueeze(0) - tb
                b, C, H, W = diff.shape
                if metric == "nuclear":
                    sv = torch.linalg.svdvals(diff.reshape(b * C, H, W))
                    d  = sv.sum(dim=1).reshape(b, C).sum(dim=1)
                else:
                    d  = diff.reshape(b, -1).pow(2).sum(dim=1)
                dists.append(d.cpu())
            dists = torch.cat(dists)
            return int(dists.argmin()), float(dists.min())

        import os
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n, 2, figsize=(4.5, 2.2 * n))
        axes = axes.reshape(n, 2)
        results: list = []
        for k in range(n):
            idx, dist = _nearest(gen[k])
            results.append((idx, dist))
            g_img = gen[k]
            t_img = (dataset[idx][0] + 1) / 2
            for ax, img, title in ((axes[k, 0], g_img, "generated"),
                                   (axes[k, 1], t_img, f"nearest ({metric}={dist:.1f})")):
                arr = img.permute(1, 2, 0).cpu().numpy()
                ax.imshow(arr.squeeze(-1), cmap="gray") if arr.shape[-1] == 1 else ax.imshow(arr)
                ax.set_title(title); ax.axis("off")
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path)
            print(f"[show_nearest_train] saved: {save_path}")
        plt.show() if show else plt.close()
        return results

    # ------------------------------------------------------------------ #
    # Visualisation hook: only the loss-curve labels differ from the base.
    # ------------------------------------------------------------------ #
    def _loss_label(self) -> tuple:
        return ("Average MSE Loss", "DDPM Training Loss Curve")
