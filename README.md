# gen

A small library packaging the two generative models from this project — a
variational autoencoder (`genimg.VAE`) and a denoising diffusion probabilistic
model (`genimg.DDPM`) — behind one consistent, configuration-driven interface.

The model code is ported from the tutorial repositories (`Pytorch-DDPM-tutorial`,
`Pytorch-VAE-tutorial`): `SmallUNet` / `UNet` from `model.py`, the VAE
`Encoder` / `Decoder` / `VAE` from `vae_model.py`, and the trainer/sampler
logic from `ddpm.py` / `vae.py`.

## Install

```bash
pip install -e .            # from the project root
```

## Usage

```python
import genimg

# --- VAE (MNIST) ---
vae = genimg.VAE(latent_dim=64, epochs=50)
vae.train()
imgs = vae.generate(16)                 # (16, 784) in [0, 1]
vae.show_reconstruction(n=8)

# --- a different built-in dataset, by name ---
ddpm = genimg.DDPM(dataset_name="chestmnist")   # needs: pip install medmnist
ddpm.train()

# --- your own data ---
vae = genimg.VAE(x_dim=64 * 64, latent_dim=128)
vae.set_dataset(my_dataset)             # items: (image_tensor in [0,1], label)
vae.train()

# --- DDPM (MNIST, default) ---
ddpm = genimg.DDPM(timesteps=500)
ddpm.train()
imgs = ddpm.sample(16)                   # full ancestral sampling
fast = ddpm.ddim_sample(16, ddim_steps=50, seed=0)   # faster, reproducible

# --- DDPM on RGB with the attention U-Net (e.g. 128px faces) ---
ddpm = genimg.DDPM(
    arch="attention", channels=3, image_size=128,
    base_channels=64, channel_mults=(1, 2, 4, 8),
    attn_resolutions=(16, 8), num_heads=4, time_emb_dim=256,
    timesteps=1000, beta_start=1e-4, beta_end=0.02, lr=2e-4,
)
ddpm.set_dataset(my_rgb_dataset)         # items: (image_tensor in [-1,1], label)
ddpm.train(epochs=100)
ddpm.show_nearest_train(n=4, metric="nuclear")   # memorization check
fid = ddpm.compute_fid(n_samples=10000)  # n_samples must exceed `feature` (2048)
```

Every model shares the same control surface: `get` / `set` / `set_config` for
validated configuration, `set_dataset` to swap in your own data, lazy `build`,
`save` / `load`, and the visualisation helpers `show_samples` / `plot_samples`
/ `show_loss`.

## Models & configuration

### `genimg.DDPM`

Trainer + sampler for a denoising diffusion model. Two architectures via `arch`:

* `"small"` — the 2-level `SmallUNet` (default; good for 28px MNIST-style data).
* `"attention"` — the deeper `UNet` with self-attention (for larger / RGB data).

| config key | default | notes |
|---|---|---|
| `image_size` / `channels` | 28 / 1 | square images; 3 for RGB |
| `arch` | `"small"` | `"small"` or `"attention"` |
| `base_channels` / `time_emb_dim` | 32 / 128 | capacity (both archs); `base_channels` must be a multiple of 8 (GroupNorm) and `time_emb_dim` even |
| `channel_mults` | `(1,2,4,8)` | per-level width (attention only); at most `floor(log2(image_size))` levels |
| `attn_resolutions` | `(16,8)` | resolutions that get attention (attention only) |
| `num_heads` | 4 | attention heads (attention only); must divide `base_channels` |
| `timesteps` | 500 | diffusion steps |
| `beta_start` / `beta_end` | 1e-4 / 0.06 | linear noise schedule |
| `batch_size` / `num_workers` / `epochs` / `lr` | 128 / 1 / 50 / 1e-3 | training |
| `dataset_name` | `"mnist"` | built-in dataset, or `"custom"` via `set_dataset` |
| `dataset_path` / `save_dir` / `device` | — | IO |

Methods: `train` (with gradient clipping), `sample` / `ddim_sample` (both clip
the predicted `x0` each step to avoid over-exposed samples), `generate`,
`compute_fid` (grayscale is repeated 1→3 channels for Inception; RGB passes
through), `set_dataset`, `hp_search` / `tune_configs` (FID-ranked search), and
`show_nearest_train` (nuclear- or L2-distance nearest-training-image
memorization check).

Both searches take `save_best` (one checkpoint, overwritten whenever a trial
improves) and `save_trials` (a directory keeping every trial, named after the
values it used, with each result carrying its `path`). One of them is worth
using: a search leaves the object holding the *last* trial's weights, not the
best ones, so the files are the only way back to a particular trial.

### `genimg.VAE`

Trainer + generator for a fully connected VAE (BCE + KL, `-ELBO` objective).

| config key | default |
|---|---|
| `x_dim` / `hidden_dim` / `latent_dim` | 784 / 400 / 200 |
| `batch_size` / `num_workers` / `epochs` / `lr` | 100 / 1 / 30 / 1e-3 |

Methods: `train`, `generate`, `reconstruct`, `show_reconstruction`,
`set_dataset`.

A dataset passed to `set_dataset` must yield pixels in `[0, 1]` (the decoder
emits Bernoulli probabilities and the loss is a BCE against them), must match
`x_dim` once flattened, and needs `x_dim` to be a perfect square for the
visualisation helpers to reshape it.

## Datasets

`dataset_name` selects a built-in dataset; each model applies the pixel range it
needs, so the same name works for either one (`[0, 1]` for the VAE, `[-1, 1]`
for the DDPM).

| source | names |
|---|---|
| torchvision | `mnist`, `fashionmnist`, `kmnist` |
| [MedMNIST](https://medmnist.com) (`pip install medmnist`) | `chestmnist`, `pneumoniamnist`, `octmnist`, `breastmnist`, `tissuemnist`, `organamnist`, `organcmnist`, `organsmnist`, `pathmnist`, `dermamnist`, `retinamnist`, `bloodmnist` |
| your own | `custom` — set automatically by `set_dataset(...)` |

All of them are 28×28. The last four MedMNIST sets are RGB, so pass
`channels=3`; a mismatch against the configured `channels` / `image_size` is
reported when the loader is built. MedMNIST labels are arrays (ChestMNIST is
multi-label with 14 classes) and are normalised away, since nothing here uses
them.

```python
ddpm = genimg.DDPM(dataset_name="pathmnist", channels=3)
ddpm = genimg.DDPM(dataset_name="chestmnist", dataset_path="/content/drive/MyDrive/data")
```

### A folder of your own images

Large corpora — FFHQ, CelebA-HQ — are not in the table because they cannot be
fetched programmatically: **you download and unpack them yourself**, then point
`FolderImages` at the directory. It globs recursively for `.png`, `.jpg`,
`.jpeg`, `.webp` and `.bmp`, resizes the short side, centre-crops to a square,
and normalises. torchvision's `ImageFolder` is not a substitute — it expects
one subdirectory per class, which an unlabelled corpus does not have.

```python
from genimg import DDPM, FolderImages

ddpm = DDPM(arch="attention", channels=3, image_size=128,
            base_channels=64, time_emb_dim=256,
            attn_resolutions=(16, 8), timesteps=1000, beta_end=0.02, lr=2e-4)

ddpm.set_dataset(FolderImages(
    "~/data/ffhq/thumbnails128x128",   # whatever you unpacked it to
    image_size=128,
    max_images=5000,                   # a subset, for a first run
    seed=0,                            # reproducible subset
))
ddpm.train()
```

`pixel_range` defaults to `(-1, 1)` for the DDPM; pass `(0.0, 1.0)` for the
VAE. `channels=1` reads the images as greyscale. A directory with no images
raises rather than yielding an empty dataset, and `set_dataset` reports any
disagreement with the model's `channels` / `image_size`.

At 128px use `arch="attention"` — `SmallUNet` pools only twice, nowhere near
enough receptive field for a whole face.

## Package layout

```
genimg/
├── pyproject.toml          # pip-installable metadata
├── README.md
└── genimg/
    ├── __init__.py         # public API: genimg.VAE, genimg.DDPM, genimg.BaseModel
    ├── base.py             # BaseModel: config/schema/staleness/build/save/load + plots
    ├── vae.py              # VAE(BaseModel)
    ├── ddpm.py             # DDPM(BaseModel) — adds the noise-schedule component
    └── nn/                 # raw nn.Module definitions
        ├── __init__.py
        ├── vae_modules.py  # Encoder, Decoder, VAE (aliased VAENet)
        └── unet.py         # SmallUNet, UNet, AttentionBlock, blocks
```

## How the shared skeleton works

`BaseModel` owns everything the two trainers have in common. A concrete model
declares its `_SCHEMA` (reusing `BaseModel._BASE_SCHEMA`) and `_DEFAULTS`, then
implements `_build_model`, `_build_data`, `train`, and `generate`. Components
beyond model/optimizer/data — such as the DDPM noise schedule — are added by
declaring a new rebuild tag, registering a builder in `_builders()`, and
persisting any cached tensors through the `_extra_state` / `_load_extra_state`
hooks. Adding a third model is one new subclass.

The visualisation helpers (`show_samples`, `plot_samples`, `show_loss`,
`_display_image`) live once in `BaseModel`. Models tailor them through two small
hooks instead of reimplementing the plotting: `_grid_images` (the VAE overrides
it to reshape its flat output into images; the DDPM inherits it) and
`_loss_label` (each returns its own y-axis label and title). `show_reconstruction`
stays on the VAE, since the DDPM has no reconstruction step.

## Notes

* `set()` accepts a list where a tuple is expected, so `channel_mults=[1,2,4]`
  is coerced to `(1,2,4)`.
* Single-key constraints are enforced by `set()`; combinations that only make
  sense together (`num_heads` vs `base_channels`, `channel_mults` depth vs
  `image_size`) are checked by `build()` before anything is constructed.
* Both samplers take `seed`, which seeds a private `torch.Generator` rather
  than calling `torch.manual_seed` — sampling never disturbs the RNG of the
  code that called it. `generate(n, use_ddim=False)` still rejects the options
  that only DDIM has (`ddim_steps`, `eta`) instead of forwarding them to
  `sample()`; these also reach `generate` from `show_samples` / `plot_samples`.
* `set_dataset` takes one look at the first item and warns if its pixel range
  or shape does not match what the model expects. Feeding `[0, 1]` images to
  the DDPM otherwise trains perfectly happily and just produces washed-out
  samples. It also flips `dataset_name` to `"custom"`, so a saved config still
  describes where the data came from.
* `arch="attention"` warns when `attn_resolutions` matches none of the
  resolutions the network actually visits — at the default 28px it walks
  28/14/7/3, so the default `(16, 8)` would leave attention on at the
  bottleneck only.
* `compute_fid` needs the `fid` extra: `pip install -e '.[fid]'`.
* `compute_fid` estimates a `feature`x`feature` covariance per image set, so
  `n_samples` must exceed `feature` for the score to mean anything. The
  defaults (`n_samples=1000`, `feature=2048`) do **not** — they are sized for a
  quick smoke test and warn when used as-is. The same applies to `hp_search` /
  `tune_configs` via `fid_n_samples` / `fid_feature`, which rank trials by FID.
* `set_dataset` lives on `BaseModel`, so both models take one; the required
  pixel range differs (`[0, 1]` for the VAE, `[-1, 1]` for the DDPM). Passing
  `None` restores the built-in MNIST.
* `show_nearest_train` and `compute_fid` search the currently registered
  dataset — call `set_dataset(...)` first so the check runs against the data the
  model was actually trained on.
* EMA (exponential moving average of weights) is not implemented.
