"""Shared skeleton for all generative models in the ``genimg`` library.

``BaseModel`` factors out everything the VAE and the DDPM trainers have in
common: a validated configuration dictionary, a declarative schema, lazy
("stale set") component rebuilding, optimizer construction, checkpoint
save/load, and the visualisation helpers. Concrete models (``genimg.VAE``,
``genimg.DDPM``) subclass it and supply only what is genuinely model-specific.

A subclass is expected to:

* define ``_SCHEMA`` and ``_DEFAULTS`` (may extend the base entries),
* implement ``_build_model`` (and any extra ``_build_*`` for components it
  adds, e.g. the DDPM noise schedule),
* implement ``_build_data`` so that it honours ``self._custom_dataset`` before
  falling back to its own built-in dataset,
* implement ``train`` and ``generate``,
* optionally override ``_validate_config`` to reject config *combinations*
  that the per-key validators cannot see,
* optionally override ``_extra_state`` / ``_load_extra_state`` to persist
  anything beyond the model and optimizer,
* optionally override ``_grid_images`` / ``_loss_label`` to customise the
  shared visualisation helpers.
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Rebuildable-component tags. Subclasses may add their own (e.g. the DDPM
# adds a "schedule" tag) but these three are understood by the base class.
REBUILD_MODEL = "model"
REBUILD_OPTIM = "optim"
REBUILD_DATA = "data"

# Built-in datasets, selected by name through config["dataset_name"]. The key
# is what callers type; the value is the class to pull out of the providing
# package. Everything here is MNIST-shaped (28x28) unless noted.
TORCHVISION_DATASETS = {
    "mnist":        "MNIST",
    "fashionmnist": "FashionMNIST",
    "kmnist":       "KMNIST",
}
MEDMNIST_DATASETS = {
    "chestmnist":     "ChestMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "octmnist":       "OCTMNIST",
    "breastmnist":    "BreastMNIST",
    "tissuemnist":    "TissueMNIST",
    "organamnist":    "OrganAMNIST",
    "organcmnist":    "OrganCMNIST",
    "organsmnist":    "OrganSMNIST",
    "pathmnist":      "PathMNIST",      # 3-channel -- set channels=3
    "dermamnist":     "DermaMNIST",     # 3-channel
    "retinamnist":    "RetinaMNIST",    # 3-channel
    "bloodmnist":     "BloodMNIST",     # 3-channel
}
# "custom" means "whatever set_dataset() registered".
SUPPORTED_DATASETS = (set(TORCHVISION_DATASETS)
                      | set(MEDMNIST_DATASETS)
                      | {"custom"})


class _Rescale:
    """Map [0, 1] to [lo, hi].

    A module-level class rather than a lambda so the transform can be pickled
    to DataLoader workers, which spawn-based platforms (macOS, Windows) need.
    """

    def __init__(self, lo: float, hi: float) -> None:
        self.lo, self.span = lo, hi - lo

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.span + self.lo


class _ImagesOnly(Dataset):
    """Wrap a dataset whose label is not a plain scalar.

    The MedMNIST sets return a numpy array per item -- length 14 for the
    multi-label ChestMNIST. Nothing here uses labels, so they are replaced with
    0 and every dataset unpacks the same way.
    """

    def __init__(self, ds: Any) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.ds[i][0], 0

# Type of a single schema row: (expected_type, validator, affected_tags), plus
# an optional 4th element -- a plain-English statement of what the validator
# requires, quoted back to the caller when a value is rejected. Rows that leave
# it out still work; the rejection is just less chatty.
SchemaEntry = Union[
    Tuple[type, Callable[[Any], bool], Tuple[str, ...]],
    Tuple[type, Callable[[Any], bool], Tuple[str, ...], str],
]


class BaseModel(ABC):
    """Abstract trainer/generator with config, lazy build, and IO plumbing."""

    # Subclasses override both of these. The base rows below are common to
    # every model and can be reused via ``{**BaseModel._BASE_SCHEMA, ...}``.
    _BASE_SCHEMA: Dict[str, SchemaEntry] = {
        "dataset_path": (str, lambda v: len(v) > 0, (REBUILD_DATA,)),
        "dataset_name": (str, lambda v: v.lower() in SUPPORTED_DATASETS, (REBUILD_DATA,),
                         f"must be one of {sorted(SUPPORTED_DATASETS)}"),
        "batch_size":   (int, lambda v: v > 0,      (REBUILD_DATA,)),
        "num_workers":  (int, lambda v: v >= 0,     (REBUILD_DATA,)),
        "epochs":       (int, lambda v: v > 0,      ()),
        "lr":           (float, lambda v: v > 0,    (REBUILD_OPTIM,)),
        "save_dir":     (str, lambda v: len(v) > 0, ()),
        "device":       (str, lambda v: v in ("cuda", "cpu", "mps"),
                         (REBUILD_MODEL, REBUILD_OPTIM)),
    }

    _SCHEMA: Dict[str, SchemaEntry] = {}
    _DEFAULTS: Dict[str, Any] = {}

    # Pixel range a caller-supplied dataset is expected to already be in.
    # Subclasses set it to whatever their loss and sampler assume.
    _DATA_RANGE: Tuple[float, float] = (0.0, 1.0)

    # The set of component tags this model knows how to (re)build, in build
    # order. Subclasses that add components prepend/append their own tags.
    _BUILD_ORDER: Tuple[str, ...] = (REBUILD_MODEL, REBUILD_OPTIM, REBUILD_DATA)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, **overrides: Any) -> None:
        # Everything starts stale; first use triggers construction. This MUST
        # be initialized before applying overrides, because set() updates it.
        self._stale = set(self._BUILD_ORDER)

        self._config: Dict[str, Any] = dict(self._DEFAULTS)
        for k, v in overrides.items():
            self.set(k, v)

        self._model: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        # Set by set_dataset(); None means "build the model's own dataset".
        self._custom_dataset: Any = None
        self.loss_history: list = []

    # ------------------------------------------------------------------ #
    # Validated configuration access (shared by every model)
    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Any:
        if key not in self._SCHEMA:
            raise KeyError(f"unknown config key: {key!r}. valid keys: {sorted(self._SCHEMA)}")
        return self._config[key]

    def _coerce_and_check(self, key: str, value: Any) -> Any:
        """Coerce and validate ``value`` for ``key``, returning what ``set``
        would store. Raises exactly what ``set`` raises but changes nothing, so
        callers that only want to vet a value ahead of time (the DDPM's
        hyperparameter search screening its grid) can use it on its own.
        """
        if key not in self._SCHEMA:
            raise KeyError(f"unknown config key: {key!r}. valid keys: {sorted(self._SCHEMA)}")
        entry = self._SCHEMA[key]
        expected_type, validator = entry[0], entry[1]
        requirement = entry[3] if len(entry) > 3 else None

        if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if expected_type is tuple and isinstance(value, list):
            value = tuple(value)      # accept a list where a tuple is expected
        # bool subclasses int, so isinstance(True, int) passes and a stray True
        # would quietly become batch_size=1.
        if expected_type is int and isinstance(value, bool):
            raise TypeError(f"config {key!r} expected int, got bool ({value!r})")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"config {key!r} expected {expected_type.__name__}, "
                f"got {type(value).__name__} ({value!r})"
            )
        if not validator(value):
            detail = f" -- {requirement}" if requirement else ""
            raise ValueError(f"config {key!r} has invalid value: {value!r}{detail}")
        return value

    def set(self, key: str, value: Any) -> None:
        value = self._coerce_and_check(key, value)
        if self._config.get(key) != value:
            self._config[key] = value
            self._stale.update(self._SCHEMA[key][2])

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def set_config(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.set(k, v)

    # ------------------------------------------------------------------ #
    # Lazy component construction
    # ------------------------------------------------------------------ #
    def _device(self) -> torch.device:
        return torch.device(self._config["device"])

    @abstractmethod
    def _build_model(self) -> None:
        """Instantiate ``self._model`` from the current config and move it to
        the device. Must clear REBUILD_MODEL and re-flag REBUILD_OPTIM."""

    def _build_optimizer(self) -> None:
        """Default Adam optimizer over the model parameters. Override if a
        model needs something else."""
        assert self._model is not None, "model must be built before the optimizer"
        self._optimizer = Adam(self._model.parameters(), lr=self._config["lr"])
        self._stale.discard(REBUILD_OPTIM)

    def _default_transform(self):
        """ToTensor, then rescale into ``_DATA_RANGE``.

        ToTensor already yields [0, 1], so a model that wants that range gets
        no extra step and a model that wants [-1, 1] gets one multiply-add.
        """
        lo, hi = self._DATA_RANGE
        steps = [transforms.ToTensor()]
        if (lo, hi) != (0.0, 1.0):
            steps.append(_Rescale(lo, hi))
        return transforms.Compose(steps)

    def _resolve_dataset(self) -> Any:
        """Return the dataset named by ``config["dataset_name"]``."""
        name = self._config["dataset_name"].lower()

        if name == "custom":
            if self._custom_dataset is None:
                raise RuntimeError(
                    "dataset_name='custom' but no dataset was registered. "
                    "Call set_dataset(my_dataset) first, or set dataset_name "
                    f"to one of {sorted(SUPPORTED_DATASETS - {'custom'})}.")
            return self._custom_dataset

        root = os.path.expanduser(self._config["dataset_path"])
        transform = self._default_transform()

        if name in TORCHVISION_DATASETS:
            from torchvision import datasets as tv_datasets
            cls = getattr(tv_datasets, TORCHVISION_DATASETS[name])
            return cls(root=root, train=True, download=True, transform=transform)

        try:
            import medmnist
        except ImportError as e:
            raise ImportError(
                f"dataset_name={name!r} needs the medmnist package. "
                f"Install: pip install medmnist") from e
        # medmnist will not create `root` itself -- it raises "Failed to setup
        # the default `root` directory" if the folder is not already there.
        os.makedirs(root, exist_ok=True)
        cls = getattr(medmnist, MEDMNIST_DATASETS[name])
        return _ImagesOnly(cls(split="train", transform=transform,
                               download=True, root=root))

    def _build_data(self) -> None:
        """Build the training loader from whichever dataset is selected."""
        dataset = self._resolve_dataset()
        self._check_dataset(dataset)
        self._train_loader = DataLoader(
            dataset,
            batch_size=self._config["batch_size"],
            shuffle=True,
            num_workers=self._config["num_workers"],
            pin_memory=(self._config["device"] == "cuda"),
        )
        self._stale.discard(REBUILD_DATA)

    def set_dataset(self, dataset: Any) -> None:
        """Train on a caller-supplied dataset instead of the built-in MNIST.

        Each item must be ``(image_tensor, label)``, with the image already in
        the pixel range the model trains on: ``[0, 1]`` for :class:`genimg.VAE`
        (its Bernoulli/BCE reconstruction term needs probabilities) and
        ``[-1, 1]`` for :class:`genimg.DDPM` (its sampler returns to that range).
        Passing ``None`` restores the built-in dataset.

        Marks the data loader stale, so the next ``build`` rebuilds it.
        """
        self._custom_dataset = dataset
        # Keep dataset_name honest about where the data now comes from, so
        # config snapshots (hp_search, save/load, repr) describe reality.
        self.set("dataset_name",
                 self._DEFAULTS["dataset_name"] if dataset is None else "custom")
        self._stale.add(REBUILD_DATA)
        if dataset is not None:
            self._check_dataset(dataset)

    def _check_dataset(self, dataset: Any) -> None:
        """Warn when the first item does not match what this model expects.

        Handing [0, 1] images to the DDPM (which wants [-1, 1]) trains happily
        and just produces washed-out samples, so nothing would ever raise --
        one cheap look at one item is the only warning anyone gets. Advisory
        only: a dataset that genuinely occupies an unusual range is fine.
        """
        try:
            x = dataset[0][0]
            seen_lo, seen_hi = float(x.min()), float(x.max())
            shape = tuple(x.shape)
        except Exception:
            return          # not indexable, or not tensors: nothing to check

        lo, hi = self._DATA_RANGE
        tol = 0.05 * (hi - lo)
        outside = seen_lo < lo - tol or seen_hi > hi + tol
        # A signed range whose data never goes negative is the classic mix-up.
        unsigned = lo < 0 <= seen_lo
        if outside or unsigned:
            fix = ("Rescale it with `x * 2 - 1`." if unsigned
                   else "Check the dataset's transform.")
            warnings.warn(
                f"{type(self).__name__} expects dataset images in "
                f"[{lo:g}, {hi:g}], but the first item spans "
                f"[{seen_lo:.3g}, {seen_hi:.3g}]. {fix}",
                stacklevel=3,
            )

        self._check_dataset_shape(shape)

    def _check_dataset_shape(self, shape: Tuple[int, ...]) -> None:
        """Compare one item's shape against the config. Overridden per model,
        since the VAE trains on flattened vectors and the DDPM on images."""

    # Maps a component tag to its builder. Subclasses extend this dict.
    def _builders(self) -> Dict[str, Callable[[], None]]:
        return {
            REBUILD_MODEL: self._build_model,
            REBUILD_OPTIM: self._build_optimizer,
            REBUILD_DATA:  self._build_data,
        }

    def _validate_config(self) -> None:
        """Check constraints that span more than one config key.

        A ``_SCHEMA`` validator only ever sees a single value, so anything
        relating two keys (e.g. "num_heads must divide base_channels") belongs
        here instead. ``build`` calls it before constructing anything, so a bad
        combination is reported as a config error rather than surfacing as a
        shape mismatch deep inside a module.
        """

    def build(self, skip: Tuple[str, ...] = ()) -> None:
        """Rebuild every stale component, in ``_BUILD_ORDER``.

        ``skip`` lets callers avoid building components they do not need
        (e.g. ``generate`` skips the data loader).
        """
        self._validate_config()
        builders = self._builders()
        for tag in self._BUILD_ORDER:
            if tag in skip:
                continue
            if tag in self._stale and tag in builders:
                builders[tag]()

    # ------------------------------------------------------------------ #
    # Required model-specific behaviour
    # ------------------------------------------------------------------ #
    @abstractmethod
    def train(self, epochs: Optional[int] = None, verbose: bool = True) -> list:
        """Train the model and return the per-epoch loss history."""

    @abstractmethod
    def generate(self, n: int, **kwargs: Any) -> torch.Tensor:
        """Draw ``n`` samples from the trained model."""

    # ------------------------------------------------------------------ #
    # Checkpointing (shared; subclasses add state via the two hooks below)
    # ------------------------------------------------------------------ #
    def _extra_state(self) -> Dict[str, Any]:
        """State to persist beyond model/optimizer/config. Override as needed."""
        return {}

    def _load_extra_state(self, state: Dict[str, Any]) -> None:
        """Restore whatever ``_extra_state`` saved. Override as needed."""

    def save(self, path: str, include_optimizer: bool = False) -> None:
        self.build(skip=(REBUILD_DATA,))
        ckpt = {
            "kind": type(self).__name__,
            "config": self.get_config(),
            "model_state": self._model.state_dict(),
            "loss_history": self.loss_history,
            "extra": self._extra_state(),
        }
        if include_optimizer and self._optimizer is not None:
            ckpt["optimizer_state"] = self._optimizer.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = False) -> None:
        ckpt = torch.load(path, map_location=self._device())
        self.set_config(**ckpt["config"])
        self.build(skip=(REBUILD_DATA,))
        self._model.load_state_dict(ckpt["model_state"])
        self.loss_history = ckpt.get("loss_history", [])
        self._load_extra_state(ckpt.get("extra", {}))
        if load_optimizer and "optimizer_state" in ckpt:
            self._optimizer.load_state_dict(ckpt["optimizer_state"])

    # ------------------------------------------------------------------ #
    # Visualisation (shared). Subclasses customise behaviour through the two
    # small hooks below rather than reimplementing the plotting code.
    # ------------------------------------------------------------------ #
    def _grid_images(self, n: int, **kwargs: Any) -> torch.Tensor:
        """Return an ``(n, C, H, W)`` CPU tensor ready for ``make_grid``.

        Default: call ``generate`` and assume it already returns image-shaped
        tensors. Models whose ``generate`` returns flat vectors (the VAE)
        override this to reshape. ``kwargs`` is forwarded to ``generate`` so
        sampler options (e.g. the DDPM's ``use_ddim``) pass straight through.
        """
        return self.generate(n, **kwargs).detach().cpu()

    def _loss_label(self) -> Tuple[str, str]:
        """Return ``(y_axis_label, plot_title)`` for the loss curve."""
        return ("Loss", f"{type(self).__name__} Training Loss")

    @staticmethod
    def _display_image(path: str) -> None:
        """Show a saved PNG inline in a notebook, or print its path otherwise."""
        try:
            from IPython.display import Image, display  # type: ignore
            display(Image(filename=path))
        except Exception:
            print(f"image saved to: {path}  (open it with `open {path}`)")

    def show_samples(self, n: int = 16, save_path: str = "samples.png",
                     show: bool = True, nrow: int = 4, **kwargs: Any) -> str:
        """Generate ``n`` images, save them as a grid PNG, and optionally show."""
        import os
        from torchvision.utils import make_grid, save_image

        grid = make_grid(self._grid_images(n, **kwargs), nrow=nrow)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_image(grid, save_path)
        print(f"[show_samples] saved: {save_path}")
        if show:
            self._display_image(save_path)
        return save_path

    def plot_samples(self, n: int = 16, nrow: int = 4,
                     figsize: Tuple[float, float] = (6, 6),
                     title: Optional[str] = None,
                     save_path: Optional[str] = None,
                     show: bool = True, **kwargs: Any) -> None:
        """Render generated samples directly with matplotlib."""
        import os
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        grid = make_grid(self._grid_images(n, **kwargs), nrow=nrow)
        img = grid.permute(1, 2, 0).numpy()

        plt.figure(figsize=figsize)
        if grid.shape[0] == 1:
            plt.imshow(img.squeeze(-1), cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path)
            print(f"[plot_samples] saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def show_loss(self, save_path: str = "loss.png", show: bool = True) -> str:
        """Plot the per-epoch loss curve and save it as a PNG."""
        if not self.loss_history:
            raise RuntimeError("loss_history is empty; call train() first.")
        import os
        import matplotlib.pyplot as plt

        ylabel, title = self._loss_label()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig = plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[show_loss] saved: {save_path}")
        if show:
            self._display_image(save_path)
        return save_path

    # ------------------------------------------------------------------ #
    @property
    def model(self) -> torch.nn.Module:
        self.build(skip=(REBUILD_DATA,))
        return self._model

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.get_config()!r})"
