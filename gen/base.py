"""Shared skeleton for all generative models in the ``gen`` library.

``BaseModel`` factors out everything the VAE and the DDPM trainers have in
common: a validated configuration dictionary, a declarative schema, lazy
("stale set") component rebuilding, optimizer construction, checkpoint
save/load, and the visualisation helpers. Concrete models (``gen.VAE``,
``gen.DDPM``) subclass it and supply only what is genuinely model-specific.

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

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.optim import Adam

# Rebuildable-component tags. Subclasses may add their own (e.g. the DDPM
# adds a "schedule" tag) but these three are understood by the base class.
REBUILD_MODEL = "model"
REBUILD_OPTIM = "optim"
REBUILD_DATA = "data"

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

    @abstractmethod
    def _build_data(self) -> None:
        """Build the data loader(s) from ``self._custom_dataset`` -- or, when
        that is None, from the model's own dataset -- and clear REBUILD_DATA."""

    def set_dataset(self, dataset: Any) -> None:
        """Train on a caller-supplied dataset instead of the built-in MNIST.

        Each item must be ``(image_tensor, label)``, with the image already in
        the pixel range the model trains on: ``[0, 1]`` for :class:`gen.VAE`
        (its Bernoulli/BCE reconstruction term needs probabilities) and
        ``[-1, 1]`` for :class:`gen.DDPM` (its sampler returns to that range).
        Passing ``None`` restores the built-in dataset.

        Marks the data loader stale, so the next ``build`` rebuilds it.
        """
        self._custom_dataset = dataset
        self._stale.add(REBUILD_DATA)

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
