"""Atomic training checkpoints and random-state restoration for exact continuation."""

import random
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_VERSION = 3


def read_checkpoint(path: str | Path) -> dict:
    """Load a learned checkpoint with the current model configuration fields."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    version = saved["version"]
    if version == 1:
        model_config = saved["model_config"]
        if model_config.pop("cost_structure", "spd") != "spd":
            raise ValueError("Unsupported checkpoint cost structure")
        # Version 1 counted only hidden layers in the model and CLI settings.
        model_config["depth"] += 1
        settings = saved.get("settings", {})
        if "depth" in settings:
            settings["depth"] += 1
        saved["version"] = CHECKPOINT_VERSION
    elif version not in (2, CHECKPOINT_VERSION):
        raise ValueError("Unsupported learned checkpoint version")
    if version in (1, 2):
        # Old checkpoints predate solver selection and used the whitening solve.
        saved["solve_config"].setdefault("method", "cholesky_pinv")
        saved.setdefault("settings", {}).setdefault("ik_solver", saved["solve_config"]["method"])
        saved["version"] = CHECKPOINT_VERSION
    return saved


def write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def random_state() -> dict:
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
