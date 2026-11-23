"""Atomic training checkpoints and random-state restoration for exact continuation."""

import random

import numpy as np
import torch


def write_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def random_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_random_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
