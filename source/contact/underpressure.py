"""UnderPressure contact with an unchanged upstream model in a worker.

Inputs are canonical world joint positions in meters. Outputs are frame-aligned
Boolean [left front, right front, left rear, right rear] labels. Prefix mode
evaluates each frame from its available past only. Offline mode is comparison-only.
"""

import atexit
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence

import numpy as np

from .dual_detection import SITE_JOINTS, SITE_LABELS

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "e7ab73e7466444f262ec1c55f2f26afe51168796"
ASSETS = ("models.py", "data.py", "util.py", "anim.py", "pretrained.tar", "footskate_samples/0.pt")
# The former vendored checkout used CRLF for the four Python files.
# Both digests include the exact bytes of the weights and reference sample.
VENDORED_FINGERPRINT = "8318647ec1e344ae1bf8c2b5fcf27f62923cd8796ff9992611483d66024d2bd5"
UPSTREAM_FINGERPRINT = "021a2ff3c60b29c887aa08b5fa6f9000dba772f058678f44c1968d3647bdc0a6"


def canonical_fingerprint(fingerprint: str) -> str:
    return UPSTREAM_FINGERPRINT if fingerprint == VENDORED_FINGERPRINT else fingerprint


def upstream_directory() -> Path:
    return Path(os.environ.get("SYNERGYIK_UNDERPRESSURE_ROOT", ROOT / "third_party/UnderPressure")).resolve()


@lru_cache(maxsize=None)
def asset_fingerprint(directory: str | Path) -> str:
    digest = hashlib.sha256()
    for name in ASSETS:
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing UnderPressure asset: {path}; see third_party/README.md")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def spine_levels(directory: str | Path) -> np.ndarray:
    import torch
    sample = torch.load(Path(directory) / "footskate_samples/0.pt", map_location="cpu", weights_only=True)
    reference = sample["skeleton"][0].numpy()
    return reference[1:5, 2] / reference[5, 2]


def model_positions(
    positions: np.ndarray,
    names: Sequence[str],
    static: np.ndarray,
    directory: str | Path | None = None,
    ground_y: float = 0.0,
) -> np.ndarray:
    """Map Bones-seed to the existing experimental 23-joint, Z-up position adapter."""
    directory = Path(directory) if directory is not None else upstream_directory()
    indices = {name: i for i, name in enumerate(names)}
    spine_names = ("Hips", "Spine", "Spine1", "Spine2", "Neck1", "Neck2")
    lengths = np.linalg.norm(static[[indices[n] for n in spine_names[1:]], :3, 3], axis=-1)
    arc = np.r_[0, np.cumsum(lengths)]
    arc /= arc[-1]
    spine = positions[:, [indices[n] for n in spine_names]]
    virtual = []
    for level in spine_levels(str(directory)):
        j = np.searchsorted(arc, level) - 1
        fraction = (level - arc[j]) / (arc[j + 1] - arc[j])
        virtual.append(spine[:, j] * (1 - fraction) + spine[:, j + 1] * fraction)
    result = [positions[:, indices["Hips"]], *virtual,
              positions[:, indices["Neck2"]], positions[:, indices["Head"]]]
    for side in ("Right", "Left"):
        result.extend(positions[:, indices[side + joint]] for joint in ("Shoulder", "Arm", "ForeArm", "Hand"))
    for side in ("Right", "Left"):
        result.extend(positions[:, indices[side + joint]] for joint in ("UpLeg", "Leg", "Foot", "ToeBase"))
    result = np.stack(result, axis=1).copy()
    result -= result[0, 0] * np.array([1, 0, 1])
    result[..., 1] -= ground_y
    return (result[..., [0, 2, 1]] * np.array([1, -1, 1])).astype(np.float32)


class _Worker:
    """One model load per process; serialize requests from viewer/background threads."""
    def __init__(self, directory: Path) -> None:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        self.process = subprocess.Popen(
            [sys.executable, "-B", "-u", str(Path(__file__).with_name("underpressure_worker.py")), str(directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", **options)
        self.lock = threading.Lock()

    def predict(self, positions: np.ndarray, fps: float, mode: str) -> dict[str, np.ndarray]:
        with self.lock, tempfile.TemporaryDirectory(prefix="synergyik-contact-") as directory:
            source, target = Path(directory) / "input.npz", Path(directory) / "output.npz"
            np.savez(source, positions=positions, fps=fps)
            self.process.stdin.write(json.dumps(dict(source=str(source), target=str(target), mode=mode)) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"UnderPressure worker exited ({self.process.poll()}); see its stderr traceback")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError("UnderPressure inference failed:\n" + response["error"])
            with np.load(target, allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait()
        self.process.stdout.close()


_workers = {}
_workers_lock = threading.Lock()


def predict_model_positions(
    positions: np.ndarray,
    fps: float,
    directory: str | Path | None = None,
    *,
    mode: str = "offline",
) -> dict[str, np.ndarray]:
    directory = Path(directory).resolve() if directory is not None else upstream_directory()
    asset_fingerprint(str(directory))
    key = (os.getpid(), str(directory))
    with _workers_lock:
        if key not in _workers:
            _workers[key] = _Worker(directory)
        worker = _workers[key]
    return worker.predict(positions, fps, mode)


def close_workers() -> None:
    for (pid, _), worker in list(_workers.items()):
        if pid == os.getpid():
            worker.close()
    _workers.clear()


atexit.register(close_workers)


@dataclass(frozen=True)
class UnderPressureContact:
    method: str = "underpressure-prefix-v1"
    ground_y: float = 0.0
    fingerprint: str | None = None

    width = 4
    labels = SITE_LABELS
    joints = SITE_JOINTS

    def __post_init__(self) -> None:
        if self.method not in ("underpressure-prefix-v1", "underpressure-offline-v1"):
            raise ValueError(f"Unknown UnderPressure timing: {self.method}")

    def to_dict(self) -> dict:
        fingerprint = canonical_fingerprint(asset_fingerprint(str(upstream_directory())))
        if self.fingerprint is not None and canonical_fingerprint(self.fingerprint) != fingerprint:
            raise ValueError("UnderPressure assets differ from the recorded contact contract")
        return dict(method=self.method, ground_y=self.ground_y, fingerprint=fingerprint)

    @classmethod
    def from_dict(cls, data: dict) -> 'UnderPressureContact':
        return cls(**data)

    def predict(
        self,
        world_positions: np.ndarray,
        names: Sequence[str],
        static: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        self.to_dict()
        positions = model_positions(world_positions, names, static, ground_y=self.ground_y)
        mode = "prefix" if self.method == "underpressure-prefix-v1" else "offline"
        return predict_model_positions(positions, fps, mode=mode)["contacts"]

    def predict_current(
        self,
        world_positions_history: np.ndarray,
        names: Sequence[str],
        static: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """Use only the provided realized history; return its latest label without revising earlier ones."""
        if self.method != "underpressure-prefix-v1":
            raise ValueError("Offline reference contacts cannot be used for live IK")
        self.to_dict()
        positions = model_positions(world_positions_history, names, static, ground_y=self.ground_y)
        return predict_model_positions(positions, fps)["contacts"][-1]
