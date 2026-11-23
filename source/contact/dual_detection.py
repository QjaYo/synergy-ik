"""Dual detection: front/rear per foot, four channels; meters and seconds."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

from .threshold import detect_ground_contact


SITE_LABELS = ("Left front", "Right front", "Left rear", "Right rear")
SITE_JOINTS = ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")


@dataclass(frozen=True)
class FootContact:
    local_offsets_m: tuple[tuple[float, float, float], ...]
    rig_sha256: str
    height_threshold: float = 0.040
    speed_threshold: float = 0.25
    ground_y: float = 0.0
    method: str = "rig-fixed-sole-sites-v1"
    joints: tuple[str, ...] = SITE_JOINTS
    labels: tuple[str, ...] = SITE_LABELS

    @property
    def width(self):
        return len(self.joints)

    def __post_init__(self):
        offsets = np.asarray(self.local_offsets_m)
        if offsets.shape != (len(self.joints), 3) or not np.isfinite(offsets).all():
            raise ValueError("Sole offsets must be one finite xyz vector per attached joint")
        if len(self.labels) != len(self.joints) or self.height_threshold <= 0 or self.speed_threshold <= 0:
            raise ValueError("Invalid sole labels or contact thresholds")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("joints", "labels"):
            data[key] = tuple(data[key])
        data["local_offsets_m"] = tuple(tuple(v) for v in data["local_offsets_m"])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def points(self, world_rotations, world_positions, names):
        indices = [names.index(name) for name in self.joints]
        return (np.einsum("...cij,cj->...ci", world_rotations[..., indices, :, :],
                          np.asarray(self.local_offsets_m)) + world_positions[..., indices, :])

    def detect(self, current, previous, dt):
        return detect_ground_contact(current, previous, dt, self.height_threshold,
                                     self.speed_threshold, self.ground_y)
