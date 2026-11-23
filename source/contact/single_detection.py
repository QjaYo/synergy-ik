"""Single detection: one ankle center per foot, two default channels.

Reference: sunny-Codes/fairmotion, fairmotion/utils/contact_utils.py,
get_all_joint_ground_contact. Use meters and actual elapsed time here instead
of the reference's centimeter coordinates and fixed 30 FPS.
"""

from dataclasses import asdict, dataclass

from .threshold import detect_ground_contact


ANKLE_JOINTS = ("LeftFoot", "RightFoot")


@dataclass(frozen=True)
class JointGroundContact:
    joints: tuple[str, ...] = ANKLE_JOINTS
    height_threshold: float = 0.120
    speed_threshold: float = 0.4
    ground_y: float = 0.0
    method: str = "joint-ground-v1"

    def __post_init__(self):
        if not self.joints or len(set(self.joints)) != len(self.joints):
            raise ValueError("Contact joints must be nonempty and unique")
        if self.height_threshold <= 0 or self.speed_threshold <= 0:
            raise ValueError("Contact thresholds must be positive")

    @property
    def width(self):
        return len(self.joints)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["joints"] = tuple(data["joints"])
        return cls(**data)

    def points(self, world_rotations, world_positions, names):
        """Select joint centers in channel order; no sole offsets or mesh are used."""
        indices = [names.index(name) for name in self.joints]
        return world_positions[..., indices, :]

    def detect(self, current, previous, dt):
        return detect_ground_contact(current, previous, dt, self.height_threshold,
                                     self.speed_threshold, self.ground_y)
