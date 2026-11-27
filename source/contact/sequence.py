"""Motion geometry and contact tracks shared by single and comparison viewers."""

from typing import TYPE_CHECKING
from dataclasses import dataclass, replace
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .single_detection import JointGroundContact
from .underpressure import UnderPressureContact
from .dual_detection import FootContact

if TYPE_CHECKING:
    from fairmotion.core.motion import Motion


def world_transforms(motion: "Motion") -> np.ndarray:
    local = np.stack([pose.data for pose in motion.poses])
    world = np.empty_like(local)
    for index, joint in enumerate(motion.skel.joints):
        relative = joint.xform_from_parent_joint @ local[:, index]
        parent = joint.parent_joint
        world[:, index] = (relative if parent is None else
                           world[:, motion.skel.get_index_joint(parent)] @ relative)
    return world


@dataclass
class ContactTrack:
    """One method's ordered markers and labels; no rendering or model inference."""
    title: str
    labels: tuple[str, ...]
    points: np.ndarray
    contacts: np.ndarray
    known: np.ndarray
    speeds: np.ndarray | None = None
    detector: FootContact | JointGroundContact | None = None
    dt: float | None = None
    note: str = ""

    def set_thresholds(self, height: float | None = None, speed: float | None = None) -> None:
        if self.detector is None:
            raise ValueError("This contact track has no adjustable threshold detector")
        self.detector = replace(
            self.detector,
            height_threshold=self.detector.height_threshold if height is None else height,
            speed_threshold=self.detector.speed_threshold if speed is None else speed,
        )
        previous = np.concatenate((self.points[:1], self.points[:-1]))
        self.contacts = self.detector.detect(self.points, previous, self.dt)
        self.contacts[~self.known] = False

    def summary(self) -> dict:
        return dict(method=self.title, labels=self.labels,
                    contact_fraction=self.contacts[self.known].mean(axis=0).tolist()
                    if self.known.any() else None,
                    known_frames=int(self.known.sum()))


@dataclass
class ContactClip:
    motion: "Motion"
    positions: np.ndarray
    tracks: tuple[ContactTrack, ...]
    note: str = ""


def detector_track(
    detector: FootContact | JointGroundContact | UnderPressureContact,
    motion: "Motion",
    world: np.ndarray,
    *,
    first_frame_known: bool = True,
) -> ContactTrack:
    names = tuple(j.name for j in motion.skel.joints)
    if isinstance(detector, UnderPressureContact):
        static = np.stack([j.xform_from_parent_joint for j in motion.skel.joints])
        contacts = detector.predict(world[..., :3, 3], names, static, motion.fps)
        points = world[..., :3, 3][:, [names.index(n) for n in detector.joints]]
        prefix = detector.method == "underpressure-prefix-v1"
        return ContactTrack("UnderPressure past/current" if prefix else "UnderPressure offline",
                            detector.labels, points, contacts, np.ones(len(contacts), dtype=bool),
                            note="Joint-center markers | " + ("No future input" if prefix else "USES FUTURE FRAMES"))
    points = detector.points(world[..., :3, :3], world[..., :3, 3], names)
    previous = np.concatenate((points[:1], points[:-1]))
    dt = 1 / motion.fps
    known = np.ones(motion.num_frames(), dtype=bool)
    known[0] = first_frame_known
    contacts = detector.detect(points, previous, dt)
    contacts[~known] = False
    speeds = np.linalg.norm(points - previous, axis=-1) / dt
    if not first_frame_known:
        speeds[0] = np.nan
    joint_centers = isinstance(detector, JointGroundContact)
    labels = detector.joints if joint_centers else detector.labels
    return ContactTrack("single_detection" if joint_centers else "dual_detection",
                        labels, points, contacts, known, speeds, detector, dt,
                        "First frame: stationary assumption" if first_frame_known else "First frame: no history")


def prepare_contact_clip(
    path: Path,
    detectors: Iterable[FootContact | JointGroundContact | UnderPressureContact],
) -> ContactClip:
    from util import load_canonicalized_bvh
    motion = load_canonicalized_bvh(path)
    world = world_transforms(motion)
    tracks = tuple(detector_track(detector, motion, world) for detector in detectors)
    return ContactClip(motion, world[..., :3, 3], tracks)
