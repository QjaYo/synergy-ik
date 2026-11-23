"""Interchangeable contact detectors shared by preprocessing and inference.

All detectors expose width, ground_y and to_dict(). detect_sequence handles
full clips; heuristic detectors use pairs of points, UnderPressure uses prefixes.
Positions are world-space meters; dt is elapsed physical time in seconds.
"""

import json
from pathlib import Path

from .single_detection import ANKLE_JOINTS, JointGroundContact
from .dual_detection import SITE_JOINTS, SITE_LABELS, FootContact
from .underpressure import UnderPressureContact


def contact_from_dict(data):
    """Restore the recorded method; do not replace it with another detector."""
    methods = {"rig-fixed-sole-sites-v1": FootContact, "joint-ground-v1": JointGroundContact,
               "underpressure-offline-v1": UnderPressureContact, "underpressure-prefix-v1": UnderPressureContact}
    method = data["method"]
    if method not in methods:
        raise ValueError(f"Unknown contact method: {method}")
    return methods[method].from_dict(data)


def load_contact(path):
    return contact_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def encode_contact(contact):
    """The baseline passes binary contact directly; no learned encoder."""
    return contact


def detect_sequence(detector, motion, world_rotations, world_positions):
    """Return F x C labels, preserving each detector's temporal semantics."""
    import numpy as np
    names = tuple(j.name for j in motion.skel.joints)
    if isinstance(detector, UnderPressureContact):
        static = np.stack([j.xform_from_parent_joint for j in motion.skel.joints])
        return detector.predict(world_positions, names, static, motion.fps)
    points = detector.points(world_rotations, world_positions, names)
    previous = np.concatenate((points[:1], points[:-1]))
    return detector.detect(points, previous, 1 / motion.fps)


def encode_surface_contact(contact):
    raise NotImplementedError("Surface-contact encoding is a later experiment")


def detect_body_contact(*args, **kwargs):
    raise NotImplementedError("Whole-body contact needs agreed sites and thresholds")
