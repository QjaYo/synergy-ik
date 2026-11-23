"""Manually specified cost matrix in q_dot order."""

import numpy as np

from util import get_ik_dof_count
from fairmotion.core.motion import Skeleton


def create_fixed_weights(skel: Skeleton) -> np.ndarray:
    """Unit costs for root translation and non-leaf local rotations."""
    return np.eye(get_ik_dof_count(skel))
