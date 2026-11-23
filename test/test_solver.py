"""Verify reduced IK against independent finite differences and full-coordinate IK."""

from contextlib import redirect_stdout
from functools import partial
from io import StringIO
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from util import (
    copy_pose, get_ik_dof_count, get_joint_world_positions,
    get_rotation_joint_indices,
)
from solver import (
    SolverMode, apply_joint_velocity, compute_jacobian_analytic,
    compute_jacobian_autograd, compute_pseudoinverse, solve_ik_step,
)
from playback import DEFAULT_END_EFFECTORS, IKPlayback
from weights.fixed import create_fixed_weights
from fairmotion.core.motion import Joint, Motion, Pose, Skeleton


# Canonical topology, with nonzero offsets and rotations to expose indexing errors.
JOINTS = (
    ("Hips", None), ("Spine", 0), ("Spine1", 1), ("Spine2", 2),
    ("Neck1", 3), ("Neck2", 4), ("Head", 5), ("HeadEnd", 6),
    ("LeftShoulder", 3), ("LeftArm", 8), ("LeftForeArm", 9),
    ("LeftHand", 10), ("LeftHandEnd", 11),
    ("RightShoulder", 3), ("RightArm", 13), ("RightForeArm", 14),
    ("RightHand", 15), ("RightHandEnd", 16),
    ("LeftUpLeg", 0), ("LeftLeg", 18), ("LeftFoot", 19),
    ("LeftToeBase", 20), ("LeftToeEnd", 21),
    ("RightUpLeg", 0), ("RightLeg", 23), ("RightFoot", 24),
    ("RightToeBase", 25), ("RightToeEnd", 26),
)
LEAF_INDICES = (7, 12, 17, 22, 27)
ROTATION_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26)


def make_pose() -> Pose:
    rng = np.random.default_rng(413)
    skel = Skeleton()
    data = np.repeat(np.eye(4)[None], len(JOINTS), axis=0)
    for j, (name, parent) in enumerate(JOINTS):
        static = np.eye(4)
        if parent is not None:
            static[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.2, 3)).as_matrix()
            static[:3, 3] = rng.uniform(-0.3, 0.3, 3)
        skel.add_joint(
            Joint(name, dof=6 if parent is None else 3, xform_from_parent_joint=static),
            None if parent is None else JOINTS[parent][0],
        )
        data[j, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix()
    data[0, :3, 3] = (0.4, 1.1, -0.2)
    return Pose(skel, data)


def apply_reference_delta(pose: Pose, delta: np.ndarray, indices: tuple[int, ...]) -> Pose:
    """Independent SciPy update, including all rotations when given all 28 indices."""
    result = copy_pose(pose)
    result.data[0, :3, 3] += delta[:3]
    for i, j in enumerate(indices):
        result.data[j, :3, :3] = pose.data[j, :3, :3] @ Rotation.from_rotvec(
            delta[3 + 3*i:6 + 3*i]
        ).as_matrix()
    return result


def finite_difference_jacobian(pose: Pose, names: tuple[str, ...], indices: tuple[int, ...]) -> np.ndarray:
    epsilon = 1e-6
    J = np.empty((3 * len(names), 3 + 3 * len(indices)))
    for column in range(J.shape[1]):
        delta = np.zeros(J.shape[1])
        delta[column] = epsilon
        plus = get_joint_world_positions(apply_reference_delta(pose, delta, indices), names)
        minus = get_joint_world_positions(apply_reference_delta(pose, -delta, indices), names)
        J[:, column] = ((plus - minus) / (2 * epsilon)).reshape(-1)
    return J


class ReducedIKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = make_pose()

    def test_mapping_and_fixed_weights_keep_head_and_wrist_rotations(self) -> None:
        self.assertEqual(get_rotation_joint_indices(self.pose.skel), ROTATION_INDICES)
        self.assertEqual(get_ik_dof_count(self.pose.skel), 72)
        np.testing.assert_array_equal(create_fixed_weights(self.pose.skel), np.eye(72))
        self.assertEqual(self.pose.data.shape, (28, 4, 4))

    def test_analytic_and_autograd_match_independent_finite_difference(self) -> None:
        # All node positions exercise head/wrist rotations and both body branches.
        names = tuple(name for name, _ in JOINTS)
        analytic = compute_jacobian_analytic(self.pose, names)
        autograd = compute_jacobian_autograd(self.pose, names)
        reference = finite_difference_jacobian(self.pose, names, ROTATION_INDICES)
        self.assertEqual(analytic.shape, (84, 72))
        np.testing.assert_allclose(analytic, reference, atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(autograd, reference, atol=1e-8, rtol=1e-7)

    def test_ee_selection_changes_only_rows(self) -> None:
        for names in ((), ("Hips",), ("HeadEnd",), DEFAULT_END_EFFECTORS):
            with self.subTest(names=names):
                J = compute_jacobian_analytic(self.pose, names)
                self.assertEqual(J.shape, (3 * len(names), 72))
                np.testing.assert_array_equal(J[:, :3], np.tile(np.eye(3), (len(names), 1)))

    def test_velocity_update_preserves_leaf_transforms_and_static_skeleton(self) -> None:
        velocity = np.linspace(-0.7, 0.8, 72)
        dt = 1 / 120
        before = self.pose.data.copy()
        static = np.array([j.xform_from_parent_joint.copy() for j in self.pose.skel.joints])
        expected = apply_reference_delta(self.pose, velocity * dt, ROTATION_INDICES)
        result = apply_joint_velocity(self.pose, velocity, dt)
        self.assertIs(result, self.pose)
        np.testing.assert_allclose(result.data, expected.data, atol=1e-14)
        np.testing.assert_array_equal(result.data[list(LEAF_INDICES)], before[list(LEAF_INDICES)])
        np.testing.assert_array_equal([j.xform_from_parent_joint for j in result.skel.joints], static)

    def test_diagonal_reduction_matches_full_87_component_solution(self) -> None:
        J_full = finite_difference_jacobian(self.pose, DEFAULT_END_EFFECTORS, tuple(range(28)))
        J = compute_jacobian_analytic(self.pose, DEFAULT_END_EFFECTORS)
        active_columns = [0, 1, 2] + [3 + 3*j + a for j in ROTATION_INDICES for a in range(3)]
        full_weights = np.linspace(0.3, 4.0, 87)
        S = np.diag(full_weights ** -0.5)
        velocity_target = np.linspace(-0.3, 0.5, 12)
        full_velocity = S @ np.linalg.pinv(J_full @ S) @ velocity_target
        reduced_pinv = compute_pseudoinverse(J, SolverMode.DIAGONAL, np.diag(full_weights[active_columns]))
        self.assertEqual(reduced_pinv.shape, (72, 12))
        velocity = reduced_pinv @ velocity_target
        np.testing.assert_allclose(velocity, full_velocity[active_columns], atol=1e-8, rtol=1e-7)
        expected = apply_reference_delta(self.pose, full_velocity / 120, tuple(range(28)))
        result = apply_joint_velocity(copy_pose(self.pose), velocity, 1 / 120)
        np.testing.assert_allclose(result.data, expected.data, atol=1e-10)

    def test_solver_and_playback_use_reduced_weights_with_full_poses(self) -> None:
        motion = Motion(skel=self.pose.skel, fps=120)
        for frame in range(3):
            data = self.pose.data.copy()
            data[0, :3, 3] += (0.003 * frame, -0.001 * frame, 0.002 * frame)
            motion.add_one_frame(data)
        W = create_fixed_weights(motion.skel)
        solver = partial(solve_ik_step, mode=SolverMode.DIAGONAL, W=W)
        playback = IKPlayback(motion, solver, panel_label="FIXED DIAGONAL IK")
        self.assertTrue(playback.paused)
        self.assertEqual(playback.identity_solver.keywords["W"].shape, (72, 72))
        playback.set_all_end_effectors(True)
        with redirect_stdout(StringIO()):
            playback.step_frame()
        self.assertEqual(playback.status, "IK step applied")
        for pose in (playback.ik_pose, playback.identity_pose):
            self.assertEqual(pose.data.shape, (28, 4, 4))
            np.testing.assert_array_equal(pose.data[list(LEAF_INDICES)], self.pose.data[list(LEAF_INDICES)])
            errors = get_joint_world_positions(pose, playback.ee_names) - playback.target_positions()
            self.assertLessEqual(np.linalg.norm(errors, axis=1).max(), 1e-3)
        held = playback.ik_pose.data.copy()
        playback.set_all_end_effectors(False)
        playback.step_frame()
        np.testing.assert_array_equal(playback.ik_pose.data, held)
        playback.reset_pose()
        np.testing.assert_array_equal(playback.ik_pose.data, playback.reference_pose.data)

    def test_old_dimensions_fail_before_updating_pose(self) -> None:
        original = self.pose.data.copy()
        with self.assertRaisesRegex(ValueError, "q_dot must have shape"):
            apply_joint_velocity(self.pose, np.zeros(87), 1 / 120)
        np.testing.assert_array_equal(self.pose.data, original)
        J = compute_jacobian_analytic(self.pose, DEFAULT_END_EFFECTORS)
        with self.assertRaisesRegex(ValueError, "W must have shape"):
            compute_pseudoinverse(J, SolverMode.DIAGONAL, np.eye(87))


if __name__ == "__main__":
    unittest.main()
