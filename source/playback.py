"""Playback state, solver calls, and per-frame result reporting."""

from collections.abc import Callable
from functools import partial
import traceback

import numpy as np

from util import copy_pose, get_joint_world_positions, get_ik_dof_count
from solver import IKStepResult, SolverMode, solve_ik_step
from fairmotion.core.motion import Motion, Pose


IKSolver = Callable[[Pose, tuple[str, ...], np.ndarray, float], IKStepResult]
DEFAULT_END_EFFECTORS = (
    "LeftHandEnd", "RightHandEnd", "LeftToeEnd", "RightToeEnd",
)


def print_ik_result(result: IKStepResult, ee_names: tuple[str, ...]) -> None:
    """Print iteration count, convergence, and final per-EE errors in millimeters."""
    error_text = ", ".join(
        f"{name}={distance * 1000:.6f} mm"
        for name, distance in zip(ee_names, result.ee_errors)
    ) or "no EE"
    if result.orientation_errors is not None:
        error_text += " | orientation: " + ", ".join(
            f"{name}={np.degrees(angle):.6f} deg" for name, angle in zip(ee_names, result.orientation_errors))
    print(
        f"IK updates={result.iterations}/{result.max_iterations} | "
        f"converged={result.converged} | {error_text}"
    )


class IKPlayback:
    """Manage frames, EE selection, and independent baseline/weighted IK poses."""

    def __init__(
        self,
        motion: Motion,
        solver: IKSolver,
        *,
        panel_label: str,
        prepare_solver=None,
        reset_solver=None,
        fixed_ee_names: tuple[str, ...] | None = None,
        pinv_rtol: float = 1e-12,
        constrain_orientation: bool = False,
    ) -> None:
        self.motion = motion
        self.solver = solver
        self.panel_label = panel_label
        self.prepare_solver = prepare_solver
        self.reset_solver = reset_solver
        self.fixed_ee_names = fixed_ee_names
        self.constrain_orientation = constrain_orientation
        num_dofs = get_ik_dof_count(motion.skel)
        self.identity_solver = partial(
            solve_ik_step, mode=SolverMode.DIAGONAL, W=np.eye(num_dofs), pinv_rtol=pinv_rtol,
        )
        self.dt = 1.0 / motion.fps
        self.joint_names = tuple(j.name for j in motion.skel.joints)
        self.selected_ee = set(fixed_ee_names or DEFAULT_END_EFFECTORS)
        self.frame_index = 0
        self.paused = True
        self.accumulator = 0.0
        self.reset_pose()

    @property
    def reference_pose(self) -> Pose:
        return self.motion.get_pose_by_frame(self.frame_index)

    @property
    def ee_names(self) -> tuple[str, ...]:
        if self.fixed_ee_names is not None:
            return self.fixed_ee_names
        return tuple(name for name in self.joint_names if name in self.selected_ee)

    def target_positions(self) -> np.ndarray:
        return get_joint_world_positions(self.reference_pose, self.ee_names)

    def comparison_poses(self) -> list[tuple[str, Pose]]:
        """Return panel labels and poses in reference, identity, weighted order."""
        return [
            ("REFERENCE MOTION", self.reference_pose),
            ("IDENTITY IK", self.identity_pose),
            (self.panel_label, self.ik_pose),
        ]

    def reset_pose(self) -> None:
        """Synchronize both IK poses to the current reference without changing playback time."""
        self.ik_pose = copy_pose(self.reference_pose)
        self.previous_ik_pose = copy_pose(self.ik_pose)
        self.identity_pose = copy_pose(self.reference_pose)
        self.accumulator = 0.0
        self.status = "Pose synchronized; ready"
        if self.reset_solver is not None:
            self.reset_solver()

    def rewind(self) -> None:
        self.frame_index = 0
        self.paused = True
        self.reset_pose()

    def pause(self, status: str = "Paused") -> None:
        """Pause without changing the frame or either IK pose."""
        self.paused = True
        self.accumulator = 0.0
        self.status = status

    def toggle_playback(self) -> None:
        """Toggle playback, or reset to the first pose and play after reaching the end."""
        if self.frame_index == self.motion.num_frames() - 1:
            self.rewind()
            self.paused = self.motion.num_frames() == 1
        else:
            self.paused = not self.paused
        self.accumulator = 0.0

    def toggle_end_effector(self, name: str) -> None:
        """Pause and toggle an EE without resetting either IK pose."""
        if self.fixed_ee_names is not None:
            raise ValueError("EE selection is fixed by this learned checkpoint")
        if name not in self.joint_names:
            raise ValueError(f"Unknown joint: {name}")
        self.paused = True
        self.accumulator = 0.0
        if name in self.selected_ee:
            self.selected_ee.remove(name)
        else:
            self.selected_ee.add(name)
        self.status = "EE selection changed; press Space to play"

    def set_all_end_effectors(self, selected: bool) -> None:
        """Select or clear every joint and pause, preserving both IK poses."""
        if self.fixed_ee_names is not None:
            raise ValueError("EE selection is fixed by this learned checkpoint")
        self.paused = True
        self.accumulator = 0.0
        self.selected_ee = set(self.joint_names) if selected else set()
        self.status = "All EE selected; press Space to play" if selected else "No EE selected; IK poses held"

    def step_frame(self) -> None:
        """Pause playback and advance exactly one frame."""
        self.paused = True
        self.accumulator = 0.0
        self.step_once()

    def step_once(self) -> None:
        """Advance one reference frame and update both IK poses against the same targets."""
        if self.frame_index >= self.motion.num_frames() - 1:
            self.paused = True
            return
        next_frame = self.frame_index + 1
        names = self.ee_names
        if not names:
            self.previous_ik_pose = copy_pose(self.ik_pose)
            self.frame_index = next_frame
            self.status = "No EE selected; IK poses held"
        else:
            try:
                targets = get_joint_world_positions(self.motion.get_pose_by_frame(next_frame), names)
                task_options = {}
                if self.constrain_orientation:
                    reference = self.motion.get_pose_by_frame(next_frame)
                    task_options["target_rotations"] = np.stack([
                        reference.get_transform(name, local=False)[:3, :3] for name in names])
                if self.prepare_solver is not None:
                    self.prepare_solver(self.ik_pose, self.previous_ik_pose, self.dt, frame_index=self.frame_index)
                solver_runs = [
                    ("IDENTITY IK", self.identity_solver, self.identity_pose),
                    (self.panel_label, self.solver, self.ik_pose),
                ]

                results = []
                print(f"Frame {next_frame + 1}/{self.motion.num_frames()}")
                for label, solver, prev_pose in solver_runs:
                    print(f"{label} | ", end="")
                    result = solver(copy_pose(prev_pose), names, targets.copy(), self.dt, **task_options)
                    if not isinstance(result, IKStepResult):
                        raise ValueError("The solver must return an IKStepResult")
                    if result.pose.skel is not self.motion.skel:
                        raise ValueError("The result pose must use prev_pose.skel")
                    data = np.asarray(result.pose.data)
                    if data.shape != (len(self.joint_names), 4, 4) or not np.isfinite(data).all():
                        raise ValueError("Returned pose must have finite (N, 4, 4) transforms")
                    print_ik_result(result, names)
                    results.append(copy_pose(result.pose))

                # Commit only after both solver results have been validated.
                self.identity_pose = results[0]
                self.previous_ik_pose = copy_pose(self.ik_pose)
                self.ik_pose = results[-1]
                self.frame_index = next_frame
                self.status = "IK step applied"
            except Exception as error:
                self.paused = True
                self.status = f"IK error: {type(error).__name__}; see terminal"
                traceback.print_exc()
            finally:
                print()
        if self.frame_index == self.motion.num_frames() - 1:
            self.paused = True

    def advance_time(self, elapsed: float) -> None:
        """Advance in fixed time steps without skipping frames, independently of rendering."""
        if self.paused:
            return
        self.accumulator += elapsed
        # Process at most eight frames and carry the remaining time to the next call.
        for _ in range(8):
            if self.paused or self.accumulator + 1e-12 < self.dt:
                break
            self.accumulator -= self.dt
            self.step_once()
        if self.paused:
            self.accumulator = 0.0
