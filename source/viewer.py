"""OpenGL rendering, cameras, buttons, and input events for IK comparison."""

import numpy as np
import time

from util import get_joint_world_positions
from playback import IKPlayback
from weight_editor import DiagonalWeightEditor
from fairmotion.core.motion import Motion, Pose


REFERENCE_LINK_COLOR = (0.08, 0.20, 0.45, 1.0)
REFERENCE_JOINT_COLOR = (0.30, 0.70, 1.0, 1.0)
IK_LINK_COLOR = (0.50, 0.29, 0.15, 1.0)
IK_JOINT_COLOR = (0.20, 0.65, 0.25, 1.0)
WEIGHTED_LINK_COLOR = (0.40, 0.15, 0.56, 1.0)
WEIGHTED_JOINT_COLOR = (0.75, 0.46, 0.90, 1.0)
TARGET_COLOR = (0.95, 0.40, 0.05, 1.0)


def create_initial_camera(motion: Motion):
    """Position the camera to include the complete skeleton in the first frame."""
    from fairmotion.viz import camera

    first_pose = motion.get_pose_by_frame(0)
    positions = np.array([
        first_pose.get_transform(joint, local=False)[:3, 3]
        for joint in motion.skel.joints
    ])
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    radius = max(float(np.linalg.norm(positions - center, axis=1).max()), 0.5)
    view_direction = np.array([1.0, 0.4, 1.5])
    camera_distance = 1.2 * radius / np.sin(np.deg2rad(45.0 / 2))
    return camera.Camera(
        pos=center + camera_distance * view_direction / np.linalg.norm(view_direction),
        origin=center,
        vup=np.array([0.0, 1.0, 0.0]),
        fov=45.0,
    )


def create_comparison_viewer(
    playback: IKPlayback, title: str = "Position IK",
    weight_editor: DiagonalWeightEditor | None = None,
    *, bvh_path=None, uniform_root=None, uniform_rig=None, surface_height_threshold=0.040,
):
    """Create a three-panel reference, identity, and weighted IK viewer."""
    from OpenGL import GL, GLU, GLUT
    from fairmotion.viz import gl_render, glut_viewer

    class ComparisonViewer(glut_viewer.Viewer):
        def __init__(self) -> None:
            super().__init__(title=f"SynergyIK | {title}".encode("utf-8"),
                             cam=create_initial_camera(playback.motion), size=(1680, 860))
            self.buttons = []
            self.dragging_scene = False
            self.show_mesh = True
            self.show_skeleton = True
            self.surface = None
            self.surface_error = ""
            self.contact_threshold = surface_height_threshold
            if bvh_path is not None:
                from surface import IKSurface, DEFAULT_UNIFORM_ROOT, DEFAULT_UNIFORM_RIG
                try:
                    self.surface = IKSurface(playback.motion, bvh_path, uniform_root or DEFAULT_UNIFORM_ROOT,
                                             uniform_rig or DEFAULT_UNIFORM_RIG)
                except Exception as error:
                    self.surface_error = f"Mesh unavailable: {type(error).__name__}: {error}"
                    print(self.surface_error, flush=True)
            else:
                self.surface_error = "Mesh unavailable: no source BVH path"

        def scene_layout(self) -> tuple[int, int, int, int, int]:
            width, height = self.window_size
            columns = max(1, width // 195)
            rows = (len(playback.joint_names) + columns - 1) // columns
            bottom = 90 + rows * 29
            return width, height, columns, bottom, max(1, height - bottom - 110)

        def resize_GL(self, width: int, height: int) -> None:
            self.window_size = (max(width, 1), max(height, 1))

        def draw_pose(
            self, pose: Pose,
            link_color: tuple[float, float, float, float],
            joint_color: tuple[float, float, float, float],
        ) -> None:
            positions = get_joint_world_positions(pose, playback.joint_names)
            for index, joint in enumerate(pose.skel.joints):
                gl_render.render_point(positions[index], radius=0.015, color=joint_color)
                if joint.parent_joint is not None:
                    parent_index = pose.skel.get_index_joint(joint.parent_joint)
                    gl_render.render_line(positions[parent_index], positions[index],
                                          color=link_color, line_width=5.0)

        def draw_targets(self, pose: Pose, show_error: bool) -> None:
            targets = playback.target_positions()
            actual = get_joint_world_positions(pose, playback.ee_names)
            for target, current in zip(targets, actual):
                # All panels show the same target positions as crosses in world coordinates.
                for axis in np.eye(3) * 0.035:
                    gl_render.render_line(target - axis, target + axis,
                                          color=TARGET_COLOR, line_width=2.5)
                if show_error:
                    gl_render.render_line(current, target, color=TARGET_COLOR, line_width=1.0)
            if playback.constrain_orientation:
                for name, target in zip(playback.ee_names, targets):
                    rotation = playback.reference_pose.get_transform(name, local=False)[:3, :3]
                    gl_render.render_arrow(target, target + 0.15 * rotation[:, 2], D=0.012,
                                           color=(0.1, 0.35, 0.95, 1.0))

        def draw_GL(self, swap_buffer: bool = True) -> None:
            width, height, _, bottom, scene_height = self.scene_layout()
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_LIGHTING)
            scenes = playback.comparison_poses()
            colors = (
                (REFERENCE_LINK_COLOR, REFERENCE_JOINT_COLOR),
                (IK_LINK_COLOR, IK_JOINT_COLOR),
                (WEIGHTED_LINK_COLOR, WEIGHTED_JOINT_COLOR),
            )
            for panel, (_, pose) in enumerate(scenes):
                x = width * panel // len(scenes)
                panel_width = max(1, width * (panel + 1) // len(scenes) - x)
                links, joints = colors[panel]
                GL.glViewport(x, bottom, panel_width, scene_height)
                GL.glMatrixMode(GL.GL_PROJECTION)
                GL.glLoadIdentity()
                GLU.gluPerspective(self.cam_cur.fov, panel_width / scene_height, 0.01, 200.0)
                GL.glMatrixMode(GL.GL_MODELVIEW)
                GL.glLoadIdentity()
                GLU.gluLookAt(*self.cam_cur.pos, *self.cam_cur.origin, *self.cam_cur.vup)
                gl_render.render_ground(size=[100, 100], color=[0.82, 0.84, 0.87, 1],
                                        axis="y", origin=False)
                if self.show_mesh and self.surface is not None:
                    self.surface.draw(panel, pose, playback.frame_index, self.contact_threshold)
                if self.show_skeleton:
                    self.draw_pose(pose, links, joints)
                self.draw_targets(pose, show_error=panel > 0)
            GL.glViewport(0, 0, width, height)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glOrtho(0, width, height, 0, -1, 1)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            self.draw_controls()
            if swap_buffer:
                GLUT.glutSwapBuffers()

        def rectangle(
            self, x: float, y: float, width: float, height: float,
            color: tuple[float, float, float, float],
        ) -> None:
            GL.glColor4f(*color)
            GL.glBegin(GL.GL_QUADS)
            for point in ((x, y), (x+width, y), (x+width, y+height), (x, y+height)):
                GL.glVertex2f(*point)
            GL.glEnd()

        def text(
            self, message: str, x: float, y: float,
            color: tuple[float, float, float, float] = (0.14, 0.18, 0.24, 1),
        ) -> None:
            gl_render.render_text(message, [x, y], font=GLUT.GLUT_BITMAP_HELVETICA_12, color=color)

        def button(
            self, label: str, x: float, y: float, width: float,
            action: str, selected: bool = False,
            enabled: bool = True,
        ) -> None:
            color = (0.76, 0.88, 0.98, 1) if selected else (0.89, 0.91, 0.94, 1)
            self.rectangle(x, y, width, 24, color)
            self.text(label, x+7, y+16, (0.14, 0.18, 0.24, 1) if enabled else (0.55, 0.58, 0.62, 1))
            if enabled:
                self.buttons.append((x, y, width, 24, action))

        def draw_controls(self) -> None:
            width, height, columns, bottom, scene_height = self.scene_layout()
            self.buttons = []
            self.rectangle(0, 0, width, 110, (0.97, 0.98, 0.99, 1))
            self.rectangle(0, height-bottom, width, bottom, (0.97, 0.98, 0.99, 1))
            scenes = playback.comparison_poses()
            for panel in range(1, len(scenes)):
                x = width * panel // len(scenes)
                self.rectangle(x, 110, 1, scene_height, (0.73, 0.77, 0.82, 1))
            self.button("Play / Pause [Space]", 12, 12, 150, "play")
            self.button("Step [.]", 170, 12, 80, "step")
            self.button("Reset pose [R]", 258, 12, 120, "reset")
            self.button("Rewind", 386, 12, 80, "rewind")
            status_x = 480
            if weight_editor is not None:
                self.button("Edit W", 474, 12, 90, "edit_weights")
                status_x = 578
            label = "PAUSED" if playback.paused else "PLAYING"
            if playback.frame_index == playback.motion.num_frames()-1:
                label = "END"
            self.text(f"{label} | Frame {playback.frame_index+1}/{playback.motion.num_frames()} | "
                      f"dt = {playback.dt:.6f} s | EE: {len(playback.ee_names)}", status_x, 28)
            self.text(playback.status, 12, 56)
            self.text(self.surface_error or "Orange crosses = EE targets | Surface colors show height only", 12, 76)
            self.button("View Mesh [M]", width-440, 44, 120, "mesh", self.show_mesh, self.surface is not None)
            self.button("View Skeleton [S]", width-312, 44, 140, "skeleton", self.show_skeleton)
            self.button("Band -", width-164, 44, 70, "band_down")
            self.button("Band +", width-86, 44, 70, "band_up")
            self.text(f"Display band: {self.contact_threshold*1000:.0f} mm", width-160, 76)
            label_colors = (REFERENCE_LINK_COLOR, IK_JOINT_COLOR, WEIGHTED_LINK_COLOR)
            for panel, (panel_label, _) in enumerate(scenes):
                self.text(panel_label, width * panel // len(scenes) + 12, 96, label_colors[panel])
            y = height-bottom
            editable = playback.fixed_ee_names is None
            self.text("EE position constraints - click to toggle" if editable else "EE position constraints fixed by checkpoint", 12, y+20)
            self.button("Select all", 12, y+30, 100, "select_all", enabled=editable)
            self.button("Clear all", 120, y+30, 100, "clear_all", enabled=editable)
            cell_width = (width-24) / columns
            for index, name in enumerate(playback.joint_names):
                selected = name in playback.selected_ee
                self.button(("[x] " if selected else "[ ] ") + name,
                            12 + (index % columns)*cell_width, y+60+(index//columns)*29,
                            cell_width-6, name, selected, enabled=editable)
            self.text("Reset = sync all IK poses | Drag: orbit | Right drag: pan | Wheel: zoom | Esc: close",
                      12, height-9)

        def perform_action(self, action: str) -> None:
            if weight_editor is not None and weight_editor.is_open:
                if action == "edit_weights":
                    weight_editor.window.lift()
                return
            if action == "play":
                playback.toggle_playback()
            elif action == "step":
                playback.step_frame()
            elif action == "reset":
                playback.reset_pose()
            elif action == "rewind":
                playback.rewind()
            elif action == "select_all":
                playback.set_all_end_effectors(True)
            elif action == "clear_all":
                playback.set_all_end_effectors(False)
            elif action == "mesh":
                if self.surface is not None:
                    self.show_mesh = not self.show_mesh
            elif action == "skeleton":
                self.show_skeleton = not self.show_skeleton
            elif action == "band_up":
                self.contact_threshold += 0.005
            elif action == "band_down":
                self.contact_threshold = max(0.0, self.contact_threshold - 0.005)
            elif action == "edit_weights" and weight_editor is not None:
                playback.pause("Editing W; Apply or Cancel to continue")
                weight_editor.open(lambda applied: playback.pause(
                    "Fixed W updated; press Space to play" if applied
                    else "W unchanged; press Space to play"
                ))
            else:
                playback.toggle_end_effector(action)
            self.time_checker.begin()
            GLUT.glutPostRedisplay()

        def keyboard_callback(self, key: bytes) -> bool:
            if key == b"\x1b":
                GLUT.glutLeaveMainLoop()
                return True
            actions = {b" ": "play", b".": "step", b"r": "reset", b"R": "reset", b"m": "mesh", b"s": "skeleton"}
            if key in actions:
                self.perform_action(actions[key])
                return True
            return False

        def mouse_func(self, button: int, state: int, x: int, y: int) -> None:
            if state == GLUT.GLUT_UP:
                self.dragging_scene = False
                self.pressed_button = None
                self.mouse_last_pos = None
                return
            if button == GLUT.GLUT_LEFT_BUTTON:
                for bx, by, bw, bh, action in self.buttons:
                    if bx <= x < bx+bw and by <= y < by+bh:
                        self.perform_action(action)
                        return
            _, height, _, bottom, _ = self.scene_layout()
            if 110 <= y < height-bottom:
                super().mouse_func(button, state, x, y)
                self.dragging_scene = button in (GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_RIGHT_BUTTON)

        def motion_func(self, x: int, y: int) -> None:
            if self.dragging_scene and self.mouse_last_pos is not None:
                super().motion_func(x, y)

        def idle_callback(self) -> None:
            if weight_editor is not None:
                weight_editor.poll()
            elapsed = self.time_checker.get_time(restart=True)
            playback.advance_time(elapsed)
            time.sleep(0.01)
            GLUT.glutPostRedisplay()

        def run(self) -> None:
            from viewer_window import run_window
            run_window(self)

    return ComparisonViewer()


def run_ik_viewer(
    playback: IKPlayback, title: str = "Position IK",
    weight_editor: DiagonalWeightEditor | None = None,
    *, bvh_path=None, uniform_root=None, uniform_rig=None, surface_height_threshold=0.040,
) -> None:
    """Open the comparison window for an already configured playback session."""
    try:
        create_comparison_viewer(playback, title, weight_editor, bvh_path=bvh_path,
                                 uniform_root=uniform_root, uniform_rig=uniform_rig,
                                 surface_height_threshold=surface_height_threshold).run()
    finally:
        if weight_editor is not None:
            weight_editor.close()
