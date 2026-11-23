"""Shared playback, camera, skeleton, and surface UI for motion viewers."""

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from util import get_joint_world_positions
from OpenGL import GL, GLU, GLUT
from fairmotion.viz import gl_render, glut_viewer
from viewer_window import run_window


class MotionView(glut_viewer.Viewer):
    """Subclasses provide motion loading, a header, and application actions."""

    def __init__(self, title, speed, show_mesh, mesh_colorizer):
        super().__init__(title=title, size=(1280, 820))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bvh-loader")
        self.motion = None
        self.mesh_loader = None
        self.mesh_colorizer = mesh_colorizer
        self.contact_threshold = 0.040
        self.mesh_loading = None
        self.mesh = None
        self.show_mesh = show_mesh
        self.show_skeleton = True
        self.mesh_error = ""
        self.mesh_frame = -1
        self.mesh_colors = {}
        self.buttons = []
        self.play_time = 0.0
        self.speed = speed
        self.paused = False
        self.follow_root = True
        self.last_root = None
        self.status = ""
        self.dragging_scene = False

    def clear_mesh(self) -> None:
        if self.mesh_loading is not None:
            self.mesh_loading.cancel()
        self.mesh_loading = None
        self.mesh = None
        self.mesh_error = ""
        self.mesh_frame = -1
        self.mesh_colors = {}

    def start_mesh_loading(self) -> None:
        if self.show_mesh and self.motion is not None and self.mesh is None and self.mesh_loading is None and not self.mesh_error:
            self.mesh_loading = self.executor.submit(
                self.mesh_loader, self.motion)

    def poll_mesh_loading(self) -> None:
        if self.mesh_loading is None or not self.mesh_loading.done():
            return
        future, self.mesh_loading = self.mesh_loading, None
        try:
            self.mesh = future.result()
        except Exception as error:
            self.mesh_error = f"Mesh unavailable: {type(error).__name__}: {error}"

    def draw_mesh(self, frame: int, *, height_threshold: float | None = None) -> None:
        result = self.mesh.frame(frame)
        threshold = self.contact_threshold if height_threshold is None else height_threshold
        if frame != self.mesh_frame:
            self.mesh_colors.clear()
            self.mesh_frame = frame
        if threshold not in self.mesh_colors:
            self.mesh_colors[threshold] = self.mesh_colorizer(result.vertices, self.mesh.triangles, threshold)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        try:
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, result.vertices)
            GL.glColorPointer(3, GL.GL_FLOAT, 0, self.mesh_colors[threshold])
            GL.glDrawElements(GL.GL_TRIANGLES, self.mesh.triangles.size, GL.GL_UNSIGNED_INT, self.mesh.triangles)
        finally:
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    def resize_GL(self, width: int, height: int) -> None:
        self.window_size = (max(width, 1), max(height, 1))

    def frame_index(self) -> int:
        return int(self.play_time * self.motion.fps + 1e-6) % self.motion.num_frames()

    def draw_GL(self, swap_buffer: bool = True) -> None:
        width, height = self.window_size
        top, bottom = self.header_height(), self.playback_bar_height()
        scene_height = max(1, height - top - bottom)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glViewport(0, bottom, width, scene_height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(self.cam_cur.fov, max(1, width) / scene_height, 0.01, 200.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        if self.motion is not None:
            pose = self.motion.get_pose_by_frame(self.frame_index())
            positions = get_joint_world_positions(pose, [j.name for j in self.motion.skel.joints])
            root = positions[0].copy()
            root[1] = 0.0
            if self.follow_root and self.last_root is not None:
                delta = root - self.last_root
                self.cam_cur.pos += delta
                self.cam_cur.origin += delta
            self.last_root = root
        GLU.gluLookAt(*self.cam_cur.pos, *self.cam_cur.origin, *self.cam_cur.vup)
        gl_render.render_ground(size=[100, 100], color=[0.83, 0.85, 0.88, 1], axis="y", origin=False)
        if self.show_mesh and self.mesh is not None and self.motion is not None:
            try:
                self.draw_mesh(self.frame_index())
            except Exception as error:
                self.mesh_error = f"Mesh rendering failed: {error}"
                self.mesh = None
        if self.motion is not None and self.show_skeleton:
            # Overlay the canonical skeleton, including joints inside the skin.
            if self.show_mesh:
                GL.glDisable(GL.GL_DEPTH_TEST)
            for index, joint in enumerate(self.motion.skel.joints):
                gl_render.render_point(positions[index], radius=0.015, color=(0.2, 0.65, 0.25, 1))
                if joint.parent_joint is not None:
                    parent = self.motion.skel.get_index_joint(joint.parent_joint)
                    gl_render.render_line(positions[parent], positions[index],
                                          color=(0.5, 0.29, 0.15, 1), line_width=5.0)
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

    def rectangle(self, x: float, y: float, width: float, height: float, color: tuple) -> None:
        GL.glColor4f(*color)
        GL.glBegin(GL.GL_QUADS)
        for point in ((x, y), (x + width, y), (x + width, y + height), (x, y + height)):
            GL.glVertex2f(*point)
        GL.glEnd()

    def text(self, message: str, x: float, y: float, large: bool = False) -> None:
        readable = message.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
        gl_render.render_text(readable.encode("ascii", "replace").decode(), [x, y],
                              font=GLUT.GLUT_BITMAP_HELVETICA_18 if large else GLUT.GLUT_BITMAP_HELVETICA_12,
                              color=(0.12, 0.16, 0.2, 1))

    def button(self, label: str, x: int, y: int, width: int, action: str, enabled: bool = True, selected: bool = False) -> None:
        color = {"include": (0.80, 0.90, 0.81, 1), "exclude": (0.95, 0.82, 0.82, 1),
                 "uncertain": (0.97, 0.89, 0.67, 1),
                 "quality_exclude": (0.87, 0.80, 0.94, 1)}.get(action, (0.85, 0.89, 0.94, 1))
        if selected:
            self.rectangle(x - 2, y - 2, width + 4, 34, (0.12, 0.16, 0.2, 1))
        self.rectangle(x, y, width, 30, color if enabled else (0.91, 0.91, 0.91, 1))
        self.text(("[*] " if selected else "") + label, x + 9, y + 20)
        self.buttons.append((x, y, width, 30, action, enabled))

    def mouse_func(self, button: int, state: int, x: int, y: int) -> None:
        if state == GLUT.GLUT_UP:
            self.dragging_scene = False
            self.pressed_button = self.mouse_last_pos = None
            return
        if button == GLUT.GLUT_LEFT_BUTTON:
            for bx, by, bw, bh, action, enabled in self.buttons:
                if bx <= x < bx + bw and by <= y < by + bh:
                    if enabled:
                        self.act(action)
                    return
            width, height = self.window_size
            if height - 17 <= y <= height - 3 and self.motion is not None:
                fraction = np.clip((x - 12) / max(1, width - 24), 0, 1)
                self.play_time = fraction * (self.motion.num_frames() - 1) / self.motion.fps
                self.paused = True
                return
        if self.header_height() <= y < self.window_size[1] - self.playback_bar_height():
            super().mouse_func(button, state, x, y)
            self.dragging_scene = button in (GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_RIGHT_BUTTON)

    def motion_func(self, x: int, y: int) -> None:
        if self.dragging_scene and self.mouse_last_pos is not None:
            super().motion_func(x, y)

    def idle_callback(self) -> None:
        elapsed = self.time_checker.get_time(restart=True)
        self.poll_loading()
        self.poll_mesh_loading()
        if not self.paused and self.motion is not None:
            self.play_time += min(elapsed, 0.1) * self.speed
        time.sleep(0.01)
        GLUT.glutPostRedisplay()

    def run(self) -> None:
        run_window(self)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def playback_controls(self):
        ready = self.motion is not None
        return [
            ("Play/Pause [Space]", 155, "play", ready, False),
            ("Restart [R]", 105, "restart", ready, False),
            ("- speed", 70, "slower", True, False),
            ("+ speed", 70, "faster", True, False),
            ("Follow root [F]", 125, "follow", True, self.follow_root),
            ("View Skeleton [S]", 160, "skeleton", True, self.show_skeleton),
            ("View Mesh [M]", 145, "mesh", not self.mesh_error, self.show_mesh and not self.mesh_error),
            ("-", 32, "band_down", self.contact_threshold > 0, False),
            (f"Band {self.contact_threshold * 1000:g} mm", 100, "band_label", False, False),
            ("+", 32, "band_up", True, False),
        ]

    def set_contact_threshold(self, value):
        """Change the shared surface band; subclasses may update contact labels."""
        self.contact_threshold = max(0.0, round(value, 6))
        self.mesh_frame = -1

    def playback_layout(self):
        x, row = 12, 0
        layout = []
        for label, width, action, enabled, selected in self.playback_controls():
            if x > 12 and x + width > self.window_size[0] - 12:
                x, row = 12, row + 1
            layout.append((x, row * 38, label, width, action, enabled, selected))
            x += width + 8
        return layout

    def playback_bar_height(self):
        return 100 + self.playback_layout()[-1][1]

    def draw_playback_controls(self, status=None):
        width, height = self.window_size
        top = height - self.playback_bar_height()
        self.rectangle(0, top, width, height - top, (0.97, 0.98, 0.99, 1))
        for x, offset, label, button_width, action, enabled, selected in self.playback_layout():
            self.button(label, x, top + 10 + offset, button_width, action, enabled, selected)
        frame_text = "" if self.motion is None else f" | Frame {self.frame_index() + 1}/{self.motion.num_frames()}"
        self.text(f"{self.speed:g}x | {'PAUSED' if self.paused else 'PLAYING'}{frame_text}", 12, height - 43)
        message = self.status if status is None else status
        self.text(message[:max(0, (width - 330) // 7)], 318, height - 43)
        self.text("Drag: orbit | Right drag: pan | Wheel: zoom | [ / ]: frame | Click timeline to seek", 12, height - 23)
        progress = 0 if self.motion is None else self.frame_index() / max(1, self.motion.num_frames() - 1)
        self.rectangle(12, height - 14, max(1, width - 24), 7, (0.78, 0.81, 0.85, 1))
        self.rectangle(12, height - 14, max(1, width - 24) * progress, 7, (0.2, 0.5, 0.75, 1))

    def handle_playback_action(self, action):
        if action == "play":
            self.paused = not self.paused
        elif action == "restart":
            self.play_time = 0.0
        elif action == "slower":
            self.speed = max(0.125, self.speed / 2)
        elif action == "faster":
            self.speed = min(8.0, self.speed * 2)
        elif action == "follow":
            self.follow_root = not self.follow_root
        elif action == "mesh":
            if self.mesh_error:
                return True
            self.show_mesh = not self.show_mesh
            if self.show_mesh:
                self.start_mesh_loading()
        elif action == "skeleton":
            self.show_skeleton = not self.show_skeleton
        elif action in ("band_down", "band_up"):
            self.set_contact_threshold(self.contact_threshold + (0.005 if action == "band_up" else -0.005))
        elif action == "close":
            GLUT.glutLeaveMainLoop()
        else:
            return False
        return True

    def keyboard_callback(self, key):
        actions = {b" ": "play", b"r": "restart", b"f": "follow", b"m": "mesh",
                   b"s": "skeleton", b"-": "slower", b"+": "faster", b"=": "faster", b"\x1b": "close"}
        if key.lower() in actions:
            self.act(actions[key.lower()])
        elif key in (b"[", b"]") and self.motion is not None:
            self.paused = True
            frame = (self.frame_index() + (1 if key == b"]" else -1)) % self.motion.num_frames()
            self.play_time = (frame + 1e-6) / self.motion.fps
        return True
