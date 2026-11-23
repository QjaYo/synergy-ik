"""Shared contact display: synchronized panels, labels, markers and playback."""

from functools import partial
from pathlib import Path
import random
import sys
import textwrap

import numpy as np

from .playlist import DEFAULT_UNIFORM_ROOT, DEFAULT_UNIFORM_RIG, original_key
from motion_view import MotionView
from viewer import create_initial_camera
from uniform_mesh import load_aligned_uniform_mesh, surface_colors
from OpenGL import GL, GLU, GLUT
from fairmotion.viz import gl_render


def create_contact_camera(clip, aspect):
    """Fit the clip's full pose range while following horizontal root motion."""
    camera = create_initial_camera(clip.motion)
    root_ground = clip.positions[:, :1].copy()
    root_ground[..., 1] = 0
    relative = clip.positions - root_ground
    center = (relative.min(axis=(0, 1)) + relative.max(axis=(0, 1))) / 2
    radius = float(np.linalg.norm(relative - center, axis=-1).max())
    vertical_half = np.deg2rad(camera.fov / 2)
    half_angle = min(vertical_half, np.arctan(np.tan(vertical_half) * aspect))
    direction = camera.pos - camera.origin
    direction /= np.linalg.norm(direction)
    camera.origin = center + root_ground[0, 0]
    camera.pos = camera.origin + 1.15 * max(radius, .5) / np.sin(half_angle) * direction
    return camera


class ContactViewer(MotionView):
    """Display loaded ContactClips; delegate threshold changes to their tracks."""

    def __init__(self, entries, loader, descriptions=None, *, speed=1, show_mesh=True,
                 uniform_root=DEFAULT_UNIFORM_ROOT, uniform_rig=DEFAULT_UNIFORM_RIG,
                 title=b"SynergyIK | Contact viewer", height_band=0.040, footer=""):
        if not entries:
            raise ValueError("Contact viewer needs at least one motion")
        super().__init__(title, speed, show_mesh, surface_colors)
        self.window_size = (1400, 900)
        self.entries, self.loader = list(entries), loader
        self.descriptions = descriptions or {}
        self.uniform_root, self.uniform_rig = Path(uniform_root), Path(uniform_rig)
        self.footer = footer
        self.index = 0
        self.clip = None
        self.loading = None
        self.thresholds = {}
        super().set_contact_threshold(height_band)
        self.select(0)

    def select(self, index):
        if not 0 <= index < len(self.entries):
            return
        if self.loading is not None:
            self.loading.cancel()
        self.index = index
        self.motion = self.clip = None
        self.clear_mesh()
        self.last_root = None
        self.play_time = 0.0
        self.paused = False
        entry = self.entries[index]
        self.mesh_loader = partial(load_aligned_uniform_mesh, self.uniform_root / entry.key, self.uniform_rig)
        self.status = "Loading motion and contact results..."
        self.loading = self.executor.submit(self.loader, entry)

    def poll_loading(self):
        if self.loading is None or not self.loading.done():
            return
        future, self.loading = self.loading, None
        try:
            clip = future.result()
            if not clip.tracks:
                raise ValueError("Contact clip has no display tracks")
            for index, track in enumerate(clip.tracks):
                if index in self.thresholds:
                    track.set_thresholds(*self.thresholds[index])
            self.clip, self.motion = clip, clip.motion
            scene_height = max(1, self.window_size[1] - self.header_height() - self.playback_bar_height())
            self.cam_cur = create_contact_camera(clip, self.panel_bounds()[0][1] / scene_height)
            self.status = ""
            self.start_mesh_loading()
            print(f"READY {self.entries[self.index].key}", flush=True)
        except Exception as error:
            self.motion = self.clip = None
            self.status = f"Contact unavailable: {type(error).__name__}: {error}"
            print(self.status, file=sys.stderr, flush=True)

    def set_panel_threshold(self, index, *, height=None, speed=None):
        track = self.clip.tracks[index]
        track.set_thresholds(height, speed)
        self.thresholds[index] = (track.detector.height_threshold, track.detector.speed_threshold)
        self.mesh_colors.clear()
        if index == 0:
            super().set_contact_threshold(track.detector.height_threshold)

    def set_contact_threshold(self, value):
        # Keep the existing comparison's Band control linked to the first detector.
        if self.clip is not None and self.clip.tracks[0].detector is not None:
            self.set_panel_threshold(0, height=max(.005, round(value, 6)))
        else:
            super().set_contact_threshold(value)

    def act(self, action):
        if action in ("previous", "next"):
            self.select(self.index + (-1 if action == "previous" else 1))
        elif action == "random":
            if len(self.entries) > 1:
                offset = random.randrange(1, len(self.entries))
                self.select((self.index + offset) % len(self.entries))
        elif action.startswith("threshold:"):
            _, panel, kind, direction = action.split(":")
            index = int(panel)
            detector = self.clip.tracks[index].detector
            step = 1 if direction == "up" else -1
            if kind == "height":
                self.set_panel_threshold(index, height=max(.005, round(detector.height_threshold + step * .005, 6)))
            else:
                self.set_panel_threshold(index, speed=max(.05, round(detector.speed_threshold + step * .05, 6)))
        else:
            self.handle_playback_action(action)

    def keyboard_callback(self, key):
        navigation = {b"p": "previous", b"n": "next", b"g": "random"}
        if key.lower() in navigation:
            self.act(navigation[key.lower()])
            return True
        return super().keyboard_callback(key)

    def panel_bounds(self):
        count = len(self.clip.tracks) if self.clip is not None else 1
        width = self.window_size[0]
        edges = [round(i * width / count) for i in range(count + 1)]
        return [(edges[i], edges[i + 1] - edges[i]) for i in range(count)]

    def header_height(self):
        if self.clip is None:
            return 283
        return max(self.panel_layout(width, len(track.labels))[2]
                   for (_, width), track in zip(self.panel_bounds(), self.clip.tracks))

    @staticmethod
    def panel_layout(width, channels):
        columns = 2 if width >= 560 else 1
        card_top = 207 if width >= 450 else 245
        rows = (channels + columns - 1) // columns
        return columns, card_top, card_top + rows * 45 + 31

    def draw_panel_controls(self, index, track, left, width):
        self.text(track.title, left + 14, 136, large=True)
        if track.detector is not None:
            detector = track.detector
            self.text("CONTACT = height < H AND speed < limit", left + 14, 157)
            controls = [("-", 32, "height", "down", detector.height_threshold > .005),
                        (f"H {detector.height_threshold * 1000:g} mm", 90, "height", "label", False),
                        ("+", 32, "height", "up", True),
                        ("-", 32, "speed", "down", detector.speed_threshold > .05),
                        (f"Speed {detector.speed_threshold:g}", 110, "speed", "label", False),
                        ("+", 32, "speed", "up", True)]
            x, y = left + 14, 168
            for label, button_width, kind, direction, enabled in controls:
                if kind == "speed" and direction == "down" and width < 450:
                    x, y = left + 14, y + 38
                self.button(label, x, y, button_width, f"threshold:{index}:{kind}:{direction}", enabled)
                x += button_width + 6
        else:
            self.text(track.note[:max(1, (width - 28) // 7)], left + 14, 157)
        frame = self.frame_index()
        columns, card_top, _ = self.panel_layout(width, len(track.labels))
        for channel, label in enumerate(track.labels):
            row, column = divmod(channel, columns)
            x, y = left + 14 + column * (width // columns), card_top + row * 45
            known = bool(track.known[frame])
            contact = known and bool(track.contacts[frame, channel])
            color = (0.73, 0.90, 0.75, 1) if contact else (0.89, 0.91, 0.93, 1)
            self.rectangle(x, y, width // columns - 25, 40, color)
            value = "CONTACT" if contact else "no contact" if known else "no history"
            self.text(f"{label}: {value}", x + 8, y + 16)
            if track.detector is not None:
                height = (track.points[frame, channel, 1] - track.detector.ground_y) * 1000
                speed = "--" if not known else f"{track.speeds[frame, channel]:.3f}"
                self.text(f"h {height:+.1f} mm | speed {speed} m/s", x + 8, y + 32)

    def draw_controls(self):
        width, height = self.window_size
        self.buttons = []
        self.rectangle(0, 0, width, self.header_height(), (0.97, 0.98, 0.99, 1))
        self.button("Previous [P]", 12, 10, 115, "previous", self.index > 0)
        self.button("Next [N]", 137, 10, 100, "next", self.index + 1 < len(self.entries))
        self.button("Random [G]", 247, 10, 115, "random", len(self.entries) > 1)
        self.text(f"Motion {self.index + 1} / {len(self.entries)}", 380, 31, large=True)
        self.button("Close [Esc]", width - 112, 10, 100, "close")
        entry = self.entries[self.index]
        description = self.descriptions.get(original_key(entry.key))
        title = description.text if description else entry.group.replace("_", " ")
        for i, line in enumerate(textwrap.wrap(title, max(40, (width - 24) // 10))[:2]):
            self.text(line, 12, 66 + i * 20, large=True)
        self.text(textwrap.shorten(entry.key, max(30, (width - 24) // 7), placeholder="..."), 12, 108)
        if self.clip is not None:
            for index, ((left, panel_width), track) in enumerate(zip(self.panel_bounds(), self.clip.tracks)):
                self.draw_panel_controls(index, track, left, panel_width)
                if index:
                    self.rectangle(left - 1, 118, 2, height - 118 - self.playback_bar_height(), (0.65, .7, .75, 1))
        message = self.mesh_error or self.status
        if not message and self.show_mesh and self.mesh is None:
            message = "Loading original Uniform surface..."
        message = message or "Green markers = CONTACT | Mesh red/yellow = height only | Speed is measured in world space."
        self.text(message[:max(40, (width - 24) // 7)], 12, self.header_height() - 12)
        self.draw_playback_controls(self.footer or (self.clip.tracks[0].note if self.clip else self.status))

    def draw_scene(self, panel, frame):
        if self.clip is None:
            return
        track = self.clip.tracks[panel]
        if self.show_mesh and self.mesh is not None:
            try:
                threshold = track.detector.height_threshold if track.detector is not None else self.contact_threshold
                self.draw_mesh(frame, height_threshold=threshold)
            except Exception as error:
                self.mesh_error = f"Mesh rendering failed: {error}"
                self.mesh = None
        positions = self.clip.positions[frame]
        if self.show_skeleton:
            if self.show_mesh:
                GL.glDisable(GL.GL_DEPTH_TEST)
            for j, joint in enumerate(self.motion.skel.joints):
                gl_render.render_point(positions[j], radius=.01, color=(.43, .25, .64, 1))
                if joint.parent_joint is not None:
                    parent = self.motion.skel.get_index_joint(joint.parent_joint)
                    gl_render.render_line(positions[parent], positions[j], color=(.66, .34, .48, 1), line_width=3)
        GL.glDisable(GL.GL_DEPTH_TEST)
        for point, flag in zip(track.points[frame], track.contacts[frame]):
            color = (.05, .8, .25, 1) if track.known[frame] and flag else (.35, .40, .46, 1)
            gl_render.render_point(point, radius=.025, color=color)

    def draw_GL(self, swap_buffer=True):
        width, height = self.window_size
        bottom = self.playback_bar_height()
        scene_height = max(1, height - self.header_height() - bottom)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        frame = self.frame_index() if self.motion is not None else 0
        if self.clip is not None:
            root = self.clip.positions[frame, 0].copy()
            root[1] = 0
            if self.follow_root and self.last_root is not None:
                delta = root - self.last_root
                self.cam_cur.pos += delta
                self.cam_cur.origin += delta
            self.last_root = root
        for panel, (left, panel_width) in enumerate(self.panel_bounds()):
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glViewport(left, bottom, panel_width, scene_height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GLU.gluPerspective(self.cam_cur.fov, max(1, panel_width) / scene_height, .01, 200)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GLU.gluLookAt(*self.cam_cur.pos, *self.cam_cur.origin, *self.cam_cur.vup)
            gl_render.render_ground(size=[100, 100], color=[.83, .85, .88, 1], axis="y", origin=False)
            if self.motion is not None:
                self.draw_scene(panel, frame)
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
