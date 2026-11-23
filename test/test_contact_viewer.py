"""Shared contact display, detector parity and asynchronous playlist state."""

from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from contact import JointGroundContact
from contact.playlist import Entry
from contact.sequence import ContactClip, ContactTrack, detector_track, world_transforms
import contact.viewer as viewer
from test_solver import make_pose
from fairmotion.core.motion import Motion


class ContactSequenceTests(unittest.TestCase):
    def test_fk_and_labels_match_independent_pose_reference_at_actual_fps(self):
        pose = make_pose()
        motion = Motion(skel=pose.skel, fps=60)
        for i in range(4):
            data = pose.data.copy()
            data[0, :3, 3] += [i * .003, 0, 0]
            motion.add_one_frame(data)
        world = world_transforms(motion)
        expected = np.array([[p.get_transform(j.name, local=False) for j in motion.skel.joints]
                             for p in motion.poses])
        np.testing.assert_allclose(world, expected, atol=1e-12)
        detector = JointGroundContact(height_threshold=2)
        track = detector_track(detector, motion, world)
        reference_points = expected[:, [motion.skel.get_index_joint(n) for n in detector.joints]][..., :3, 3]
        reference_speed = np.zeros(reference_points.shape[:2])
        reference_speed[1:] = np.linalg.norm(np.diff(reference_points, axis=0), axis=-1) * 60
        reference = (reference_points[..., 1] < 2) & (reference_speed < .4)
        np.testing.assert_array_equal(track.contacts, reference)
        np.testing.assert_allclose(track.speeds, reference_speed, atol=1e-12)
        unknown = detector_track(detector, motion, world, first_frame_known=False)
        self.assertFalse(unknown.known[0])
        self.assertFalse(unknown.contacts[0].any())
        self.assertTrue(np.isnan(unknown.speeds[0]).all())
        unknown.set_thresholds(height=3)
        self.assertFalse(unknown.contacts[0].any())


class ContactViewerTests(unittest.TestCase):
    def setUp(self):
        self.motion = Mock(fps=30)
        self.motion.num_frames.return_value = 3
        points = np.array([[[0, .06, 0], [0, .02, 0]],
                           [[0, .06, 0], [.02, .02, 0]],
                           [[0, .06, 0], [.02, .02, 0]]])
        previous = np.concatenate((points[:1], points[:-1]))
        detector = JointGroundContact(height_threshold=.05)
        self.track = ContactTrack("Ankle centers", detector.joints, points,
                                  detector.detect(points, previous, 1 / 30), np.ones(3, dtype=bool),
                                  np.linalg.norm(points - previous, axis=-1) * 30, detector, 1 / 30)
        self.entries = [Entry("first", "date/first.bvh"), Entry("second", "date/second.bvh")]
        camera = patch.object(viewer, "create_contact_camera", return_value=Mock())
        camera.start()
        self.addCleanup(camera.stop)

    def create(self, *, entries=None, tracks=None, show_mesh=False):
        def load(entry):
            return ContactClip(self.motion, np.zeros((3, 2, 3)), tuple(deepcopy(t) for t in (tracks or (self.track,))))
        instance = viewer.ContactViewer(entries or self.entries, load, show_mesh=show_mesh)
        self.addCleanup(instance.shutdown)
        self.finish(instance)
        return instance

    @staticmethod
    def finish(instance):
        instance.loading.result(timeout=5)
        instance.poll_loading()

    @staticmethod
    def controls(instance):
        with patch.object(instance, "rectangle"), patch.object(instance, "text") as text:
            instance.draw_controls()
        return {b[4]: b[5] for b in instance.buttons}, [call.args[0] for call in text.call_args_list]

    def test_navigation_toggles_and_thresholds_persist_without_mutating_config(self):
        instance = self.create()
        instance.keyboard_callback(b"s")
        instance.set_panel_threshold(0, height=.07, speed=.7)
        self.assertTrue(instance.clip.tracks[0].contacts.all())
        self.assertEqual(self.track.detector.height_threshold, .05)
        instance.paused, instance.play_time = True, 2
        instance.keyboard_callback(b"n")
        self.finish(instance)
        self.assertEqual(instance.index, 1)
        self.assertFalse(instance.show_skeleton)
        self.assertFalse(instance.show_mesh)
        self.assertFalse(instance.paused)
        self.assertEqual(instance.play_time, 0)
        self.assertEqual(instance.clip.tracks[0].detector.height_threshold, .07)
        self.assertEqual(instance.clip.tracks[0].detector.speed_threshold, .7)
        self.assertTrue(instance.clip.tracks[0].contacts.all())
        instance.keyboard_callback(b"n")
        self.assertEqual(instance.index, 1)

    def test_single_motion_disables_navigation_and_reports_actual_contact(self):
        instance = self.create(entries=self.entries[:1])
        buttons, labels = self.controls(instance)
        self.assertFalse(buttons["previous"])
        self.assertFalse(buttons["next"])
        self.assertIn("LeftFoot: no contact", labels)
        self.assertIn("RightFoot: CONTACT", labels)
        self.assertIn("h +60.0 mm | speed 0.000 m/s", labels)
        instance.keyboard_callback(b"]")
        _, labels = self.controls(instance)
        self.assertIn("RightFoot: no contact", labels)
        self.assertIn("h +20.0 mm | speed 0.600 m/s", labels)

    def test_panels_adjust_independently_and_predictions_are_read_only(self):
        prediction = ContactTrack("Model", self.track.labels, self.track.points,
                                  np.ones((3, 2), dtype=bool), np.ones(3, dtype=bool))
        instance = self.create(tracks=(self.track, self.track, prediction))
        snapshot = instance.clip.tracks[1].contacts.copy()
        instance.set_panel_threshold(0, height=.07, speed=.7)
        np.testing.assert_array_equal(instance.clip.tracks[1].contacts, snapshot)
        self.assertTrue(instance.clip.tracks[2].contacts.all())
        buttons, _ = self.controls(instance)
        self.assertNotIn("threshold:2:height:up", buttons)
        instance.set_contact_threshold(.04)
        self.assertEqual(instance.contact_threshold, .04)
        self.assertEqual(instance.clip.tracks[0].detector.height_threshold, .04)
        self.assertEqual(instance.clip.tracks[1].detector.height_threshold, .05)

    def test_missing_mesh_keeps_contact_and_skeleton_available(self):
        with patch.object(viewer, "load_aligned_uniform_mesh", side_effect=FileNotFoundError("missing mesh")):
            instance = self.create(show_mesh=True)
            with self.assertRaises(FileNotFoundError):
                instance.mesh_loading.result(timeout=5)
            instance.poll_mesh_loading()
            buttons, labels = self.controls(instance)
            self.assertFalse(buttons["mesh"])
            self.assertTrue(buttons["skeleton"])
            self.assertIn("RightFoot: CONTACT", labels)
            self.assertIn("missing mesh", instance.mesh_error)
            instance.act("mesh")
            self.assertTrue(instance.show_mesh)

    def test_mesh_band_uses_each_panel_height_including_paused_edits(self):
        instance = self.create(tracks=(self.track, self.track))
        vertices = np.array([[0, .06, 0], [1, .06, 0], [0, .06, 1]], dtype=np.float32)
        instance.mesh = Mock(triangles=np.array([[0, 1, 2]], dtype=np.uint32))
        instance.mesh.frame.return_value = Mock(vertices=vertices)
        instance.show_mesh, instance.show_skeleton, instance.paused = True, False, True
        instance.set_panel_threshold(0, height=.10)
        instance.set_panel_threshold(1, height=.04)
        with patch("motion_view.GL") as gl, patch.object(viewer, "GL"), \
             patch.object(viewer.gl_render, "render_point"):
            def colors(panel):
                instance.draw_scene(panel, 0)
                return gl.glColorPointer.call_args.args[3].copy()
            left, right = colors(0), colors(1)
            self.assertTrue((left[:, 0] > right[:, 0]).all())  # yellow versus blue
            self.assertTrue((left[:, 2] < right[:, 2]).all())
            instance.set_panel_threshold(1, height=.08)
            np.testing.assert_array_equal(colors(1), left)
            np.testing.assert_array_equal(colors(0), left)
            instance.set_panel_threshold(0, height=.02)
            np.testing.assert_array_equal(colors(0), right)
            np.testing.assert_array_equal(colors(1), left)
            # A single-panel caller still uses the common display threshold.
            instance.draw_mesh(0)
            np.testing.assert_array_equal(gl.glColorPointer.call_args.args[3], right)
        instance.clear_mesh()
        self.assertEqual(instance.mesh_colors, {})

    def test_running_old_load_is_ignored_after_next(self):
        instance = self.create()
        pending = Future()
        pending.set_running_or_notify_cancel()
        instance.loading = pending
        instance.act("next")
        self.finish(instance)
        selected = instance.clip
        pending.set_exception(ValueError("old clip failed"))
        instance.poll_loading()
        self.assertIs(instance.clip, selected)
        self.assertEqual(instance.status, "")


if __name__ == "__main__":
    unittest.main()
