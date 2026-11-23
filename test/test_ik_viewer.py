"""IK viewer controls remain independent from learned feature/contact settings."""

from functools import partial
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from test_solver import make_pose
from fairmotion.core.motion import Motion
from playback import IKPlayback, DEFAULT_END_EFFECTORS
from solver import solve_ik_step
from viewer import create_comparison_viewer


class IKViewerTests(unittest.TestCase):
    def create(self, **kwargs):
        pose = make_pose()
        motion = Motion(skel=pose.skel, fps=120)
        motion.add_one_frame(pose.data)
        playback = IKPlayback(motion, partial(solve_ik_step, W=np.eye(72)), panel_label="LEARNED",
                              fixed_ee_names=DEFAULT_END_EFFECTORS)
        viewer = create_comparison_viewer(playback, **kwargs)
        return viewer, playback

    def test_missing_mesh_and_fixed_ee_controls(self):
        viewer, _ = self.create()
        self.assertIn("Mesh unavailable", viewer.surface_error)
        self.assertTrue(viewer.show_skeleton)
        with patch.object(viewer, "rectangle"), patch.object(viewer, "text"):
            viewer.draw_controls()
        actions = {row[-1] for row in viewer.buttons}
        self.assertNotIn("mesh", actions)
        self.assertNotIn("select_all", actions)
        self.assertNotIn("LeftHandEnd", actions)
        self.assertIn("skeleton", actions)

    def test_visibility_survives_reset_and_display_band_is_only_ui(self):
        with patch("surface.IKSurface", return_value=Mock()):
            viewer, playback = self.create(bvh_path=Path("231019/example.bvh"))
        self.assertTrue(viewer.show_mesh)
        with patch("OpenGL.GLUT.glutPostRedisplay"):
            viewer.perform_action("mesh")
            viewer.perform_action("reset")
            viewer.perform_action("band_up")
        self.assertFalse(viewer.show_mesh)
        self.assertAlmostEqual(viewer.contact_threshold, .045)
        self.assertEqual(playback.fixed_ee_names, DEFAULT_END_EFFECTORS)
        with patch("OpenGL.GLUT.glutLeaveMainLoop") as leave:
            self.assertTrue(viewer.keyboard_callback(b"\x1b"))
            leave.assert_called_once()


if __name__ == "__main__":
    unittest.main()
