"""Check draft isolation, validation, and in-place application using real Tk widgets."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from weight_editor import DiagonalWeightEditor
from util import get_joint_world_positions
from fairmotion.core.motion import Joint, Pose, Skeleton


class WeightEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tkinter as tk
        except ImportError as error:
            raise unittest.SkipTest("Tk is unavailable") from error
        try:
            root = tk.Tk()
        except tk.TclError as error:
            raise unittest.SkipTest("A desktop display is required") from error
        root.withdraw()
        root.destroy()

    def setUp(self):
        self.names = ("Hips",) + tuple(f"Joint{i}" for i in range(1, 28))
        self.leaf_indices = (4, 10, 16, 22, 27)
        self.rotation_indices = tuple(i for i in range(28) if i not in self.leaf_indices)
        self.W = np.diag(np.linspace(0.2, 10.0, 72))
        self.original = self.W.copy()
        self.results = []
        skel = Skeleton()
        for index, name in enumerate(self.names):
            offset = np.eye(4)
            offset[:3, 3] = (0.1 * (index % 2), 0.05, 0.02) if index else (0, 0, 0)
            parent_index = index - 2 if index - 1 in self.leaf_indices else index - 1
            skel.add_joint(Joint(name, xform_from_parent_joint=offset), None if index == 0 else self.names[parent_index])
        preview_pose = Pose(skel, np.repeat(np.eye(4)[None], len(self.names), axis=0))
        self.editor = DiagonalWeightEditor(self.W, preview_pose)
        np.testing.assert_allclose(self.editor.skeleton_positions, get_joint_world_positions(preview_pose, self.names))
        self.editor.open(self.results.append)
        self.editor.poll()
        self.editor.window.withdraw()

    def tearDown(self):
        self.editor.close()

    def set_value(self, index: int, text: str):
        entry = self.editor.entries[index]
        entry.delete(0, "end")
        entry.insert(0, text)

    def test_cancel_preserves_live_matrix_and_reopens_current_values(self):
        self.assertEqual(len(self.editor.entries), 72)
        self.assertEqual(self.editor.row_joint_indices, (0,) + self.rotation_indices)
        self.assertEqual(len(self.editor.joint_items), 28)
        for j in self.leaf_indices:
            self.assertNotIn(f"{self.names[j]} rotation", self.editor.row_names)
        self.set_value(0, "200")
        np.testing.assert_array_equal(self.W, self.original)
        window = self.editor.window
        self.editor.open(self.results.append)
        self.assertIs(self.editor.window, window)
        self.editor.cancel_button.invoke()
        self.assertFalse(self.editor.is_open)
        np.testing.assert_array_equal(self.W, self.original)
        self.assertEqual(self.results, [False])
        self.editor.open(self.results.append)
        self.editor.window.withdraw()
        self.assertEqual(float(self.editor.entries[0].get()), self.original[0, 0])

    def test_apply_is_atomic_rejects_invalid_values_and_preserves_matrix_reference(self):
        reference = self.W
        self.set_value(0, "250")
        for text in ("", "abc", "0", "-1", "nan", "inf", "1e309"):
            with self.subTest(text=text):
                self.set_value(71, text)
                self.editor.apply_button.invoke()
                self.assertTrue(self.editor.is_open)
                self.assertIn("Joint26 rotation / z", self.editor.error_text.get())
                np.testing.assert_array_equal(self.W, self.original)
                self.assertFalse(self.results)
        self.set_value(71, "2.5e-2")
        self.editor.apply_button.invoke()
        self.assertIs(self.editor.W, reference)
        self.assertFalse(self.editor.is_open)
        expected = self.original.copy()
        expected[0, 0], expected[71, 71] = 250, 0.025
        np.testing.assert_array_equal(self.W, expected)
        self.assertEqual(self.results, [True])

    def test_unchanged_apply_keeps_full_float_precision(self):
        self.editor.apply_button.invoke()
        np.testing.assert_array_equal(self.W, self.original)

    def test_window_close_cancels_and_poll_handles_destroyed_window(self):
        self.set_value(4, "99")
        root = self.editor.window
        root.after(0, lambda: root.tk.call(root.protocol("WM_DELETE_WINDOW")))
        self.editor.poll()
        self.editor.poll()
        self.assertFalse(self.editor.is_open)
        self.assertEqual(self.results, [False])
        np.testing.assert_array_equal(self.W, self.original)

    def test_hover_links_rows_and_xyz_entries_to_the_same_joint(self):
        from weight_editor import HOVER_COLOR, JOINT_COLOR

        for row in (0, 1, 6, 12, 23):
            expected_joint = ((0,) + self.rotation_indices)[row]
            for widget in [self.editor.row_labels[row], *self.editor.entries[3 * row:3 * row + 3]]:
                widget.event_generate("<Enter>")
                self.assertEqual(self.editor.hovered_joint, expected_joint)
                self.assertEqual(self.editor.hover_name.get(), f"Joint: {self.names[expected_joint]}")
                item = self.editor.joint_items[expected_joint]
                self.assertEqual(self.editor.preview_canvas.itemcget(item, "fill"), HOVER_COLOR)
                widget.event_generate("<Leave>")
                self.assertIsNone(self.editor.hovered_joint)
                item = self.editor.joint_items[expected_joint]
                self.assertEqual(self.editor.preview_canvas.itemcget(item, "fill"), JOINT_COLOR)
        np.testing.assert_array_equal(self.W, self.original)

    def test_preview_hover_and_rotation_preserve_skeleton_positions(self):
        from types import SimpleNamespace

        original_positions = self.editor.skeleton_positions.copy()
        x, y = self.editor.preview_points[27]
        self.editor.hover_preview(SimpleNamespace(x=x, y=y))
        self.assertEqual(self.editor.hovered_joint, 27)
        projection = self.editor.preview_points.copy()
        self.editor.begin_preview_drag(SimpleNamespace(x=30, y=30))
        self.editor.rotate_preview(SimpleNamespace(x=100, y=50))
        self.assertFalse(np.allclose(self.editor.preview_points, projection))
        np.testing.assert_array_equal(self.editor.skeleton_positions, original_positions)
        np.testing.assert_array_equal(self.W, self.original)


if __name__ == "__main__":
    unittest.main()
