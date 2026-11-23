"""Playlist navigation and shared display controls without an OpenGL window."""

from concurrent.futures import Future
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import visualize_canonicalized_bvh as viewer


class PlaylistTests(unittest.TestCase):
    def setUp(self):
        self.motion = Mock(fps=30)
        self.motion.num_frames.return_value = 60
        self.camera_patch = patch.object(viewer, "create_initial_camera", return_value=Mock())
        self.camera = self.camera_patch.start().return_value
        self.addCleanup(self.camera_patch.stop)

    def create(self, paths=None, **kwargs):
        kwargs.setdefault("show_mesh", False)
        instance = viewer.create_canonicalized_bvh_viewer(
            self.motion, 1, paths or [Path("230123/first.bvh"), Path("230123/second.bvh")], **kwargs)
        self.addCleanup(instance.shutdown)
        return instance

    def draw_controls(self, instance):
        with patch.object(instance, "rectangle"), patch.object(instance, "text"):
            instance.draw_controls()

    def finish_loading(self, instance):
        instance.loading.result(timeout=5)
        instance.poll_loading()

    def test_button_drag_does_not_move_camera_and_scene_drag_does(self):
        from OpenGL import GLUT
        with patch.object(viewer, "load_canonicalized_bvh", return_value=self.motion) as load:
            instance = self.create()
            self.draw_controls(instance)
            instance.mouse_func(GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_DOWN, 220, 30)
            instance.motion_func(300, 200)
            instance.mouse_func(GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_UP, 300, 200)
            self.finish_loading(instance)
            self.assertEqual(instance.motion_index, 1)
            self.assertIsNone(instance.mouse_last_pos)
            self.camera.rotate.assert_not_called()
            self.draw_controls(instance)
            instance.mouse_func(GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_DOWN, 220, 30)
            instance.mouse_func(GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_UP, 220, 30)
            self.assertEqual(load.call_count, 1)
            instance.mouse_func(GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_DOWN, 400, 300)
            instance.motion_func(410, 310)
            self.camera.rotate.assert_called_once()
            instance.mouse_func(GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_UP, 220, 30)
            self.assertIsNone(instance.mouse_last_pos)
            instance.mouse_func(GLUT.GLUT_RIGHT_BUTTON, GLUT.GLUT_DOWN, 400, 300)
            instance.motion_func(410, 310)
            self.camera.translate.assert_called_once()
            instance.mouse_func(GLUT.GLUT_RIGHT_BUTTON, GLUT.GLUT_UP, 410, 310)

    def test_relative_paths_comments_and_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.bvh").touch()
            playlist = root / "list.txt"
            playlist.write_text("\ufeff# comment\n\none.bvh\n", encoding="utf-8")
            args = viewer.parse_viewer_arguments(["--bvh-list", str(playlist)])
            self.assertEqual(viewer.read_bvh_paths(args), [root / "one.bvh"])
            playlist.write_text("# empty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                viewer.read_bvh_paths(args)
            playlist.write_text("missing.bvh", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                viewer.read_bvh_paths(args)

    def test_navigation_autoplays_and_single_clip_disables_both_buttons(self):
        with patch.object(viewer, "load_canonicalized_bvh", return_value=self.motion) as load:
            instance = self.create()
            instance.play_time, instance.paused = 5, True
            instance.keyboard_callback(b"n")
            self.finish_loading(instance)
            self.assertEqual(instance.motion_index, 1)
            self.assertIs(instance.motion, self.motion)
            self.assertEqual(instance.play_time, 0)
            self.assertFalse(instance.paused)
            instance.keyboard_callback(b"n")
            self.assertEqual(load.call_count, 1)
            single = self.create([Path("230123/one.bvh")])
            self.draw_controls(single)
            buttons = {b[4]: b[5] for b in single.buttons}
            self.assertFalse(buttons["previous"])
            self.assertFalse(buttons["next"])
            single.keyboard_callback(b"n")
            single.keyboard_callback(b"p")
            self.assertEqual(single.motion_index, 0)

    def test_independent_visibility_persists_and_mesh_tracks_selected_clip(self):
        with patch.object(viewer, "load_canonicalized_bvh", return_value=self.motion), \
             patch.object(viewer, "load_aligned_uniform_mesh", return_value=Mock()) as mesh_load:
            instance = self.create()
            self.draw_controls(instance)
            self.assertTrue({"mesh", "skeleton"}.issubset(b[4] for b in instance.buttons))
            instance.keyboard_callback(b"s")
            self.assertFalse(instance.show_skeleton)
            self.assertFalse(instance.show_mesh)
            instance.keyboard_callback(b"m")
            instance.mesh_loading.result(timeout=5)
            instance.poll_mesh_loading()
            self.assertTrue(instance.show_mesh)
            self.assertFalse(instance.show_skeleton)
            instance.keyboard_callback(b"n")
            self.finish_loading(instance)
            instance.mesh_loading.result(timeout=5)
            instance.poll_mesh_loading()
            self.assertFalse(instance.show_skeleton)
            self.assertTrue(instance.show_mesh)
            self.assertEqual(mesh_load.call_args.args[0], viewer.DEFAULT_UNIFORM_ROOT / "230123/second.bvh")
            instance.keyboard_callback(b"s")
            self.assertTrue(instance.show_skeleton)
            self.assertTrue(instance.show_mesh)

    def test_late_mesh_result_cannot_replace_current_clip(self):
        instance = self.create()
        old_mesh = Future()
        old_mesh.set_running_or_notify_cancel()
        instance.mesh_loading = old_mesh
        with patch.object(viewer, "load_canonicalized_bvh", return_value=self.motion):
            instance.select_motion(1)
            self.finish_loading(instance)
        old_mesh.set_result(Mock())
        instance.poll_mesh_loading()
        self.assertIsNone(instance.mesh)

    def test_mesh_error_is_visible_and_does_not_disable_skeleton(self):
        with patch.object(viewer, "load_aligned_uniform_mesh", side_effect=ValueError("rig mismatch")):
            instance = self.create(show_mesh=True)
            self.assertIsInstance(instance.mesh_loading.exception(timeout=5), ValueError)
            instance.poll_mesh_loading()
            self.assertIn("rig mismatch", instance.mesh_error)
            self.assertTrue(instance.show_skeleton)
            controls = {entry[2]: entry for entry in instance.playback_controls()}
            self.assertFalse(controls["mesh"][3])
            self.assertFalse(controls["mesh"][4])
            instance.keyboard_callback(b"m")
            self.assertTrue(instance.show_mesh)
            self.assertIn("rig mismatch", instance.mesh_error)
            with patch.object(viewer, "load_canonicalized_bvh", return_value=self.motion), \
                 patch.object(viewer, "load_aligned_uniform_mesh", return_value=Mock()):
                instance.select_motion(1)
                self.finish_loading(instance)
                instance.mesh_loading.result(timeout=5)
                instance.poll_mesh_loading()
            self.assertTrue(instance.show_mesh)
            self.assertEqual(instance.mesh_error, "")
            self.assertTrue({entry[2]: entry for entry in instance.playback_controls()}["mesh"][3])

    def test_default_is_visible_and_off_remains_off_after_navigation(self):
        args = viewer.parse_viewer_arguments(["--bvh-file", "clip.bvh"])
        self.assertTrue(args.mesh)
        self.assertFalse(viewer.parse_viewer_arguments(["--bvh-file", "clip.bvh", "--no-mesh"]).mesh)
        with patch.object(viewer, "load_canonicalized_bvh", return_value=self.motion), \
             patch.object(viewer, "load_aligned_uniform_mesh", return_value=Mock()) as load_mesh:
            instance = self.create(show_mesh=True)
            instance.mesh_loading.result(timeout=5)
            instance.poll_mesh_loading()
            instance.keyboard_callback(b"m")
            instance.select_motion(1)
            self.finish_loading(instance)
            self.assertFalse(instance.show_mesh)
            self.assertIsNone(instance.mesh_loading)
            self.assertEqual(load_mesh.call_count, 1)

    def test_event_loop_returns_on_close_without_pending_render_timer(self):
        import motion_view
        from contextlib import ExitStack
        instance = self.create()
        functions = ["glutInit", "glutInitDisplayMode", "glutInitWindowSize",
                     "glutInitWindowPosition", "glutCreateWindow", "glutSetOption",
                     "glutIgnoreKeyRepeat", "glutDisplayFunc", "glutIdleFunc",
                     "glutReshapeFunc", "glutKeyboardFunc", "glutMouseFunc",
                     "glutMotionFunc", "glutMainLoop", "glutTimerFunc", "glutPostRedisplay"]
        with ExitStack() as stack:
            calls = {name: stack.enter_context(patch.object(motion_view.GLUT, name))
                     for name in functions}
            stack.enter_context(patch.object(instance, "_init_GL"))
            stack.enter_context(patch.object(instance, "poll_loading"))
            stack.enter_context(patch.object(motion_view.time, "sleep"))
            instance.time_checker = Mock()
            instance.time_checker.get_time.return_value = 0.01
            # One live idle cycle, then a native window-close event returns the loop.
            calls["glutMainLoop"].side_effect = lambda: calls["glutIdleFunc"].call_args.args[0]()
            instance.run()
            calls["glutPostRedisplay"].assert_called_once()
            calls["glutTimerFunc"].assert_not_called()
            calls["glutSetOption"].assert_called_once_with(
                motion_view.GLUT.GLUT_ACTION_ON_WINDOW_CLOSE,
                motion_view.GLUT.GLUT_ACTION_GLUTMAINLOOP_RETURNS)

    def test_playback_controls_wrap_without_losing_display_toggles(self):
        instance = self.create()
        instance.resize_GL(640, 720)
        self.draw_controls(instance)
        controls = [b for b in instance.buttons if b[4] in ("mesh", "skeleton", "play", "follow")]
        self.assertEqual(len(controls), 4)
        for x, y, width, height, *_ in controls:
            self.assertLessEqual(x + width, 640)
            self.assertLess(y + height, 720)


if __name__ == "__main__":
    unittest.main()
