"""Check grouping boundaries, reset isolation and representative decisions."""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess/motion_review"))
from motion_review_data import MotionDescription, MotionReviewStore, reset_review_catalog
from review_groups import behavior_family, build_review_groups


class GroupingTests(unittest.TestCase):
    def test_locomotion_directions_and_phases_merge_but_actions_remain_distinct(self):
        names = ["baby_full_diaper_walk_ff_180_start_r", "baby_full_diaper_walk_ff_270_loop_r",
                 "baby_full_diaper_walk_ff_360_stop_r"]
        self.assertEqual({behavior_family(n) for n in names}, {"baby_full_diaper_walk"})
        self.assertNotEqual(behavior_family(names[0]), behavior_family(names[0].replace("walk", "jog")))
        self.assertNotEqual(behavior_family(names[0]), behavior_family(names[0].replace("_walk", "_turn_walk")))
        self.assertNotEqual(behavior_family("kick_stop_walk_ff_180_r"), behavior_family("kick_walk_ff_180_r"))
        self.assertEqual(behavior_family("neutral_walk_backward"), "walk")
        self.assertEqual(behavior_family("walk_forward_stop"), "walk")
        self.assertEqual(behavior_family("dance_180_v002"), "dance_180_v002")

    def test_transfer_locations_merge_but_weight_size_and_hand_use_do_not(self):
        first = "big_heavy_one_hand_front_high_to_behind_low_r"
        second = "big_heavy_one_hand_right_side_medium_to_front_high_r"
        self.assertEqual(behavior_family(first), behavior_family(second))
        for token, replacement in [("heavy", "light"), ("big", "small"), ("one_hand", "two_hands")]:
            self.assertNotEqual(behavior_family(first), behavior_family(first.replace(token, replacement)))
        self.assertNotEqual(behavior_family(first), behavior_family("big_heavy_one_hand_walk_ff_180_r"))

    def test_context_and_missing_metadata_block_unsafe_merge(self):
        types = {"a": "walk_ff_180_r", "b": "walk_ff_270_r", "c": "walk_ff_360_r"}
        data = {"a": MotionDescription("walking forward", "None listed", "standing"),
                "b": MotionDescription("walking sideways", "None listed", "standing"),
                "c": MotionDescription("walking backward using a crutch", "crutch", "standing")}
        groups = build_review_groups(types, data)
        self.assertEqual(groups["a"][0], groups["b"][0])
        self.assertNotEqual(groups["a"][0], groups["c"][0])
        self.assertEqual(build_review_groups(types, {})["a"][0], types["a"])
        data["b"] = MotionDescription("walking while leaning on a wall", "None listed", "standing")
        groups = build_review_groups(types, data)
        self.assertNotEqual(groups["a"][0], groups["b"][0])

    def test_missing_context_name_cannot_collide_with_merged_family(self):
        types = {"unknown": "walk", "a": "walk_ff_180_r", "b": "walk_ff_270_r"}
        data = {p: MotionDescription("walking", "None listed", "standing") for p in ("a", "b")}
        groups = build_review_groups(types, data)
        self.assertEqual(groups["a"][0], groups["b"][0])
        self.assertNotEqual(groups["unknown"][0], groups["a"][0])

    def test_context_varying_within_type_is_not_propagated_to_another_type(self):
        types = {"a1": "walk_ff_180_r", "a2": "walk_ff_180_r", "b": "walk_ff_270_r"}
        data = {"a1": MotionDescription("walking", "None listed", "standing"),
                "a2": MotionDescription("walking with a box", "box", "standing"),
                "b": MotionDescription("walking", "None listed", "standing")}
        groups = build_review_groups(types, data)
        self.assertNotEqual(groups["a1"][0], groups["a2"][0])
        self.assertEqual(groups["a1"][0], groups["b"][0])

    def test_reset_backs_up_old_work_and_rebuilds_without_adopting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motions = root / "bvh"
            motions.mkdir()
            paths = ["baby_walk_ff_180_start_R_001__A001.bvh", "baby_walk_ff_270_loop_R_001__A002.bvh"]
            mirror = paths[0].replace(".bvh", "_M.bvh")
            for path in paths + [mirror]:
                (motions / path).write_text("fixture", encoding="utf-8")
            database = root / "review.sqlite3"
            old = MotionReviewStore(database)
            old.build(motions)
            old.save_action_decision("baby_walk_ff_180_start_r", "exclude", paths[0])
            old.export()
            with self.assertRaisesRegex(ValueError, "grouping changed"):
                old.build(motions, {p: MotionDescription("walking with baby", "baby", "standing") for p in paths})
            old.close()
            descriptions = {p: MotionDescription("walking with baby", "baby", "standing") for p in paths}
            backup = reset_review_catalog(database, motions, descriptions)
            with closing(sqlite3.connect(backup)) as saved:
                self.assertEqual(saved.execute("SELECT decision FROM action_reviews").fetchone()[0], "exclude")
                self.assertEqual(saved.execute("SELECT count(*) FROM action_review_events").fetchone()[0], 1)
            fresh = MotionReviewStore(database)
            try:
                self.assertEqual(fresh.action_queue("all"), ["baby_walk"])
                self.assertEqual(fresh.clips_for_action("baby_walk")[0], paths[1])
                self.assertEqual(fresh.action_summaries()[0]["representative_path"], paths[1])
                self.assertEqual(fresh.adopt_legacy_decisions()["adopted_actions"], 0)
                for table in ("action_reviews", "action_review_events", "review_events"):
                    self.assertEqual(fresh.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
                fresh.export()
                self.assertEqual((root / "excluded.txt").read_text(), "")
                self.assertIn("baby_walk", (root / "review_groups.csv").read_text())
                fresh.save_action_decision("baby_walk", "include", paths[1])
                fresh.export()
                self.assertEqual(set((root / "included.txt").read_text().splitlines()), set(paths + [mirror]))
            finally:
                fresh.close()

    def test_failed_reset_preserves_current_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "review.sqlite3"
            old = MotionReviewStore(database)
            old.close()
            before = database.read_bytes()
            with self.assertRaises(FileNotFoundError):
                reset_review_catalog(database, Path(directory) / "missing", {})
            self.assertEqual(database.read_bytes(), before)

    @unittest.skipUnless(sys.platform == "win32", "Windows viewer file-sharing guard")
    def test_open_viewer_database_blocks_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "bvh"
            data.mkdir()
            (data / "walk_001__A001.bvh").write_text("fixture")
            database = root / "review.sqlite3"
            viewer_store = MotionReviewStore(database)
            try:
                viewer_store.build(data)
                viewer_store.save_action_decision("walk", "uncertain")
                with self.assertRaisesRegex(PermissionError, "Close every"):
                    reset_review_catalog(database, data, {})
                self.assertEqual(viewer_store.action_queue("uncertain"), ["walk"])
            finally:
                viewer_store.close()


if __name__ == "__main__":
    unittest.main()
