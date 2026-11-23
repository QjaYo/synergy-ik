"""Verify persistent review decisions, mirror propagation and conservative screening."""

from pathlib import Path
import sys
import sqlite3
import csv
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess/motion_review"))
from motion_review_data import (MotionReviewStore, MotionDescription, load_motion_descriptions,
                                classify_motion_name, motion_name, original_key)


class MetadataTests(unittest.TestCase):
    def test_exact_path_join_keeps_dates_variants_and_ignores_mirror_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            fields = ["move_soma_uniform_path", "content_technical_description", "content_short_description", "content_props"]
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows([
                    ["soma_uniform/bvh/day1/same_001__A001.bvh", "Holding a stinky baby", "", "baby (sandbag)"],
                    ["soma_uniform/bvh/day1/same_001__A001_M.bvh", "Mirrored text", "", "0"],
                    ["soma_uniform\\bvh\\day2\\same_001__A001.bvh", "", "Walking backwards", "0"],
                ])
            result = load_motion_descriptions(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result["day1/same_001__A001.bvh"], MotionDescription("Holding a stinky baby", "baby (sandbag)"))
            self.assertEqual(result["day2/same_001__A001.bvh"], MotionDescription("Walking backwards", "None listed"))

    def test_conflicting_metadata_cannot_silently_replace_description(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            path.write_text("move_soma_uniform_path,content_technical_description,content_props\n"
                            "bvh/day/a.bvh,Walking,0\nbvh/day/a.bvh,Carrying load,crate\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Conflicting metadata"):
                load_motion_descriptions(path)


class ScreeningTests(unittest.TestCase):
    def test_explicit_scope_and_ambiguous_names(self):
        cases = {
            "sit_on_chair_loop_r": "exclude",
            "drinking_wall_leaning_mug_090_r": "exclude",
            "come_down_50cm_box_r": "exclude",
            "small_heavy_two_hands": "exclude",
            "small_light_one_hand": "include",
            "change_idle_crawl_to_idle_crawl_right": "include",
            "jump_and_land_heavy": "include",
            "jump_and_land_heavy_on_chair": "review",
            "walking_stairs": "review",
            "carry_box": "review",
            "unknown_action": "review",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify_motion_name(name)[0], expected)

    def test_mirror_suffix_is_not_an_action_keyword(self):
        self.assertEqual(original_key("day/looking_in_mirror_001__A001_M.bvh"),
                         "day/looking_in_mirror_001__A001.bvh")

    def test_action_name_groups_takes_actors_and_dates_but_keeps_variant(self):
        self.assertEqual(motion_name("day1/dancing_V002_001__A001.bvh"), "dancing_v002")
        self.assertEqual(motion_name("day2/dancing_V002_006__A003_M.bvh"), "dancing_v002")
        self.assertNotEqual(motion_name("day1/dancing_V001_001__A001.bvh"), "dancing_v002")
        self.assertNotEqual(motion_name("day1/dancing_V002_R_001__A001.bvh"), "dancing_v002")
        self.assertEqual(original_key("day/looking_in_mirror_001__A001.bvh"),
                         "day/looking_in_mirror_001__A001.bvh")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "bvh"
        self.data.mkdir()
        self.first = "unknown_001__A001.bvh"
        self.mirror = "unknown_001__A001_M.bvh"
        self.second = "unknown_002__A001.bvh"
        for name in (self.first, self.mirror, self.second):
            (self.data / name).write_text("fixture", encoding="utf-8")
        self.store = MotionReviewStore(self.root / "review" / "review.sqlite3")
        self.store.build(self.data)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_save_advances_fixed_queue_and_resume_excludes_reviewed_items(self):
        queue = self.store.queue()
        self.store.save_decision(queue[0], "include")
        self.assertEqual(queue[1], self.second)
        database = self.store.path
        self.store.close()
        self.store = MotionReviewStore(database)
        self.assertEqual(self.store.queue(), [self.second])
        self.store.export()
        self.assertEqual(set((database.parent / "included.txt").read_text().splitlines()),
                         {self.first, self.mirror})

    def test_rebuild_preserves_override_and_correction_changes_both_variants(self):
        self.store.save_decision(self.first, "include")
        self.store.build(self.data)
        self.assertEqual(self.store.get(self.first)["human_decision"], "include")
        self.store.save_decision(self.first, "exclude")
        summary = self.store.export()
        self.assertEqual(summary["files_exclude"], 2)
        self.assertEqual((self.store.path.parent / "included.txt").read_text(), "")
        events = self.store.connection.execute("SELECT previous_decision,decision FROM review_events").fetchall()
        self.assertEqual([tuple(row) for row in events], [(None, "include"), ("include", "exclude")])

    def test_clear_override_returns_pending_and_load_error_is_not_exclusion(self):
        self.store.save_decision(self.first, "exclude")
        self.store.save_decision(self.first, None)
        self.store.set_load_error(self.first, "bad BVH")
        self.assertIn(self.first, self.store.queue())
        self.assertIsNone(self.store.get(self.first)["human_decision"])

    def test_uncertain_type_is_reviewed_and_exported_separately(self):
        self.store.save_action_decision("unknown", "uncertain", self.first)
        self.assertEqual(self.store.action_queue(), [])
        self.assertEqual(self.store.action_queue("reviewed"), ["unknown"])
        self.assertEqual(self.store.action_queue("uncertain"), ["unknown"])
        self.assertEqual(self.store.queue("uncertain"), [self.first, self.second])
        summary = self.store.export()
        self.assertEqual(summary["uncertain_action_types"], 1)
        self.assertEqual(summary["files_uncertain"], 3)
        for filename in ("included.txt", "excluded.txt", "needs_review.txt"):
            self.assertEqual((self.store.path.parent / filename).read_text(), "")
        self.assertEqual(set((self.store.path.parent / "uncertain.txt").read_text().splitlines()),
                         {self.first, self.mirror, self.second})
        self.assertEqual((self.store.path.parent / "uncertain_types.txt").read_text().splitlines(), ["unknown"])
        self.store.save_action_decision("unknown", "include", self.second)
        self.assertEqual(self.store.action_queue("uncertain"), [])
        self.assertEqual(self.store.export()["files_include"], 3)
        self.assertEqual((self.store.path.parent / "uncertain.txt").read_text(), "")
        events = self.store.connection.execute("SELECT previous_decision,decision FROM action_review_events ORDER BY id")
        self.assertEqual([tuple(row) for row in events], [(None, "uncertain"), ("uncertain", "include")])

    def test_uncertain_clip_override_and_clear_preserve_scope(self):
        self.store.save_action_decision("unknown", "include")
        self.store.save_decision(self.first, "uncertain")
        self.assertEqual(self.store.queue("uncertain"), [self.first])
        self.assertEqual(self.store.action_queue("uncertain"), ["unknown"])
        self.assertEqual(self.store.get(self.second)["effective_decision"], "include")
        self.store.save_decision(self.first, None)
        self.assertEqual(self.store.queue("uncertain"), [])
        self.assertEqual(self.store.get(self.first)["effective_decision"], "include")
        self.store.save_decision(self.first, "uncertain")
        self.store.save_action_decision("unknown", "uncertain")
        self.assertIsNone(self.store.get(self.first)["human_decision"])
        self.assertEqual(len(self.store.queue("uncertain")), 2)

    def test_uncertain_and_final_clip_decisions_do_not_reopen_first_pass(self):
        self.store.save_decision(self.first, "uncertain")
        self.store.save_decision(self.second, "include")
        self.assertEqual(self.store.action_queue(), [])
        self.assertEqual(self.store.action_queue("uncertain"), ["unknown"])

    def test_old_schema_migration_preserves_all_rows_and_events(self):
        self.store.save_decision(self.first, "exclude")
        self.store.save_action_decision("unknown", "include", self.first)
        self.store.save_decision(self.second, "exclude")
        old_path = self.root / "old.sqlite3"
        dump = "\n".join(self.store.connection.iterdump()).replace("'include','exclude','uncertain'", "'include','exclude'")
        with sqlite3.connect(old_path) as connection:
            connection.executescript(dump)
        tables = ("settings", "motions", "files", "action_reviews", "action_review_events", "review_events")
        before = {table: [tuple(row) for row in self.store.connection.execute(f"SELECT * FROM {table} ORDER BY rowid")] for table in tables}
        migrated = MotionReviewStore(old_path)
        try:
            self.assertTrue(migrated.schema_backup.is_file())
            for table in tables:
                self.assertEqual([tuple(row) for row in migrated.connection.execute(f"SELECT * FROM {table} ORDER BY rowid")], before[table])
            with sqlite3.connect(migrated.schema_backup) as backup:
                self.assertNotIn("uncertain", backup.execute("SELECT sql FROM sqlite_master WHERE name='motions'").fetchone()[0])
                self.assertEqual(backup.execute("SELECT * FROM review_events ORDER BY rowid").fetchall(), before["review_events"])
            self.assertEqual(migrated.connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(migrated.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(migrated.connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertIsNotNone(migrated.connection.execute("SELECT 1 FROM sqlite_master WHERE name='motions_action'").fetchone())
            migrated.save_action_decision("unknown", "uncertain")
            migrated.save_decision(self.first, "uncertain")
        finally:
            migrated.close()
        migrated = MotionReviewStore(old_path)
        try:
            self.assertIsNone(migrated.schema_backup)
            self.assertEqual(migrated.queue("uncertain"), [self.first, self.second])
        finally:
            migrated.close()

    def test_uncertain_ui_advances_autoplays_and_reopens_with_saved_selection(self):
        import review_motions
        other = "zzz_pending_001__A001.bvh"
        (self.data / other).write_text("fixture", encoding="utf-8")
        self.store.build(self.data)
        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()), \
                patch.object(review_motions, "create_initial_camera", return_value=object()), \
                patch.object(review_motions, "load_aligned_uniform_mesh", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, self.store.action_queue("all"), 1.0, show_mesh=True)
            try:
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.paused = True
                viewer.keyboard_callback(b"U")
                self.assertEqual(viewer.index, 1)
                self.assertFalse(viewer.paused)
                self.assertEqual(self.store.get(self.first)["effective_decision"], "uncertain")
                self.assertEqual(review_motions.prepare_review_queue(self.store), (["unknown", "zzz_pending"], 1))
                self.assertEqual(review_motions.prepare_review_queue(self.store, "uncertain"), (["unknown"], 0))
                viewer.keyboard_callback(b"p")
                self.assertEqual(viewer.saved_verdict(), "uncertain")
                with patch.object(viewer, "rectangle"), patch.object(viewer, "text"), patch.object(viewer, "button") as button_mock:
                    viewer.draw_controls()
                    unsure = next(call for call in button_mock.call_args_list if call.args[4] == "uncertain")
                    self.assertTrue(unsure.kwargs["selected"])
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                with patch.object(self.store, "save_action_decision", side_effect=sqlite3.OperationalError("disk full")):
                    viewer.keyboard_callback(b"u")
                self.assertEqual(viewer.index, 0)
                self.assertIn("Save failed", viewer.status)
                viewer.keyboard_callback(b"x")
                self.assertEqual(self.store.get(self.first)["effective_decision"], "exclude")
            finally:
                viewer.shutdown()

    def test_invalid_verdict_does_not_mutate_database(self):
        with self.assertRaises(ValueError):
            self.store.save_decision(self.first, "bad")
        self.assertIsNone(self.store.get(self.first)["human_decision"])
        self.assertEqual(self.store.connection.execute("SELECT count(*) FROM review_events").fetchone()[0], 0)

    def test_quality_exclusion_is_local_persistent_and_exported_with_mirror(self):
        self.store.save_action_decision("unknown", "include")
        self.store.save_decision(self.first, "quality_exclude")
        self.store.build(self.data)
        for verdict in ("exclude", "uncertain", "include", None):
            self.store.save_action_decision("unknown", verdict)
            self.assertEqual(self.store.get(self.first)["effective_decision"], "quality_exclude")
            self.assertIsNone(self.store.get(self.second)["human_decision"])
        self.store.save_action_decision("unknown", "include")
        database = self.store.path
        self.store.close()
        self.store = MotionReviewStore(database)
        self.assertEqual(self.store.queue("quality-exclude"), [self.first])
        self.assertEqual(self.store.action_queue("quality-exclude"), ["unknown"])
        self.assertEqual(self.store.action_queue(), [])
        self.assertEqual(self.store.clips_for_action("unknown"), [self.second, self.first])
        self.assertEqual(self.store.clips_for_action("unknown", prefer_quality=True), [self.first, self.second])
        self.assertEqual(self.store.action_summaries()[0]["representative_path"], self.second)
        summary = self.store.export()
        self.assertEqual(summary["files_quality_exclude"], 2)
        self.assertEqual(summary["quality_excluded_action_types"], 1)
        self.assertEqual((database.parent / "included.txt").read_text().splitlines(), [self.second])
        self.assertEqual(set((database.parent / "quality_excluded.txt").read_text().splitlines()), {self.first, self.mirror})
        for filename in ("excluded.txt", "uncertain.txt", "needs_review.txt"):
            self.assertEqual((database.parent / filename).read_text(), "")
        self.store.save_decision(self.first, None)
        self.assertEqual(self.store.get(self.first)["effective_decision"], "include")
        self.assertEqual(self.store.export()["files_include"], 3)
        self.assertEqual((database.parent / "quality_excluded.txt").read_text(), "")
        events = self.store.connection.execute("SELECT previous_decision,decision FROM review_events ORDER BY id")
        self.assertEqual([tuple(row) for row in events], [(None, "quality_exclude"), ("quality_exclude", None)])

    def test_quality_exclusion_cannot_be_adopted_or_saved_as_a_group_verdict(self):
        self.store.save_decision(self.first, "quality_exclude")
        self.assertEqual(self.store.adopt_legacy_decisions()["adopted_actions"], 0)
        self.assertIsNone(self.store.get(self.second)["action_decision"])
        self.assertEqual(self.store.action_queue(), ["unknown"])
        with self.assertRaises(ValueError):
            self.store.save_action_decision("unknown", "quality_exclude")
        self.store.save_decision(self.second, "quality_exclude")
        self.assertEqual(self.store.action_queue(), [])

    def test_quality_schema_upgrade_preserves_uncertain_decisions_and_history(self):
        self.store.save_action_decision("unknown", "uncertain")
        self.store.save_decision(self.first, "exclude")
        path = self.root / "before_quality.sqlite3"
        dump = "\n".join(self.store.connection.iterdump()).replace(",'quality_exclude'", "")
        with sqlite3.connect(path) as connection:
            connection.executescript(dump)
        tables = ("settings", "motions", "files", "action_reviews", "action_review_events", "review_events")
        before = {table: [tuple(row) for row in self.store.connection.execute(f"SELECT * FROM {table} ORDER BY rowid")] for table in tables}
        upgraded = MotionReviewStore(path)
        try:
            self.assertIsNotNone(upgraded.schema_backup)
            with sqlite3.connect(upgraded.schema_backup) as backup:
                self.assertNotIn("quality_exclude", backup.execute("SELECT sql FROM sqlite_master WHERE name='motions'").fetchone()[0])
            for table in tables:
                self.assertEqual([tuple(row) for row in upgraded.connection.execute(f"SELECT * FROM {table} ORDER BY rowid")], before[table])
            self.assertEqual(upgraded.connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(upgraded.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            upgraded.save_decision(self.first, "quality_exclude")
            self.assertEqual(upgraded.get(self.second)["effective_decision"], "uncertain")
        finally:
            upgraded.close()

    def test_quality_ui_keeps_other_takes_pending_and_allows_explicit_clear(self):
        import review_motions

        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()), \
                patch.object(review_motions, "create_initial_camera", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, ["unknown"], 1.0)
            try:
                viewer.keyboard_callback(b"Q")
                self.assertIsNone(self.store.get(self.first)["human_decision"])
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"Q")
                self.assertEqual(viewer.index, 0)
                self.assertEqual(viewer.current_key, self.second)
                self.assertFalse(viewer.paused)
                self.assertEqual(self.store.get(self.first)["effective_decision"], "quality_exclude")
                self.assertEqual(self.store.get(self.second)["effective_decision"], "review")
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"o")
                self.assertEqual(viewer.index, 1)
                self.assertEqual(self.store.get(self.first)["effective_decision"], "quality_exclude")
                viewer.keyboard_callback(b"p")
                self.assertEqual(viewer.current_key, self.second)
                viewer.keyboard_callback(b"b")
                self.assertEqual(viewer.current_key, self.first)
                viewer.keyboard_callback(b"d")
                with patch.object(viewer, "rectangle"), patch.object(viewer, "text"), patch.object(viewer, "button") as buttons:
                    viewer.draw_controls()
                    quality = next(call for call in buttons.call_args_list if call.args[4] == "quality_exclude")
                    self.assertTrue(quality.kwargs["selected"])
                    self.assertIn("clear_quality", [call.args[4] for call in buttons.call_args_list])
                viewer.act("clear_quality")
                self.assertEqual(self.store.get(self.first)["effective_decision"], "include")
                self.assertEqual(self.store.get(self.second)["effective_decision"], "include")
            finally:
                viewer.shutdown()

    def test_quality_queue_displays_the_excluded_take_and_save_failure_does_not_advance(self):
        import review_motions

        self.store.save_action_decision("unknown", "include")
        self.store.save_decision(self.first, "quality_exclude")
        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()), \
                patch.object(review_motions, "create_initial_camera", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, ["unknown"], 1.0, selection="quality-exclude")
            try:
                self.assertEqual(viewer.current_key, self.first)
                viewer.keyboard_callback(b"b")
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                with patch.object(self.store, "save_decision", side_effect=sqlite3.OperationalError("disk full")):
                    viewer.keyboard_callback(b"q")
                self.assertEqual(viewer.index, 0)
                self.assertEqual(viewer.current_key, self.second)
                self.assertEqual(self.store.get(self.second)["effective_decision"], "include")
                self.assertIn("Save failed", viewer.status)
            finally:
                viewer.shutdown()

    def test_missing_original_and_changed_root_reject_without_erasing_review(self):
        self.store.save_decision(self.first, "include")
        (self.data / self.first).unlink()
        with self.assertRaises(ValueError):
            self.store.build(self.data)
        self.assertEqual(self.store.get(self.first)["human_decision"], "include")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        with self.assertRaises(ValueError):
            self.store.build(elsewhere)

    def test_file_path_cannot_escape_data_root(self):
        with self.assertRaises(ValueError):
            self.store.file_path("../outside.bvh")

    def test_ui_keys_save_advance_and_correct_without_a_window(self):
        import review_motions

        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()), \
                patch.object(review_motions, "create_initial_camera", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, self.store.queue(), 1.0, unit="clip")
            try:
                viewer.keyboard_callback(b"o")
                self.assertEqual(viewer.index, 0)  # Loading disables decisions.
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"O")
                self.assertEqual(self.store.get(self.first)["human_decision"], "include")
                self.assertEqual(viewer.index, 1)
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"x")
                self.assertEqual(self.store.queue(), [])
                self.assertEqual(viewer.index, 2)
                viewer.keyboard_callback(b"p")
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"o")
                self.assertEqual(self.store.get(self.second)["human_decision"], "include")
                self.assertEqual(self.store.export()["files_include"], 3)
            finally:
                viewer.shutdown()

    def test_ui_save_failure_stays_on_current_motion(self):
        import review_motions

        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()), \
                patch.object(review_motions, "create_initial_camera", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, self.store.queue(), 1.0, unit="clip")
            try:
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                with patch.object(self.store, "save_decision", side_effect=sqlite3.OperationalError("disk full")):
                    viewer.keyboard_callback(b"o")
                self.assertEqual(viewer.index, 0)
                self.assertIsNone(self.store.get(self.first)["human_decision"])
                self.assertIn("Save failed", viewer.status)
            finally:
                viewer.shutdown()

    def test_ui_failed_load_remains_pending_and_skip_does_not_decide(self):
        import review_motions

        with patch.object(review_motions, "load_canonicalized_bvh", side_effect=ValueError("invalid motion")):
            viewer = review_motions.create_review_viewer(self.store, self.store.queue(), 1.0, unit="clip")
            try:
                with self.assertRaises(ValueError):
                    viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.keyboard_callback(b"x")
                self.assertEqual(viewer.index, 0)
                self.assertEqual(self.store.get(self.first)["load_error"], "invalid motion")
                viewer.keyboard_callback(b"n")
                self.assertEqual(viewer.index, 1)
                self.assertEqual(len(self.store.queue()), 2)
            finally:
                viewer.shutdown()

    def test_action_verdict_applies_to_takes_mirrors_and_future_members(self):
        self.assertEqual(self.store.action_queue(), ["unknown"])
        self.store.save_action_decision("unknown", "include", self.first)
        self.assertEqual(self.store.queue(), [])
        self.assertEqual(self.store.action_queue(), [])
        self.assertEqual(self.store.get(self.second)["decision_source"], "action")
        self.assertEqual(self.store.export()["files_include"], 3)
        new_take = "unknown_003__A009.bvh"
        (self.data / new_take).write_text("fixture", encoding="utf-8")
        self.store.build(self.data)
        self.assertEqual(self.store.get(new_take)["effective_decision"], "include")

    def test_clip_exception_and_explicit_type_replacement(self):
        self.store.save_action_decision("unknown", "include")
        self.store.save_decision(self.first, "exclude")
        self.assertEqual(self.store.get(self.first)["effective_decision"], "exclude")
        self.assertEqual(self.store.get(self.second)["effective_decision"], "include")
        self.assertEqual(self.store.export()["files_exclude"], 2)
        self.store.save_action_decision("unknown", "include")
        self.assertIsNone(self.store.get(self.first)["human_decision"])
        self.assertEqual(self.store.export()["files_include"], 3)
        self.assertEqual(self.store.connection.execute("SELECT count(*) FROM review_events").fetchone()[0], 2)

    def test_clear_type_preserves_individual_exception(self):
        self.store.save_action_decision("unknown", "include")
        self.store.save_decision(self.first, "exclude")
        self.store.save_action_decision("unknown", None)
        self.assertEqual(self.store.get(self.first)["effective_decision"], "exclude")
        self.assertEqual(self.store.queue(), [self.second])
        self.assertEqual(self.store.action_queue(), ["unknown"])

    def test_legacy_adoption_preserves_labels_and_events_and_creates_backup(self):
        self.store.save_decision(self.first, "include")
        before = [tuple(row) for row in self.store.connection.execute("SELECT * FROM review_events")]
        result = self.store.adopt_legacy_decisions()
        self.assertEqual(result["adopted_actions"], 1)
        self.assertTrue(Path(result["backup"]).is_file())
        backup = sqlite3.connect(result["backup"])
        try:
            self.assertEqual(backup.execute("SELECT count(*) FROM action_reviews").fetchone()[0], 0)
            self.assertEqual(backup.execute("SELECT human_decision FROM motions WHERE original_path=?", (self.first,)).fetchone()[0], "include")
        finally:
            backup.close()
        self.assertEqual([tuple(row) for row in self.store.connection.execute("SELECT * FROM review_events")], before)
        self.assertEqual(self.store.get(self.first)["human_decision"], "include")
        self.assertEqual(self.store.get(self.second)["effective_decision"], "include")
        self.assertEqual(self.store.adopt_legacy_decisions()["adopted_actions"], 0)
        self.store.save_action_decision("unknown", None)
        self.store.adopt_legacy_decisions()
        self.assertIsNone(self.store.get(self.second)["action_decision"])

    def test_conflicting_legacy_decisions_stay_pending_for_type_review(self):
        self.store.save_decision(self.first, "include")
        self.store.save_decision(self.second, "exclude")
        self.assertEqual(self.store.adopt_legacy_decisions()["adopted_actions"], 0)
        self.assertEqual(self.store.queue(), [])
        self.assertEqual(self.store.action_queue(), ["unknown"])
        self.assertEqual(self.store.get(self.second)["human_decision"], "exclude")

    def test_action_save_rolls_back_if_event_write_fails(self):
        self.store.save_decision(self.first, "exclude")
        self.store.connection.executescript("""CREATE TRIGGER reject_event
            AFTER INSERT ON action_review_events BEGIN SELECT RAISE(ABORT,'write failed'); END;""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_action_decision("unknown", "include")
        self.assertIsNone(self.store.get(self.first)["action_decision"])
        self.assertEqual(self.store.get(self.first)["human_decision"], "exclude")
        self.assertEqual(self.store.get(self.second)["effective_decision"], "review")

    def test_wrong_representative_cannot_label_another_type(self):
        with self.assertRaises(ValueError):
            self.store.save_action_decision("unknown", "include", "missing.bvh")
        self.assertIsNone(self.store.get(self.first)["action_decision"])

    def test_group_export_and_action_filter(self):
        self.store.export()
        self.assertEqual((self.store.path.parent / "needs_review_types.txt").read_text().splitlines(), ["unknown"])
        self.assertEqual(self.store.queue("all", "unknown"), [self.first, self.second])
        self.assertEqual(self.store.action_queue("all", "missing"), [])
        self.store.save_action_decision("unknown", "include")
        summary = self.store.export()
        self.assertEqual(summary["action_types"], 1)
        self.assertEqual(summary["pending_action_types"], 0)
        self.assertEqual(summary["reviewed_action_types"], 1)

    def test_action_ui_browses_takes_and_advances_by_type(self):
        import review_motions
        other = "zzz_other_001__A001.bvh"
        (self.data / other).write_text("fixture", encoding="utf-8")
        self.store.build(self.data)
        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()), \
                patch.object(review_motions, "create_initial_camera", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, self.store.action_queue(), 1.0)
            try:
                self.assertEqual(len(viewer.members), 2)
                viewer.keyboard_callback(b"b")
                self.assertEqual(viewer.current_key, self.second)
                self.assertEqual(viewer.index, 0)
                self.assertEqual(len(self.store.action_queue()), 2)
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"o")
                self.assertEqual(viewer.index, 1)
                self.assertEqual(viewer.current_key, other)
                self.assertEqual(self.store.get(self.first)["effective_decision"], "include")
                self.assertEqual(self.store.get(self.second)["effective_decision"], "include")
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"x")
                self.assertEqual(viewer.index, 2)
                self.assertEqual(self.store.action_queue(), [])
                viewer.keyboard_callback(b"p")
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"o")
                self.assertEqual(self.store.get(other)["effective_decision"], "include")
            finally:
                viewer.shutdown()

    def test_reopen_keeps_completed_types_and_resumes_first_pending(self):
        from review_motions import prepare_review_queue

        for name in ("aaa_done_001__A001.bvh", "walk_001__A001.bvh", "zzz_pending_001__A001.bvh"):
            (self.data / name).write_text("fixture", encoding="utf-8")
        self.store.build(self.data)
        self.store.save_action_decision("aaa_done", "exclude")
        self.store.save_action_decision("unknown", "include")
        database = self.store.path
        self.store.close()
        self.store = MotionReviewStore(database)
        keys, start = prepare_review_queue(self.store)
        self.assertEqual(keys, ["aaa_done", "unknown", "walk", "zzz_pending"])
        self.assertEqual(start, 3)
        self.assertEqual(prepare_review_queue(self.store, action_name="unknown"), (["unknown"], 0))
        self.assertEqual(prepare_review_queue(self.store, limit=2), (keys[:2], 0))
        self.store.save_action_decision("zzz_pending", "include")
        self.assertEqual(prepare_review_queue(self.store), (keys, 0))
        self.assertEqual(prepare_review_queue(self.store, selection="pending"), ([], 0))

    def test_clip_queue_retains_completed_entries_on_resume(self):
        from review_motions import prepare_review_queue

        self.store.save_decision(self.first, "include")
        self.assertEqual(prepare_review_queue(self.store, unit="clip"), ([self.first, self.second], 1))

    def test_returned_type_shows_saved_verdict_and_allows_overwrite(self):
        import review_motions

        other = "zzz_other_001__A001.bvh"
        (self.data / other).write_text("fixture", encoding="utf-8")
        self.store.build(self.data)
        self.store.save_action_decision("unknown", "include")
        keys, start = review_motions.prepare_review_queue(self.store)
        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()), \
                patch.object(review_motions, "create_initial_camera", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, keys, 1.0, start_index=start)
            try:
                self.assertEqual(viewer.index, 1)
                self.assertIsNone(viewer.saved_verdict())
                viewer.keyboard_callback(b"p")
                self.assertEqual(viewer.saved_verdict(), "include")
                # Inspect the actual control arguments without an OpenGL context.
                with patch.object(viewer, "rectangle"), patch.object(viewer, "text") as text_mock, \
                        patch.object(viewer, "button") as button_mock:
                    viewer.draw_controls()
                    decision_calls = {call.args[4]: call for call in button_mock.call_args_list if call.args[4] in ("include", "exclude")}
                    self.assertTrue(decision_calls["include"].kwargs["selected"])
                    self.assertFalse(decision_calls["exclude"].kwargs["selected"])
                    self.assertTrue(any(call.args[0] == "Group choice: INCLUDE" for call in text_mock.call_args_list))
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.ready_at = 0
                viewer.keyboard_callback(b"x")
                self.assertEqual(self.store.get(self.second)["effective_decision"], "exclude")
                viewer.keyboard_callback(b"p")
                self.assertEqual(viewer.saved_verdict(), "exclude")
                self.assertEqual([row[0] for row in self.store.connection.execute(
                    "SELECT decision FROM action_review_events ORDER BY id")], ["include", "exclude"])
            finally:
                viewer.shutdown()

    def test_automatic_label_is_not_displayed_as_saved_human_verdict(self):
        import review_motions

        other = "walk_001__A001.bvh"
        (self.data / other).write_text("fixture", encoding="utf-8")
        self.store.build(self.data)
        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, ["walk"], 1.0)
            try:
                self.assertEqual(viewer.row["effective_decision"], "include")
                self.assertIsNone(viewer.saved_verdict())
            finally:
                viewer.shutdown()

    def test_mesh_button_toggles_without_changing_review_decisions(self):
        import review_motions
        from OpenGL import GLUT

        with patch.object(review_motions, 'load_canonicalized_bvh', return_value=object()), \
                patch.object(review_motions, 'create_initial_camera', return_value=object()), \
                patch.object(review_motions, 'load_aligned_uniform_mesh', return_value=object()) as loader:
            viewer = review_motions.create_review_viewer(self.store, ['unknown'], 1.0, show_mesh=False)
            try:
                with patch.object(viewer, 'rectangle'), patch.object(viewer, 'text'):
                    viewer.draw_controls()
                button = next(b for b in viewer.buttons if b[4] == 'mesh')
                viewer.mouse_func(GLUT.GLUT_LEFT_BUTTON, GLUT.GLUT_DOWN, button[0]+10, button[1]+10)
                self.assertTrue(viewer.show_mesh)
                viewer.loading.result(timeout=2)
                viewer.poll_loading()
                viewer.mesh_loading.result(timeout=2)
                viewer.poll_mesh_loading()
                self.assertIsNotNone(viewer.mesh)
                self.assertEqual(loader.call_args.args[0], review_motions.DEFAULT_UNIFORM_ROOT / self.first)
                viewer.keyboard_callback(b'm')
                self.assertFalse(viewer.show_mesh)
                viewer.keyboard_callback(b's')
                self.assertFalse(viewer.show_skeleton)
                viewer.load_sample(1)
                self.assertIsNone(viewer.mesh)
                self.assertIsNone(self.store.get(self.first)['human_decision'])
                self.assertIsNone(self.store.get(self.first)['action_decision'])
            finally:
                viewer.shutdown()

    def test_description_follows_selected_take_and_details_do_not_change_verdicts(self):
        import review_motions
        descriptions = {self.first: MotionDescription("First take, holding a baby.", "baby"),
                        self.second: MotionDescription("Second take, walking backwards.", "None listed")}
        with patch.object(review_motions, "load_canonicalized_bvh", return_value=object()):
            viewer = review_motions.create_review_viewer(self.store, ["unknown"], 1.0, descriptions=descriptions)
            try:
                self.assertIn("First take", " ".join(viewer.description_lines()))
                with patch.object(viewer, "rectangle"), patch.object(viewer, "text"), patch.object(viewer, "button") as buttons:
                    viewer.draw_controls()
                    self.assertNotIn("clear", [c.args[4] for c in buttons.call_args_list])
                before_height = viewer.header_height()
                viewer.keyboard_callback(b"d")
                self.assertTrue(viewer.show_details)
                self.assertGreater(viewer.header_height(), before_height)
                with patch.object(viewer, "rectangle"), patch.object(viewer, "text"), patch.object(viewer, "button") as buttons:
                    viewer.draw_controls()
                    self.assertIn("clear", [c.args[4] for c in buttons.call_args_list])
                viewer.keyboard_callback(b"b")
                self.assertIn("Second take", " ".join(viewer.description_lines()))
                self.assertIsNone(self.store.get(self.first)["action_decision"])
                self.assertEqual(self.store.connection.execute("SELECT count(*) FROM action_review_events").fetchone()[0], 0)
            finally:
                viewer.shutdown()


if __name__ == "__main__":
    unittest.main()
