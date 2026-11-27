"""Conservative filename screening and persistent Bones-seed inclusion review."""

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
import sys

from review_groups import GROUPING_VERSION, build_review_groups


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "canonical_bvh"
DEFAULT_REVIEW_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_METADATA = Path("D:/data/motion/bones-seed/metadata/seed_metadata_v004.csv")
RULE_VERSION = "filename-v1"
STATUS_QUERY = """
    SELECT m.*, a.decision AS action_decision,
           COALESCE(m.human_decision,a.decision,m.auto_decision) AS effective_decision,
           CASE WHEN m.human_decision IS NOT NULL THEN 'clip'
                WHEN a.decision IS NOT NULL THEN 'action' ELSE 'automatic' END AS decision_source
    FROM motions m LEFT JOIN action_reviews a ON a.action_name=m.action_name
"""


@dataclass(frozen=True)
class MotionDescription:
    text: str
    props: str
    body_position: str = ""


def load_motion_descriptions(metadata_path: Path) -> dict[str, MotionDescription]:
    """Join official descriptions by exact original BVH path, not action similarity.

    Viewer lookups are read-only; catalog construction also uses body/prop context
    to keep incompatible behaviors in separate review groups.
    """
    descriptions = {}
    with Path(metadata_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"move_soma_uniform_path", "content_technical_description", "content_props"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Metadata is missing Uniform path/description/props columns")
        for row in reader:
            raw_path = row["move_soma_uniform_path"].replace("\\", "/")
            parts = raw_path.strip("/").split("/")
            if len(parts) < 2 or not parts[-1].lower().endswith(".bvh"):
                continue
            key = "/".join(parts[-2:])
            if original_key(key) != key:
                continue
            text = next((row.get(field, "").strip() for field in
                         ("content_technical_description", "content_natural_desc_1", "content_short_description")
                         if row.get(field, "").strip()), "")
            props = row["content_props"].strip()
            if props.lower() in ("", "0", "none", "nan"):
                props = "None listed"
            description = MotionDescription(text, props, row.get("content_body_position", "").strip())
            if key in descriptions and descriptions[key] != description:
                raise ValueError(f"Conflicting metadata rows for {key}")
            descriptions[key] = description
    return descriptions


def original_key(relative_path: str) -> str:
    """Map a supplied _M variant to its original; retain date, take and actor."""
    return re.sub(r"(__A\d+)_M(?=\.bvh$)", r"\1", relative_path, flags=re.I)


def motion_name(relative_path: str) -> str:
    """Remove actor and take suffixes for readable action grouping only."""
    name = Path(relative_path).stem.lower()
    name = re.sub(r"__a\d+(?:_m)?$", "", name)
    return re.sub(r"_\d+$", "", name)


def classify_motion_name(name: str) -> tuple[str, str, str]:
    """Return include/exclude/review, rule ID and explicit filename evidence.

    These are semantic screening decisions, not physical or visual validation.
    Unknown names stay pending. A prop or the word 'heavy' alone is insufficient.
    """
    exclusion_rules = (
        (r"(?:sit|sitting|seat)(?:_on)?_(?:chair|sofa|couch|bench)|having_a_sit_table",
         "furniture_support", "Filename explicitly describes furniture support"),
        (r"wall_lean|lean(?:ing)?_(?:against_)?wall",
         "wall_support", "Filename explicitly describes leaning on a wall"),
        (r"come_(?:up|down)_\d+cm_box|(?:climb|climbing|slide|jump|slip).*ladder|ladder_(?:climb|slide|jump|slip)",
         "raised_support", "Filename explicitly describes a box platform or ladder"),
        (r"(?:horse|motorcycle|bicycle)_riding|riding_(?:horse|motorcycle|bicycle)",
         "riding_support", "Filename explicitly describes riding external support"),
        (r"(?:^|_)(?:heavy_item|(?:small_|medium_|big_)*heavy_(?:one_hand|two_hands))|(?:lift|carry|carrying|push|pull)\w*_(?:a_)?heavy(?:_|$)",
         "explicit_heavy_load", "Filename explicitly identifies a heavy handled load"),
    )
    for pattern, rule_id, reason in exclusion_rules:
        if re.search(pattern, name):
            return "exclude", rule_id, reason

    uncertain_context = re.search(
        r"(?:^|_)(?:chair|sofa|couch|bench|wall|table|ladder|stairs?|staircase|stepup|stepdown|"
        r"box|bag|suitcase|door|cupboard|vehicle|rope|lasso|hook|sledgehammer|pickaxe|"
        r"carry|carrying|lifting|lift|push|pull|heavy|weight|load|throw|catch|ball)(?:_|$)", name
    )
    # 'heavy landing' describes impact, not a carried load.
    if re.fullmatch(r"jump_and_land_heavy(?:_r)?", name):
        return "include", "ground_jump", "Explicit jump/landing; heavy refers to landing"
    if uncertain_context:
        return "review", "context_unclear", "Check support/load context: " + uncertain_context.group(0).strip("_")

    inclusion_rules = (
        (r"^(?:neutral_|loop_)?(?:walk|walking|jog|jogging|run|running)(?:_|$)",
         "free_locomotion", "Explicit walking/jogging/running without a named support or load"),
        (r"^(?:jump_ff|jump_and_land|high_jump|reach_jump)(?:_|$)",
         "ground_jump", "Explicit jump without a named external support or load"),
        (r"^(?:change_|turn_|idle_)?(?:idle_)?crawl(?:_|$)|^(?:push_up|pushup|plank)(?:_|$)",
         "hand_ground_support", "Explicit crawling or floor exercise; hand-ground support is included"),
        (r"^(?:small_|medium_|big_)*light_(?:one_hand|two_hands)(?:_|$)|^drinking_(?:mug|cup)(?:_|$)",
         "light_object", "Explicit light-object handling without a named external body support"),
        (r"^(?:clap(?:_enthusiastic|_forced)?|salute|welcoming|don_t_know(?:_2)?|confusion|"
         r"thinking|checking_time|chefs_kiss|triumph_one_handed|triumph_two_handed)(?:_r)?$",
         "free_gesture", "Explicit free gesture without a named external support or load"),
    )
    for pattern, rule_id, reason in inclusion_rules:
        if re.search(pattern, name):
            return "include", rule_id, reason
    return "review", "not_explicit", "Filename alone does not establish the agreed inclusion criteria"


class MotionReviewStore:
    """Resolve clip overrides, then action decisions, then filename screening."""

    def __init__(self, database_path: Path) -> None:
        self.path = Path(database_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=15)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS motions (
                original_path TEXT PRIMARY KEY,
                action_name TEXT NOT NULL,
                auto_decision TEXT NOT NULL CHECK(auto_decision IN ('include','exclude','review')),
                rule_id TEXT NOT NULL, reason TEXT NOT NULL, rule_version TEXT NOT NULL,
                human_decision TEXT CHECK(human_decision IN ('include','exclude','uncertain','quality_exclude')),
                reviewed_at TEXT, load_error TEXT,
                present INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                original_path TEXT NOT NULL REFERENCES motions(original_path),
                mirrored INTEGER NOT NULL,
                present INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS files_original ON files(original_path);
            CREATE INDEX IF NOT EXISTS motions_action ON motions(action_name,present);
            CREATE TABLE IF NOT EXISTS action_reviews (
                action_name TEXT PRIMARY KEY,
                decision TEXT CHECK(decision IN ('include','exclude','uncertain')),
                reviewed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS action_review_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_name TEXT NOT NULL,
                previous_decision TEXT, decision TEXT, saved_at TEXT NOT NULL,
                origin TEXT NOT NULL, representative_path TEXT
            );
            CREATE TABLE IF NOT EXISTS review_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_path TEXT NOT NULL,
                previous_decision TEXT, decision TEXT, saved_at TEXT NOT NULL
            );
        """)
        self.schema_backup = self._migrate_manual_decisions()
        columns = {r[1] for r in self.connection.execute("PRAGMA table_info(motions)")}
        for name in ("motion_type", "grouping_reason"):
            if name not in columns:
                self.connection.execute(f"ALTER TABLE motions ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        self.connection.execute("UPDATE motions SET motion_type=action_name WHERE motion_type=''")
        self.connection.commit()

    def _migrate_manual_decisions(self) -> Path | None:
        """Back up old schemas and widen manual verdict checks without changing rows."""
        schemas = {}
        checks = {
            "motions": ("human_decision", "'include','exclude','uncertain','quality_exclude'"),
            "action_reviews": ("decision", "'include','exclude','uncertain'"),
        }
        for table, (_, values) in checks.items():
            sql = self.connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
            if any(value not in sql for value in values.split(",")):
                schemas[table] = sql
        if not schemas:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.path.parent / "backups" / f"{self.path.stem}_before_manual_verdict_migration_{stamp}.sqlite3"
        backup.parent.mkdir(parents=True, exist_ok=True)
        destination = sqlite3.connect(backup)
        try:
            self.connection.backup(destination)
        finally:
            destination.close()
        self.connection.execute("PRAGMA foreign_keys = OFF")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            for table, sql in schemas.items():
                objects = self.connection.execute("SELECT sql FROM sqlite_master WHERE tbl_name=? AND type IN ('index','trigger') AND sql IS NOT NULL", (table,)).fetchall()
                column, values = checks[table]
                widened, count = re.subn(
                    rf"CHECK\s*\(\s*{column}\s+IN\s*\([^)]*\)\s*\)",
                    f"CHECK({column} IN ({values}))", sql, flags=re.I)
                if count != 1:
                    raise ValueError(f"Unrecognized manual verdict constraint in {table}; backup: {backup}")
                temporary = table + "_verdict_migration"
                widened = re.sub(r"^(CREATE TABLE\s+)(?:\"?" + table + r"\"?)", r"\1" + temporary, widened, count=1, flags=re.I)
                self.connection.execute(widened)
                self.connection.execute(f"INSERT INTO {temporary} SELECT * FROM {table}")
                self.connection.execute(f"DROP TABLE {table}")
                self.connection.execute(f"ALTER TABLE {temporary} RENAME TO {table}")
                for row in objects:
                    self.connection.execute(row[0])
            if self.connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Review schema migration failed foreign-key validation")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            self.connection.execute("PRAGMA foreign_keys = ON")
        return backup

    def close(self) -> None:
        self.connection.close()

    @property
    def data_root(self) -> Path:
        row = self.connection.execute("SELECT value FROM settings WHERE key='data_root'").fetchone()
        if row is None:
            raise ValueError("Catalog is empty; run python preprocess/motion_review/motion_review_data.py build first")
        return Path(row[0])

    def build(self, data_root: Path, descriptions: dict[str, MotionDescription] | None = None) -> None:
        """Refresh inventory without changing existing grouping or human verdicts."""
        root = Path(data_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        existing = self.connection.execute("SELECT value FROM settings WHERE key='data_root'").fetchone()
        if existing and Path(existing[0]).resolve() != root:
            raise ValueError("This DB belongs to a different data root; use a separate DB")
        paths = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.bvh"))
        if not paths:
            raise ValueError("No BVHs found; existing catalog was not changed")
        present = set(paths)
        originals = sorted({original_key(p) for p in paths})
        missing = [key for key in originals if key not in present]
        if missing:
            raise ValueError(f"Mirrored files lack originals: {missing[:3]}")
        motion_types = {key: motion_name(key) for key in originals}
        groups = build_review_groups(motion_types, descriptions or {})
        old_groups = dict(self.connection.execute("SELECT original_path,action_name FROM motions"))
        if any(key in old_groups and old_groups[key] != groups[key][0] for key in originals):
            raise ValueError("Review grouping changed. Use the explicit reset command after agreeing on the new criteria.")
        rows = []
        for key in originals:
            name = motion_types[key]
            decision, rule_id, reason = classify_motion_name(name)
            group, grouping_reason = groups[key]
            rows.append((key, group, decision, rule_id, reason, RULE_VERSION, name, grouping_reason))
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO settings VALUES ('data_root',?)", (str(root),))
            self.connection.execute("INSERT OR REPLACE INTO settings VALUES ('grouping_version',?)", (GROUPING_VERSION,))
            self.connection.execute("UPDATE motions SET present=0")
            self.connection.execute("UPDATE files SET present=0")
            self.connection.executemany("""
                INSERT INTO motions(original_path,action_name,auto_decision,rule_id,reason,rule_version,motion_type,grouping_reason)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(original_path) DO UPDATE SET
                action_name=excluded.action_name,auto_decision=excluded.auto_decision,
                rule_id=excluded.rule_id,reason=excluded.reason,rule_version=excluded.rule_version,
                motion_type=excluded.motion_type,grouping_reason=excluded.grouping_reason,present=1
            """, rows)
            self.connection.executemany("""
                INSERT INTO files(path,original_path,mirrored) VALUES (?,?,?)
                ON CONFLICT(path) DO UPDATE SET present=1
            """, [(p, original_key(p), int(p != original_key(p))) for p in paths])
            # A merged group's conflicting automatic suggestions require review.
            self.connection.execute("""UPDATE motions SET auto_decision='review',rule_id='mixed_group_evidence',
                reason='Grouped variants have different automatic suggestions; review the behavior context'
                WHERE present=1 AND action_name IN (
                    SELECT action_name FROM motions WHERE present=1 GROUP BY action_name HAVING count(DISTINCT auto_decision)>1
                )""")

    def get(self, original_path: str) -> dict:
        row = self.connection.execute(STATUS_QUERY + " WHERE m.original_path=? AND m.present=1", (original_path,)).fetchone()
        if row is None:
            raise KeyError(original_path)
        result = dict(row)
        result["group_type_count"] = self.connection.execute(
            "SELECT count(DISTINCT motion_type) FROM motions WHERE action_name=? AND present=1",
            (result["action_name"],)).fetchone()[0]
        return result

    def queue(self, selection: str = "pending", action_name: str | None = None) -> list[str]:
        conditions = {
            "pending": "effective_decision='review'",
            "all": "1=1", "auto-include": "auto_decision='include'",
            "auto-exclude": "auto_decision='exclude'",
            "reviewed": "human_decision IS NOT NULL OR action_decision IS NOT NULL",
            "uncertain": "effective_decision='uncertain'",
            "quality-exclude": "effective_decision='quality_exclude'",
        }
        if selection not in conditions:
            raise ValueError(selection)
        query = "SELECT original_path FROM (" + STATUS_QUERY + ") WHERE present=1 AND (" + conditions[selection] + ")"
        parameters = ()
        if action_name is not None:
            query += " AND action_name=?"
            parameters = (action_name,)
        return [r[0] for r in self.connection.execute(query + " ORDER BY action_name,original_path", parameters)]

    def action_summaries(self) -> list[dict]:
        query = """
            SELECT action_name, min(original_path) AS representative_path,
                   count(*) AS original_count, count(DISTINCT motion_type) AS variant_count,
                   CASE WHEN count(DISTINCT auto_decision)=1 THEN min(auto_decision) ELSE 'review' END AS auto_decision,
                   max(action_decision) AS action_decision,
                   sum(human_decision IS NOT NULL) AS clip_override_count,
                   count(DISTINCT human_decision) AS clip_verdict_count,
                   count(DISTINCT CASE WHEN human_decision IN ('include','exclude') THEN human_decision END) AS resolved_clip_verdict_count,
                   sum(effective_decision='review') AS pending_clips,
                   sum(effective_decision='uncertain') AS uncertain_clips,
                   sum(effective_decision='quality_exclude') AS quality_excluded_clips
            FROM (""" + STATUS_QUERY + """ ) WHERE present=1
            GROUP BY action_name ORDER BY action_name
        """
        summaries = [dict(row) for row in self.connection.execute(query)]
        representatives = {}
        for row in self.connection.execute("""SELECT action_name,original_path FROM motions WHERE present=1
                ORDER BY CASE WHEN human_decision='quality_exclude' THEN 1 ELSE 0 END,
                    CASE WHEN instr('_'||motion_type||'_', '_loop_')>0 THEN 0
                    WHEN instr('_'||motion_type||'_', '_start_')>0 OR instr('_'||motion_type||'_', '_stop_')>0 THEN 2
                    ELSE 1 END,original_path"""):
            representatives.setdefault(row["action_name"], row["original_path"])
        for summary in summaries:
            summary["representative_path"] = representatives[summary["action_name"]]
        return summaries

    def action_queue(self, selection: str = "pending", action_name: str | None = None) -> list[str]:
        def selected(row: dict) -> bool:
            if selection == "pending":
                return row["pending_clips"] > 0 or (row["action_decision"] is None and row["resolved_clip_verdict_count"] > 1)
            if selection == "uncertain":
                return row["uncertain_clips"] > 0
            if selection == "quality-exclude":
                return row["quality_excluded_clips"] > 0
            if selection == "all":
                return True
            if selection in ("auto-include", "auto-exclude"):
                return row["auto_decision"] == selection.removeprefix("auto-")
            if selection == "reviewed":
                return row["action_decision"] is not None or row["clip_override_count"] > 0
            raise ValueError(selection)
        return [row["action_name"] for row in self.action_summaries()
                if (action_name is None or row["action_name"] == action_name) and selected(row)]

    def clips_for_action(self, action_name: str, prefer_quality: bool = False) -> list[str]:
        quality = set(self.queue("quality-exclude", action_name))
        def representative_order(path: str) -> tuple[bool, int, str]:
            name = motion_name(path)
            phase = 0 if re.search(r"(?:^|_)loop(?:_|$)", name) else 2 if re.search(r"(?:^|_)(start|stop)(?:_|$)", name) else 1
            return (path in quality) != prefer_quality, phase, path
        return sorted(self.queue("all", action_name), key=representative_order)

    def adopt_legacy_decisions(self) -> dict:
        """Once, back up and adopt unanimous old clip labels as action defaults.

        Keep the old clip labels and events intact. Conflicting labels remain
        unresolved at action level and appear in the pending action queue.
        """
        marker = "action_review_migrated_v1"
        if self.connection.execute("SELECT 1 FROM settings WHERE key=?", (marker,)).fetchone():
            return {"adopted_actions": 0, "backup": None}
        backup = None
        if self.connection.execute("SELECT 1 FROM motions WHERE human_decision IS NOT NULL LIMIT 1").fetchone():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = self.path.parent / "backups" / f"{self.path.stem}_before_action_review_{stamp}.sqlite3"
            backup.parent.mkdir(parents=True, exist_ok=True)
            destination = sqlite3.connect(backup)
            try:
                self.connection.backup(destination)
            finally:
                destination.close()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connection:
            rows = self.connection.execute("""
                SELECT action_name,min(human_decision) AS decision
                FROM motions WHERE present=1 AND human_decision IN ('include','exclude','uncertain')
                AND action_name NOT IN (SELECT action_name FROM action_reviews)
                GROUP BY action_name HAVING count(DISTINCT human_decision)=1
            """).fetchall()
            for row in rows:
                self.connection.execute("INSERT INTO action_reviews VALUES (?,?,?)", (row[0], row[1], now))
                self.connection.execute("""INSERT INTO action_review_events
                    (action_name,previous_decision,decision,saved_at,origin) VALUES (?,NULL,?,?,'legacy')""",
                    (row[0], row[1], now))
            self.connection.execute("INSERT INTO settings VALUES (?,?)", (marker, now))
        return {"adopted_actions": len(rows), "backup": str(backup) if backup else None}

    def save_action_decision(self, action_name: str, decision: str | None,
                             representative_path: str | None = None) -> None:
        """O/X/U replaces semantic clip overrides; quality exclusions remain local.

        Clearing only the action verdict leaves explicit clip overrides intact.
        All replaced clip verdicts remain in the existing event history.
        """
        if decision not in (None, "include", "exclude", "uncertain"):
            raise ValueError(decision)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connection:
            # Reserve the write transaction before reading the previous state.
            self.connection.execute("UPDATE motions SET present=present WHERE action_name=? AND present=1", (action_name,))
            members = self.clips_for_action(action_name)
            if not members:
                raise KeyError(action_name)
            if representative_path is not None and representative_path not in members:
                raise ValueError("Representative does not belong to this action")
            previous = self.connection.execute("SELECT decision FROM action_reviews WHERE action_name=?", (action_name,)).fetchone()
            self.connection.execute("""INSERT INTO action_reviews VALUES (?,?,?)
                ON CONFLICT(action_name) DO UPDATE SET decision=excluded.decision,reviewed_at=excluded.reviewed_at""",
                (action_name, decision, now))
            self.connection.execute("""INSERT INTO action_review_events
                (action_name,previous_decision,decision,saved_at,origin,representative_path)
                VALUES (?,?,?,?,'explicit',?)""", (action_name, previous[0] if previous else None, decision, now, representative_path))
            if decision is not None:
                self.connection.execute("""INSERT INTO review_events(original_path,previous_decision,decision,saved_at)
                    SELECT original_path,human_decision,NULL,? FROM motions
                    WHERE action_name=? AND present=1 AND human_decision IN ('include','exclude','uncertain')""", (now, action_name))
                self.connection.execute("""UPDATE motions SET human_decision=NULL,reviewed_at=NULL
                    WHERE action_name=? AND present=1 AND human_decision IN ('include','exclude','uncertain')""", (action_name,))

    def save_decision(self, original_path: str, decision: str | None) -> None:
        """Commit before the UI advances; None clears only the human override."""
        if decision not in (None, "include", "exclude", "uncertain", "quality_exclude"):
            raise ValueError(decision)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connection:
            current = self.get(original_path)
            self.connection.execute("UPDATE motions SET human_decision=?,reviewed_at=? WHERE original_path=?",
                                    (decision, now, original_path))
            self.connection.execute("INSERT INTO review_events(original_path,previous_decision,decision,saved_at) VALUES (?,?,?,?)",
                                    (original_path, current["human_decision"], decision, now))

    def set_load_error(self, original_path: str, message: str | None) -> None:
        with self.connection:
            self.connection.execute("UPDATE motions SET load_error=? WHERE original_path=?", (message, original_path))

    def file_path(self, original_path: str) -> Path:
        path = (self.data_root / original_path).resolve()
        path.relative_to(self.data_root.resolve())
        return path

    def export(self, output_directory: Path | None = None) -> dict:
        """Write current lists atomically. The DB remains the authoritative record."""
        destination = Path(output_directory or self.path.parent)
        destination.mkdir(parents=True, exist_ok=True)
        motions = [dict(r) for r in self.connection.execute(STATUS_QUERY + " WHERE m.present=1 ORDER BY m.action_name,m.original_path")]
        variants: dict[str, list[str]] = {}
        for row in self.connection.execute("SELECT path,original_path FROM files WHERE present=1 ORDER BY path"):
            variants.setdefault(row["original_path"], []).append(row["path"])
        lists = {name: [] for name in ("included", "excluded", "uncertain", "quality_excluded", "needs_review", "auto_include", "auto_exclude")}
        counts = Counter()
        for row in motions:
            key = row["original_path"]
            final = row["effective_decision"]
            row["mirror_paths"] = " | ".join(p for p in variants[key] if p != key)
            counts["groups_" + final] += 1
            counts["files_" + final] += len(variants[key])
            if row["human_decision"]:
                counts["human_reviewed"] += 1
            if row["auto_decision"] != "review":
                lists["auto_" + row["auto_decision"]].append(key)
            if final == "review":
                lists["needs_review"].append(key)
            else:
                lists[{"include": "included", "exclude": "excluded", "uncertain": "uncertain", "quality_exclude": "quality_excluded"}[final]].extend(variants[key])
        for name, values in lists.items():
            target = destination / f"{name}.txt"
            temporary = target.with_suffix(".txt.tmp")
            temporary.write_text("".join(p + "\n" for p in values), encoding="utf-8")
            temporary.replace(target)
        columns = ["original_path", "mirror_paths", "action_name", "motion_type", "grouping_reason", "auto_decision", "rule_id", "reason",
                   "human_decision", "action_decision", "decision_source", "effective_decision", "reviewed_at", "load_error", "rule_version"]
        csv_path = destination / "review_catalog.csv"
        with csv_path.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(motions)
        csv_path.with_suffix(".csv.tmp").replace(csv_path)
        actions = self.action_summaries()
        pending_actions = self.action_queue()
        pending_path = destination / "needs_review_types.txt"
        pending_path.with_suffix(".txt.tmp").write_text("".join(name + "\n" for name in pending_actions), encoding="utf-8")
        pending_path.with_suffix(".txt.tmp").replace(pending_path)
        uncertain_actions = self.action_queue("uncertain")
        uncertain_path = destination / "uncertain_types.txt"
        uncertain_path.with_suffix(".txt.tmp").write_text("".join(name + "\n" for name in uncertain_actions), encoding="utf-8")
        uncertain_path.with_suffix(".txt.tmp").replace(uncertain_path)
        action_path = destination / "action_review_catalog.csv"
        with action_path.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, ["action_name", "representative_path", "original_count", "variant_count", "auto_decision",
                                           "action_decision", "clip_override_count", "clip_verdict_count", "resolved_clip_verdict_count",
                                           "pending_clips", "uncertain_clips", "quality_excluded_clips"])
            writer.writeheader()
            writer.writerows(actions)
        action_path.with_suffix(".csv.tmp").replace(action_path)
        grouping_path = destination / "review_groups.csv"
        with grouping_path.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["review_group", "motion_type", "original_count", "grouping_reason"])
            writer.writerows(self.connection.execute("""SELECT action_name,motion_type,count(*),grouping_reason
                FROM motions WHERE present=1 GROUP BY action_name,motion_type,grouping_reason ORDER BY action_name,motion_type"""))
        grouping_path.with_suffix(".csv.tmp").replace(grouping_path)
        summary = {"data_root": str(self.data_root), "rule_version": RULE_VERSION,
                   "grouping_version": GROUPING_VERSION,
                   "source": "filename eligibility suggestions; behavior groups checked against official prop/body/load context",
                   "original_groups": len(motions), "files": sum(len(v) for v in variants.values()),
                   "action_types": len(actions), "pending_action_types": len(pending_actions),
                   "uncertain_action_types": len(uncertain_actions),
                   "quality_excluded_action_types": len(self.action_queue("quality-exclude")),
                   "reviewed_action_types": sum(row["action_decision"] is not None for row in actions), **dict(counts)}
        summary_path = destination / "summary.json"
        summary_path.with_suffix(".json.tmp").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_path.with_suffix(".json.tmp").replace(summary_path)
        return summary


def reset_review_catalog(database_path: Path, data_root: Path,
                         descriptions: dict[str, MotionDescription]) -> Path | None:
    """Validate a fresh catalog, back up the old DB, then replace without adoption."""
    database_path = Path(database_path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    staging = database_path.with_name(f"{database_path.stem}_reset_{stamp}.sqlite3")
    fresh = MotionReviewStore(staging)
    try:
        fresh.build(data_root, descriptions)
        fresh.adopt_legacy_decisions()  # Mark migration complete with zero inherited labels.
        if fresh.connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Fresh catalog failed integrity validation; active DB unchanged")
        if fresh.connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Fresh catalog has invalid file references; active DB unchanged")
    except Exception:
        fresh.close()
        staging.unlink(missing_ok=True)
        raise
    fresh.close()
    backup = None
    try:
        if database_path.exists():
            # Windows denies this probe while another viewer holds the SQLite file.
            # Close the viewer rather than resetting underneath its cached queue.
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                              wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
                kernel.CreateFileW.restype = wintypes.HANDLE
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.CreateFileW(str(database_path), 0x80000000, 0, None, 3, 0, None)
                if handle == wintypes.HANDLE(-1).value:
                    error = ctypes.get_last_error()
                    if error in (32, 33):
                        raise PermissionError("Close every motion-review viewer and database tool before resetting its database")
                    raise ctypes.WinError(error)
                kernel.CloseHandle(handle)
            backup = database_path.parent / "backups" / f"{database_path.stem}_before_group_reset_{stamp}.sqlite3"
            backup.parent.mkdir(parents=True, exist_ok=True)
            source = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
            destination = sqlite3.connect(backup)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
        staging.replace(database_path)
    finally:
        staging.unlink(missing_ok=True)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "export", "reset"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--db", type=Path, default=DEFAULT_REVIEW_DIR / "review.sqlite3")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    descriptions = load_motion_descriptions(args.metadata) if args.command in ("build", "reset") else {}
    if args.command == "reset":
        backup = reset_review_catalog(args.db, args.data_root, descriptions)
        print(f"Fresh review catalog installed. Previous DB backup: {backup}")
    store = MotionReviewStore(args.db)
    try:
        if store.schema_backup:
            print(f"Manual verdict schema updated. Backup: {store.schema_backup}")
        if args.command == "build":
            store.build(args.data_root, descriptions)
        migration = store.adopt_legacy_decisions()
        if migration["backup"]:
            print(json.dumps(migration, indent=2))
        print(json.dumps(store.export(), indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
