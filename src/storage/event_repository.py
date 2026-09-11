"""SQLite persistence using the canonical ViolationEvent schema."""

from dataclasses import asdict
from pathlib import Path
import sqlite3

class EventRepository:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        try:
            with self.connection:
                self.connection.execute("""CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL, video_id TEXT NOT NULL, camera_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL, zone_type TEXT NOT NULL, track_id INTEGER NOT NULL,
                    object_class TEXT NOT NULL, entered_at REAL NOT NULL,
                    stationary_since REAL NOT NULL, violation_at REAL NOT NULL, left_at REAL,
                    dwell_time_sec REAL NOT NULL, inside_frame_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL,
                    snapshot_path TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('OPEN','CLOSED')),
                    config_version TEXT NOT NULL, model_version TEXT NOT NULL
                )""")
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_run_time "
                    "ON events(run_id, violation_at)"
                )
                columns = {row[1] for row in self.connection.execute("PRAGMA table_info(events)")}
                if "inside_frame_count" not in columns:
                    self.connection.execute(
                        "ALTER TABLE events ADD COLUMN inside_frame_count INTEGER NOT NULL DEFAULT 0"
                    )
        except Exception:
            self.connection.close()
            raise

    def create(self, event):
        evidence = Path(event.snapshot_path)
        if not evidence.is_file() or evidence.stat().st_size == 0:
            raise ValueError("A successfully saved snapshot is required before event insertion")
        values = asdict(event)
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with self.connection:
            self.connection.execute(f"INSERT INTO events ({columns}) VALUES ({placeholders})", tuple(values.values()))

    def get(self, event_id):
        row = self.connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return dict(row) if row else None

    def list_by_run(self, run_id):
        """Return one run's events in video-time order for reports and the web UI."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        rows = self.connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY violation_at, event_id",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close_event(self, event_id, left_at):
        with self.connection:
            result = self.connection.execute("UPDATE events SET status='CLOSED', left_at=? WHERE event_id=?",
                                             (left_at, event_id))
            if result.rowcount != 1:
                raise ValueError(f"Event not found: {event_id}")

    def close(self):
        self.connection.close()
