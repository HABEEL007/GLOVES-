import os
import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..config import config
from .models import (
    CameraRecord, StaffTrackRecord, DetectionRecord,
    HandStateRecord, ViolationEventRecord, EvidenceRecord,
    ModelVersionRecord
)

class DatabaseService:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.SQLITE_PATH
        self.lock = threading.Lock()
        self._init_sqlite()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        with self.lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            
            # Cameras
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                stream_url TEXT NOT NULL,
                location TEXT,
                is_active INTEGER DEFAULT 1
            );
            """)

            # Staff Tracks
            cur.execute("""
            CREATE TABLE IF NOT EXISTS staff_tracks (
                track_id INTEGER PRIMARY KEY,
                camera_id TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                is_staff INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1
            );
            """)

            # Detection Records
            cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_records (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_number INTEGER,
                camera_id TEXT NOT NULL,
                track_id INTEGER,
                class_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                assigned_side TEXT NOT NULL,
                association_confidence REAL,
                wrist_distance REAL,
                timestamp TEXT NOT NULL
            );
            """)

            # Hand State Records
            cur.execute("""
            CREATE TABLE IF NOT EXISTS hand_state_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                side TEXT NOT NULL,
                state TEXT NOT NULL,
                confidence REAL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                stable_frames INTEGER,
                bare_duration REAL,
                last_transition TEXT NOT NULL
            );
            """)

            # Violation Events
            cur.execute("""
            CREATE TABLE IF NOT EXISTS violation_events (
                violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                confidence REAL NOT NULL,
                violation_type TEXT NOT NULL,
                evidence_path TEXT,
                model_version TEXT,
                status TEXT DEFAULT 'CONFIRMED'
            );
            """)

            # Evidence Snapshots
            cur.execute("""
            CREATE TABLE IF NOT EXISTS evidence_snapshots (
                evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(violation_id) REFERENCES violation_events(violation_id)
            );
            """)

            # Model Versions
            cur.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_version TEXT PRIMARY KEY,
                training_run_id TEXT,
                architecture TEXT,
                image_size INTEGER,
                epochs INTEGER,
                batch_size INTEGER,
                precision REAL,
                recall REAL,
                map50 REAL,
                map50_95 REAL,
                f1 REAL,
                right_hand_acc REAL,
                false_violation_rate REAL,
                created_at TEXT NOT NULL,
                approved_at TEXT,
                is_champion INTEGER DEFAULT 0
            );
            """)

            # Ensure default camera exists
            cur.execute("INSERT OR IGNORE INTO cameras (camera_id, name, stream_url, location) VALUES (?, ?, ?, ?)",
                        ("301", "Gold Counter Camera 301", config.DEFAULT_RTSP_URL, "Main Showcase"))

            conn.commit()
            conn.close()

    def record_track(self, track_id: int, camera_id: str = "301", is_staff: bool = True):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO staff_tracks (track_id, camera_id, first_seen, last_seen, is_staff, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(track_id) DO UPDATE SET last_seen = excluded.last_seen, is_active = 1
            """, (track_id, camera_id, now, now, 1 if is_staff else 0))
            conn.commit()
            conn.close()

    def record_detections(self, detections: List[Dict[str, Any]], frame_number: int, camera_id: str = "301"):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_connection()
            cur = conn.cursor()
            rows = []
            for d in detections:
                box = d.get("box", [0, 0, 0, 0])
                rows.append((
                    frame_number,
                    camera_id,
                    d.get("track_id"),
                    d.get("class_id", 0),
                    d.get("conf", 0.0),
                    box[0], box[1], box[2], box[3],
                    d.get("side", "UNCERTAIN"),
                    d.get("association_confidence", 0.0),
                    d.get("wrist_distance", 0.0),
                    now
                ))
            cur.executemany("""
            INSERT INTO detection_records (
                frame_number, camera_id, track_id, class_id, confidence,
                x1, y1, x2, y2, assigned_side, association_confidence,
                wrist_distance, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()

    def record_hand_state(self, track_id: int, side: str, state: str, conf: float, stable_frames: int, duration: float):
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO hand_state_records (
                track_id, side, state, confidence, first_seen, last_seen, stable_frames, bare_duration, last_transition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (track_id, side, state, conf, now, now, stable_frames, duration, now))
            conn.commit()
            conn.close()

    def create_violation(self, camera_id: str, track_id: int, conf: float, evidence_path: str, model_ver: str) -> int:
        now = datetime.now().isoformat()
        with self.lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO violation_events (
                camera_id, track_id, start_time, confidence, violation_type, evidence_path, model_version, status
            ) VALUES (?, ?, ?, ?, 'BARE_RIGHT_HAND', ?, ?, 'CONFIRMED')
            """, (camera_id, track_id, now, conf, evidence_path, model_ver))
            v_id = cur.lastrowid
            
            # Evidence record
            cur.execute("""
            INSERT INTO evidence_snapshots (
                violation_id, image_path, width, height, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (v_id, evidence_path, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT, json.dumps({"track_id": track_id, "conf": conf}), now))
            
            conn.commit()
            conn.close()
            return v_id

    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
            SELECT violation_id, camera_id, track_id, start_time, confidence, violation_type, evidence_path, model_version, status
            FROM violation_events
            ORDER BY violation_id DESC
            LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows

    def register_model_version(self, record: ModelVersionRecord):
        with self.lock:
            conn = self._get_connection()
            cur = conn.cursor()
            if record.is_champion:
                cur.execute("UPDATE model_versions SET is_champion = 0")
            cur.execute("""
            INSERT OR REPLACE INTO model_versions (
                model_version, training_run_id, architecture, image_size, epochs, batch_size,
                precision, recall, map50, map50_95, f1, right_hand_acc, false_violation_rate,
                created_at, approved_at, is_champion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.model_version, record.training_run_id, record.architecture,
                record.image_size, record.epochs, record.batch_size,
                record.precision, record.recall, record.map50, record.map50_95,
                record.f1, record.right_hand_acc, record.false_violation_rate,
                record.created_at, record.approved_at, 1 if record.is_champion else 0
            ))
            conn.commit()
            conn.close()

db_service = DatabaseService()
