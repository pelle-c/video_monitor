"""Small, disk-backed queue. Only the supervisor accesses this database."""

import hashlib
import sqlite3


def file_key(path, size, mtime_ns):
    return hashlib.sha256(f"{path}\0{size}\0{mtime_ns}".encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, path TEXT NOT NULL, size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL, observed REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'waiting', attempts INTEGER NOT NULL DEFAULT 0,
                next_try REAL NOT NULL DEFAULT 0, error TEXT
            );
            CREATE INDEX IF NOT EXISTS jobs_due ON jobs(status, next_try);
            CREATE INDEX IF NOT EXISTS jobs_path ON jobs(path);
            CREATE TABLE IF NOT EXISTS deliveries (
                job_id TEXT NOT NULL REFERENCES jobs(id), object_name TEXT NOT NULL,
                image TEXT NOT NULL, message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                next_try REAL NOT NULL DEFAULT 0, error TEXT,
                PRIMARY KEY(job_id, object_name)
            );
            CREATE INDEX IF NOT EXISTS deliveries_due ON deliveries(status, next_try);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
        """)

    def recover(self):
        with self.db:
            self.db.execute("UPDATE jobs SET status='ready' WHERE status='analyzing'")

    def needs_initial_baseline(self):
        row = self.db.execute("SELECT value FROM metadata WHERE key='initial_baseline'").fetchone()
        return row is None or row["value"] != "complete"

    def complete_initial_baseline(self):
        with self.db:
            self.db.execute("""INSERT INTO metadata(key,value) VALUES('initial_baseline','complete')
                ON CONFLICT(key) DO UPDATE SET value='complete'""")

    def ignore_existing(self, path, size, mtime_ns, now):
        key = file_key(path, size, mtime_ns)
        with self.db:
            self.db.execute("""UPDATE jobs SET status='ignored'
                WHERE path=? AND status IN ('waiting','ready')""", (path,))
            self.db.execute("""INSERT OR IGNORE INTO jobs
                (id,path,size,mtime_ns,observed,status) VALUES (?,?,?,?,?,'ignored')""",
                            (key, path, size, mtime_ns, now))
        return key

    def observe(self, path, size, mtime_ns, now, stable_seconds):
        key = file_key(path, size, mtime_ns)
        with self.db:
            self.db.execute("""UPDATE jobs SET status='superseded'
                WHERE path=? AND id<>? AND status IN ('waiting','ready')""", (path, key))
            self.db.execute("""INSERT OR IGNORE INTO jobs
                (id,path,size,mtime_ns,observed) VALUES (?,?,?,?,?)""",
                            (key, path, size, mtime_ns, now))
            self.db.execute("""UPDATE jobs SET status='ready'
                WHERE id=? AND status='waiting' AND observed<=?""",
                            (key, now - stable_seconds))
        return key

    def next_job(self, now):
        return self.db.execute("""SELECT * FROM jobs WHERE status='ready'
            AND next_try<=? ORDER BY observed,id LIMIT 1""", (now,)).fetchone()

    def start(self, key):
        with self.db:
            self.db.execute("UPDATE jobs SET status='analyzing',attempts=attempts+1 WHERE id=?", (key,))

    def defer(self, key, next_try, error, failed=False):
        with self.db:
            self.db.execute("UPDATE jobs SET status=?,next_try=?,error=? WHERE id=?",
                            ("failed" if failed else "ready", next_try, error, key))

    def supersede(self, key):
        with self.db:
            self.db.execute("UPDATE jobs SET status='superseded' WHERE id=?", (key,))

    def finish(self, key, images, video_name):
        # Atomic: either all delivery records exist or analysis remains unfinished.
        with self.db:
            if images:
                for name, image in images.items():
                    self.db.execute("""INSERT OR IGNORE INTO deliveries
                        (job_id,object_name,image,message) VALUES (?,?,?,?)""",
                                    (key, name, image, f"{name} — {video_name}"))
            else:
                self.db.execute("""INSERT OR IGNORE INTO deliveries
                    (job_id,object_name,image,message) VALUES (?,?,?,?)""",
                                (key, "no_objects", "", f"Inga objekt identifierades i {video_name}"))
            self.db.execute("UPDATE jobs SET status='done',error=NULL WHERE id=?", (key,))

    def next_delivery(self, now):
        return self.db.execute("""SELECT * FROM deliveries WHERE status='pending'
            AND next_try<=? ORDER BY next_try,job_id,object_name LIMIT 1""", (now,)).fetchone()

    def delivered(self, row):
        with self.db:
            self.db.execute("""UPDATE deliveries SET status='sent',error=NULL
                WHERE job_id=? AND object_name=?""", (row["job_id"], row["object_name"]))

    def delivery_failed(self, row, next_try, error):
        with self.db:
            self.db.execute("""UPDATE deliveries SET attempts=attempts+1,next_try=?,error=?
                WHERE job_id=? AND object_name=?""",
                            (next_try, error, row["job_id"], row["object_name"]))

    def close(self):
        self.db.close()
