"""Headless video-to-Signal supervisor; heavy dependencies live in a child process."""

import argparse
import fcntl
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time

from monitor_config import ConfigError, load_config
from monitor_store import Store, file_key


LOG = logging.getLogger("video_monitor")


def retry_delay(config, attempts):
    return min(config.retry.maximum_seconds,
               config.retry.initial_seconds * 2 ** min(max(attempts - 1, 0), 20))


def expand_command(command, **values):
    # Replace whole arguments only. Filenames/messages never become shell code.
    expanded = []
    for arg in command:
        if arg == "{image}" and not values.get("{image}"):
            if expanded and expanded[-1] in ("-a", "--attachment"):
                expanded.pop()
            continue
        expanded.append(values.get(arg, arg))
    return expanded


def scan_files(root, recursive):
    """Stream directory entries; do not follow symlinks or collect the whole share."""
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                continue
            if recursive and entry.is_dir(follow_symlinks=False):
                yield from scan_files(entry.path, True)
            elif entry.name.lower().endswith(".mp4") and entry.is_file(follow_symlinks=False):
                try:
                    stat = entry.stat(follow_symlinks=False)
                    if stat.st_size:
                        yield entry.path, stat.st_size, stat.st_mtime_ns
                except FileNotFoundError:
                    continue


class Monitor:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.watch.path)
        self.state = Path(config.storage.path)
        self.spool = self.state / "spool"
        self.spool.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.state / "queue.sqlite3")
        self.store.recover()
        self.stop = threading.Event()
        self.active = None
        self.mount_ok = False
        self.next_mount = 0
        self.next_scan = 0

    def launch(self, kind, command, timeout, row=None):
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True,
                                   cwd=Path(__file__).resolve().parent)
        self.active = (kind, process, time.monotonic() + timeout, row)

    @staticmethod
    def terminate(process):
        # Own process group, including helpers spawned by the configured command.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # A helper might remain even when the group leader has exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def scan(self, now):
        self.next_scan = time.monotonic() + self.config.watch.scan_seconds
        if not self.mount_ok or not os.path.ismount(self.root):
            return
        baseline = self.store.needs_initial_baseline()
        existing_count = 0
        try:
            for path, size, mtime in scan_files(self.root, self.config.watch.recursive):
                if baseline:
                    self.store.ignore_existing(path, size, mtime, now)
                    existing_count += 1
                else:
                    self.store.observe(path, size, mtime, now, self.config.watch.stable_seconds)
        except OSError:
            self.mount_ok = False
            LOG.exception("Cannot scan %s; waiting for remount", self.root)
            return
        if baseline:
            self.store.complete_initial_baseline()
            LOG.info("Initial inventory complete; ignoring %d existing MP4 files", existing_count)

    def job_directory(self, row):
        return self.spool / row["id"]

    def discard_images(self, row):
        directory = self.job_directory(row)
        if directory.exists():
            shutil.rmtree(directory)

    def analysis_failed(self, row, error):
        attempts = row["attempts"] + 1
        failed = attempts >= self.config.analysis.max_attempts
        self.store.defer(row["id"], time.time() + retry_delay(self.config, attempts), error, failed)
        self.discard_images(row)
        LOG.error("Analysis %s for %s: %s", "failed permanently" if failed else "will retry",
                  row["path"], error)

    def complete(self, kind, row, code, timed_out):
        error = "command timed out" if timed_out else f"command exited with status {code}"
        if kind == "mount":
            self.mount_ok = code == 0 and not timed_out and os.path.ismount(self.root)
            self.next_mount = time.monotonic() + self.config.mount.interval_seconds
            if self.mount_ok:
                LOG.info("Mount refreshed: %s", self.root)
                self.next_scan = 0
            else:
                LOG.error("Mount unavailable at %s (%s)", self.root, error)
        elif kind == "send":
            if code == 0 and not timed_out:
                self.store.delivered(row)
                LOG.info("Delivered %s for job %s", row["object_name"], row["job_id"])
                self.cleanup_sent()
            else:
                self.store.delivery_failed(row, time.time() + retry_delay(
                    self.config, row["attempts"] + 1), error)
                LOG.error("Signal delivery will retry for %s: %s", row["job_id"], error)
        elif kind == "analyze":
            if code != 0 or timed_out:
                self.analysis_failed(row, error)
                return
            try:
                stat = Path(row["path"]).stat()
                if file_key(row["path"], stat.st_size, stat.st_mtime_ns) != row["id"]:
                    self.store.supersede(row["id"])
                    self.discard_images(row)
                    LOG.warning("Video changed during analysis: %s", row["path"])
                    return
                result_path = self.job_directory(row) / "result.json"
                if result_path.stat().st_size > 1024 * 1024:
                    raise ValueError("Analysis result is too large")
                result = json.loads(result_path.read_text())
                images = result["annotated_paths"]
                if not isinstance(images, dict) or set(images) != set(result["detected_objects"]):
                    raise ValueError("Missing object images in analysis result")
                image_root = self.job_directory(row).resolve()
                for name, image in images.items():
                    if not isinstance(name, str) or not isinstance(image, str):
                        raise ValueError("Invalid object image")
                    path = Path(image).resolve()
                    if path.parent != image_root or not path.is_file() or path.stat().st_size == 0:
                        raise ValueError("Missing or invalid image file")
                self.store.finish(row["id"], images, Path(row["path"]).name)
                result_path.unlink(missing_ok=True)
                if not images:
                    self.discard_images(row)
                LOG.info("Analyzed %s: %d object classes", row["path"], len(images))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.analysis_failed(row, str(exc))

    def poll(self):
        if self.active is None:
            return
        kind, process, deadline, row = self.active
        code = process.poll()
        timed_out = time.monotonic() >= deadline and code is None
        if timed_out:
            self.terminate(process)
            code = process.returncode
        if code is not None:
            self.active = None
            self.complete(kind, row, code, timed_out)

    def cleanup_sent(self):
        # Keep delivery records for deduplication, but reclaim successfully sent JPGs.
        rows = self.store.db.execute("SELECT * FROM deliveries WHERE status='sent'")
        for row in rows:
            if not row["image"]:
                continue
            path = Path(row["image"])
            try:
                if self.spool.resolve() in path.resolve().parents:
                    path.unlink(missing_ok=True)
                    if path.parent.exists() and not any(path.parent.iterdir()):
                        path.parent.rmdir()
                    with self.store.db:
                        self.store.db.execute("""UPDATE deliveries SET status='cleaned'
                            WHERE job_id=? AND object_name=?""", (row["job_id"], row["object_name"]))
            except OSError:
                LOG.exception("Could not remove delivered image: %s", path)

    def spool_size(self):
        return sum(path.stat().st_size for path in self.spool.rglob("*") if path.is_file())

    def start_analysis(self, row):
        try:
            stat = Path(row["path"]).stat()
            if file_key(row["path"], stat.st_size, stat.st_mtime_ns) != row["id"]:
                self.store.supersede(row["id"])
                return
        except OSError as exc:
            # Disconnection does not consume the corrupt-video attempt budget.
            self.store.defer(row["id"], time.time() + self.config.retry.initial_seconds, str(exc))
            LOG.warning("Video unavailable: %s", row["path"])
            return
        self.discard_images(row)
        directory = self.job_directory(row)
        directory.mkdir()
        cfg = self.config.analysis
        command = [sys.executable, "-u", str(Path(__file__).with_name("video_analyzer.py")),
                   row["path"], "--no-isolation", "--model", cfg.model,
                   "--annotated-dir", str(directory), "--result-json", str(directory / "result.json")]
        for key in ("confidence", "min_object_hits", "max_bad_frames", "threads", "iou"):
            command.extend(["--" + key.replace("_", "-"), str(getattr(cfg, key))])
        if cfg.data:
            command.extend(["--data", cfg.data])
        self.store.start(row["id"])
        try:
            self.launch("analyze", command, cfg.timeout_seconds, row)
        except OSError as exc:
            self.analysis_failed(row, str(exc))

    def tick(self):
        self.poll()
        now = time.monotonic()
        # Never scan while the mount is being replaced.
        if now >= self.next_scan and (not self.active or self.active[0] != "mount"):
            self.scan(time.time())
        if self.active:
            return
        if now >= self.next_mount:
            self.mount_ok = False
            try:
                command = expand_command(self.config.mount.remount_command,
                                         **{"{mountpoint}": str(self.root)})
                self.launch("mount", command, self.config.mount.timeout_seconds)
            except OSError as exc:
                self.next_mount = now + self.config.mount.interval_seconds
                LOG.error("Cannot run remount command: %s", exc)
            return
        delivery = self.store.next_delivery(time.time())
        if delivery is not None:
            try:
                if delivery["image"] and not Path(delivery["image"]).is_file():
                    raise OSError("Pending attachment is missing")
                command = expand_command(self.config.signal.command, **{
                    "{image}": delivery["image"], "{message}": delivery["message"],
                    "{group_id}": self.config.signal.group_id})
                self.launch("send", command, self.config.signal.timeout_seconds, delivery)
            except OSError as exc:
                self.store.delivery_failed(delivery, time.time() + retry_delay(
                    self.config, delivery["attempts"] + 1), str(exc))
                LOG.error("Cannot send attachment: %s", exc)
            return
        if self.mount_ok and os.path.ismount(self.root):
            row = self.store.next_job(time.time())
            if row is not None:
                if self.spool_size() < self.config.storage.max_spool_mb * 1024 * 1024:
                    self.start_analysis(row)

    def run(self):
        LOG.info("Watching %s", self.root)
        self.cleanup_sent()
        try:
            while not self.stop.is_set():
                try:
                    self.tick()
                except (OSError, sqlite3.Error):
                    LOG.exception("I/O failure; will retry")
                    self.stop.wait(self.config.retry.initial_seconds)
                self.stop.wait(0.5)
        finally:
            if self.active:
                self.terminate(self.active[1])
            self.store.close()
            LOG.info("Monitor stopped")


def check_runtime(config):
    backend = "ultralytics" if Path(config.analysis.model).suffix.lower() == ".pt" else "tflite_detector"
    for name in ("cv2", "numpy", backend):
        if importlib.util.find_spec(name) is None:
            raise ConfigError(f"Missing analysis dependency: {name}")
    for label, value in (("model", config.analysis.model), ("data", config.analysis.data)):
        if value and not Path(value).is_file():
            raise ConfigError(f"Analysis {label} file not found: {value}")
    for label, command in (("mount", config.mount.remount_command), ("signal", config.signal.command)):
        if shutil.which(command[0]) is None:
            raise ConfigError(f"{label} executable not found: {command[0]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--check-config", action="store_true", help="Validate YAML only; run no commands")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.check_config:
            LOG.info("Configuration is valid")
            return 0
        check_runtime(config)
        state = Path(config.storage.path)
        state.mkdir(parents=True, exist_ok=True)
        with (state / "monitor.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConfigError("Another monitor already uses this state directory") from exc
            monitor = Monitor(config)
            for number in (signal.SIGTERM, signal.SIGINT):
                signal.signal(number, lambda *_: monitor.stop.set())
            monitor.run()
        return 0
    except (ConfigError, OSError, sqlite3.Error) as exc:
        LOG.error("Cannot start monitor: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
