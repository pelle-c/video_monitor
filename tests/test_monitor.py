"""Run on the Pi: python3 -m unittest discover -s tests -v.

No real mounting, inference or Signal messages are performed by these tests.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

from monitor_config import (Analysis, Config, ConfigError, Mount, Retry, Signal,
                            Storage, Watch, load_config)
from monitor_store import Store
from video_monitor import Monitor, expand_command, retry_delay, scan_files


PROJECT = Path(__file__).resolve().parents[1]


class TemporaryProject(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".tmp-", dir=PROJECT / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.watch = self.root / "videos"
        self.watch.mkdir()
        self.config = Config(Watch(str(self.watch)), Mount(["fake-remount"]),
                             Analysis(), Signal(["fake-signal", "{image}", "{message}"]),
                             Storage(str(self.root / "state")), Retry())

    def monitor(self):
        monitor = Monitor(self.config)
        self.addCleanup(monitor.store.close)
        return monitor

    def ready_job(self, monitor):
        video = self.watch / "test.mp4"
        video.write_bytes(b"test video")
        stat = video.stat()
        now = time.time()
        args = (str(video), stat.st_size, stat.st_mtime_ns)
        key = monitor.store.observe(*args, now - 40, 30)
        monitor.store.observe(*args, now, 30)
        return key, monitor.store.next_job(now)


class ConfigTests(TemporaryProject):
    def load(self, edit=None):
        raw = yaml.safe_load((PROJECT / "config.example.yaml").read_text())
        if edit:
            edit(raw)
        source = self.root / "config.yaml"
        source.write_text(yaml.safe_dump(raw))
        return load_config(source)

    def test_example_and_relative_paths(self):
        config = self.load()
        self.assertEqual(config.watch.path, "/mnt/video")
        self.assertEqual(config.mount.interval_seconds, 60)
        self.assertEqual(config.signal.group_id, "GnZ32dWODDI1g1RpbMGUTBmM1o9NdI8JPD6nZ5uo/p8=")
        self.assertEqual(config.analysis.threads, 1)
        self.assertEqual(config.storage.path, str(self.root / "state"))
        self.assertEqual(config.analysis.model, str(self.root / "runs/detect/train-8/weights/best.pt"))

    def test_invalid_values(self):
        cases = [("watch", "recursive", "false"), ("watch", "scan_seconds", 0),
                 ("analysis", "confidence", float("nan")), ("analysis", "threads", True),
                 ("analysis", "iou", 1.1), ("signal", "command", ["send"]),
                 ("signal", "group_id", ""),
                 ("mount", "remount_command", []), ("storage", "path", ""),
                 ("watch", "unknown", 123)]
        for section, key, value in cases:
            with self.subTest(section=section, key=key), self.assertRaises(ConfigError):
                self.load(lambda raw: raw[section].__setitem__(key, value))

    def test_state_cannot_live_on_watched_mount(self):
        with self.assertRaises(ConfigError):
            self.load(lambda raw: raw["storage"].__setitem__("path", "/mnt/video/state"))

    def test_check_config_has_no_side_effects(self):
        self.load()
        from video_monitor import main
        with patch("video_monitor.Monitor") as monitor, patch("video_monitor.check_runtime") as runtime:
            self.assertEqual(main(["--config", str(self.root / "config.yaml"), "--check-config"]), 0)
            monitor.assert_not_called()
            runtime.assert_not_called()
        self.assertFalse((self.root / "state").exists())

    def test_invalid_yaml(self):
        source = self.root / "broken.yaml"
        source.write_text("watch: [unterminated")
        with self.assertRaises(ConfigError):
            load_config(source)


class StoreTests(TemporaryProject):
    def test_stability_changes_and_deduplication(self):
        store = Store(self.root / "queue.sqlite3")
        self.addCleanup(store.close)
        old = store.observe("video.mp4", 10, 100, 0, 30)
        self.assertIsNone(store.next_job(29))
        key = store.observe("video.mp4", 20, 200, 29, 30)
        store.observe("video.mp4", 20, 200, 58, 30)
        self.assertIsNone(store.next_job(58))
        store.observe("video.mp4", 20, 200, 59, 30)
        self.assertEqual(store.next_job(59)["id"], key)
        self.assertEqual(store.db.execute("SELECT status FROM jobs WHERE id=?", (old,)).fetchone()[0],
                         "superseded")
        store.start(key)
        store.finish(key, {"person": "image.jpg"}, "video.mp4")
        store.observe("video.mp4", 20, 200, 100, 30)
        self.assertIsNone(store.next_job(100))
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0], 1)

    def test_recovery_after_process_exit(self):
        path = self.root / "queue.sqlite3"
        store = Store(path)
        key = store.observe("video.mp4", 1, 1, 0, 1)
        store.observe("video.mp4", 1, 1, 2, 1)
        store.start(key)
        store.close()
        store = Store(path)
        self.addCleanup(store.close)
        store.recover()
        row = store.next_job(3)
        self.assertEqual(row["id"], key)
        self.assertEqual(row["attempts"], 1)

    def test_delivery_retry_survives_restart(self):
        path = self.root / "queue.sqlite3"
        store = Store(path)
        key = store.observe("video.mp4", 1, 1, 0, 1)
        store.finish(key, {"cat": "cat.jpg", "dog": "dog.jpg"}, "video.mp4")
        cat = store.next_delivery(0)
        store.delivery_failed(cat, 100, "network down")
        dog = store.next_delivery(0)
        store.delivered(dog)
        store.close()
        store = Store(path)
        self.addCleanup(store.close)
        self.assertIsNone(store.next_delivery(99))
        self.assertEqual(store.next_delivery(100)["object_name"], cat["object_name"])
        self.assertIsNone(store.next_job(100))


class MonitorTests(TemporaryProject):
    def test_scanner_filters_and_does_not_follow_symlinks(self):
        (self.watch / "one.MP4").write_bytes(b"x")
        (self.watch / "empty.mp4").touch()
        (self.watch / "text.txt").write_text("x")
        (self.watch / "sub").mkdir()
        (self.watch / "sub" / "two.mp4").write_bytes(b"x")
        (self.watch / "link.mp4").symlink_to(self.watch / "one.MP4")
        self.assertEqual(len(list(scan_files(self.watch, False))), 1)
        self.assertEqual(len(list(scan_files(self.watch, True))), 2)

    def test_success_creates_delivery_and_sent_image_is_removed(self):
        monitor = self.monitor()
        key, row = self.ready_job(monitor)
        directory = monitor.job_directory(row)
        directory.mkdir()
        image = directory / "person.jpg"
        image.write_bytes(b"jpeg")
        (directory / "result.json").write_text(json.dumps({
            "annotated_paths": {"person": str(image)}, "detected_objects": ["person"]}))
        monitor.store.start(key)
        monitor.complete("analyze", row, 0, False)
        delivery = monitor.store.next_delivery(time.time())
        self.assertEqual(delivery["image"], str(image))
        monitor.complete("send", delivery, 0, False)
        self.assertFalse(image.exists())
        self.assertIsNone(monitor.store.next_delivery(time.time()))

    def test_video_changed_during_analysis_is_discarded(self):
        monitor = self.monitor()
        key, row = self.ready_job(monitor)
        Path(row["path"]).write_bytes(b"changed video with different size")
        monitor.complete("analyze", row, 0, False)
        self.assertEqual(monitor.store.db.execute("SELECT status FROM jobs WHERE id=?", (key,))
                         .fetchone()[0], "superseded")
        self.assertIsNone(monitor.store.next_delivery(time.time()))

    def test_corrupt_video_eventually_fails_without_blocking_next_job(self):
        monitor = self.monitor()
        key, row = self.ready_job(monitor)
        for attempt in range(3):
            monitor.store.start(key)
            monitor.complete("analyze", row, 1, False)
            row = monitor.store.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone()
        self.assertEqual(row["status"], "failed")
        other = monitor.store.observe("other.mp4", 1, 1, 0, 1)
        monitor.store.observe("other.mp4", 1, 1, 2, 1)
        self.assertEqual(monitor.store.next_job(time.time())["id"], other)

    def test_missing_result_is_not_marked_done(self):
        monitor = self.monitor()
        key, row = self.ready_job(monitor)
        monitor.store.start(key)
        monitor.complete("analyze", row, 0, False)
        self.assertIsNone(monitor.store.next_delivery(time.time()))
        self.assertEqual(monitor.store.db.execute("SELECT status FROM jobs WHERE id=?", (key,))
                         .fetchone()[0], "ready")

    def test_network_failure_keeps_attachment(self):
        monitor = self.monitor()
        key, row = self.ready_job(monitor)
        directory = monitor.job_directory(row)
        directory.mkdir()
        image = directory / "cat.jpg"
        image.write_bytes(b"image")
        monitor.store.finish(key, {"cat": str(image)}, "test.mp4")
        delivery = monitor.store.next_delivery(time.time())
        monitor.complete("send", delivery, 1, False)
        self.assertTrue(image.exists())
        self.assertIsNone(monitor.store.next_delivery(time.time()))
        self.assertIsNotNone(monitor.store.next_delivery(time.time() + 31))

    def test_remount_waits_for_analysis_and_stops_scanning(self):
        monitor = self.monitor()
        with patch.object(monitor, "poll"), patch.object(monitor, "scan") as scan, \
                patch.object(monitor, "launch") as launch:
            monitor.active = ("analyze", None, 0, None)
            monitor.tick()
            launch.assert_not_called()
            monitor.active = ("mount", None, 0, None)
            scan.reset_mock()
            monitor.tick()
            scan.assert_not_called()
        monitor.active = None

    def test_failed_mount_disables_analysis(self):
        monitor = self.monitor()
        monitor.mount_ok = True
        monitor.complete("mount", None, 1, False)
        self.assertFalse(monitor.mount_ok)

    def test_timeout_terminates_worker_and_schedules_retry(self):
        monitor = self.monitor()
        key, row = self.ready_job(monitor)
        monitor.store.start(key)
        from unittest.mock import Mock
        process = Mock(returncode=-15)
        process.poll.return_value = None
        monitor.active = ("analyze", process, 0, row)
        with patch.object(monitor, "terminate") as terminate:
            monitor.poll()
            terminate.assert_called_once_with(process)
        self.assertIsNone(monitor.active)
        self.assertEqual(monitor.store.db.execute("SELECT status FROM jobs WHERE id=?", (key,))
                         .fetchone()[0], "ready")

    def test_shell_metacharacters_remain_literal_arguments(self):
        message = "person; $(touch /tmp/should-not-exist)"
        command = expand_command(["send", "-g", "{group_id}", "{message}", "{image}"], **{
            "{message}": message, "{image}": "/path with spaces/image.jpg",
            "{group_id}": "group/id=="})
        self.assertEqual(command, ["send", "-g", "group/id==", message, "/path with spaces/image.jpg"])

    def test_retry_delay_is_bounded(self):
        self.assertEqual(retry_delay(self.config, 1), 30)
        self.assertEqual(retry_delay(self.config, 2), 60)
        self.assertEqual(retry_delay(self.config, 100000), 3600)


class RemountTests(TemporaryProject):
    def run_remount(self, existing="/dev/loop7", partition="yes", mounted="yes", unmount_status="0"):
        image = self.root / "disk image.img"
        image.touch()
        log = self.root / "commands.log"
        # All privileged commands AND block-device checks are replaced inside bash.
        harness = r'''
test() {
    if [[ "$1" == "-b" ]]; then
        [[ "$PARTITION_PRESENT" == yes && "$2" == /dev/loop7p1 ]]
    else
        builtin test "$@"
    fi
}
mountpoint() { [[ "$MOUNTED" == yes ]]; }
sudo() {
    if [[ "${1:-}" == -n ]]; then shift; fi
    "$@"
}
umount() { echo "umount $*" >> "$LOG"; return "$UNMOUNT_STATUS"; }
losetup() {
    echo "losetup $*" >> "$LOG"
    if [[ "$1" == --associated ]]; then
        printf '%s' "$EXISTING"
    else
        printf '%s' /dev/loop7
    fi
}
partx() { echo "partx $*" >> "$LOG"; PARTITION_PRESENT=yes; }
mount() { echo "mount $*" >> "$LOG"; }
sleep() { :; }
source "$SCRIPT" "$IMAGE" "$TARGET"
'''
        env = dict(os.environ, EXISTING=existing, PARTITION_PRESENT=partition, MOUNTED=mounted,
                   UNMOUNT_STATUS=unmount_status, LOG=str(log), SCRIPT=str(PROJECT / "remount.sh"),
                   IMAGE=str(image), TARGET=str(self.watch))
        result = subprocess.run(["/bin/bash", "-c", harness], env=env, capture_output=True, text=True)
        return result, log.read_text() if log.exists() else ""

    def test_existing_loop_is_reused(self):
        result, log = self.run_remount()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--find", log)
        self.assertNotIn("partx", log)
        self.assertIn("mount -o ro -- /dev/loop7p1", log)

    def test_missing_loop_is_created_with_partition_scan(self):
        result, log = self.run_remount(existing="", mounted="no")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--find --nooverlap --show --partscan --read-only", log)
        self.assertNotIn("umount", log)

    def test_missing_partition_is_registered_again(self):
        result, log = self.run_remount(partition="no")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("partx --add --nr 1 /dev/loop7", log)
        self.assertNotIn("--find", log)

    def test_busy_mount_aborts_without_touching_loop_devices(self):
        result, log = self.run_remount(unmount_status="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("losetup", log)
        self.assertNotIn("mount -o", log)

    def test_ambiguous_existing_mapping_is_rejected(self):
        result, log = self.run_remount(existing="/dev/loop7\n/dev/loop8")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("mount -o", log)


if __name__ == "__main__":
    unittest.main()
