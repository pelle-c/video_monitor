# Video to Signal on Raspberry Pi

The application monitors MP4 files on a mounted disk, analyzes completed files,
and sends one annotated JPG per identified **object class and video** to the
configured Signal group. If no objects are found, it sends a text message with
the video filename and a frame from one second into the video. If the video is
shorter than one second, its last available frame is used. On first startup,
existing MP4 files are inventoried and ignored; files that appear after the
inventory are analyzed. The inventory is stored in SQLite and persists across
restarts.

## Running

Run the commands on the Raspberry Pi from `/home/pelle/video_analyzer`.
`./.venv-pi` is the isolated runtime environment with the CPU build of PyTorch
and TorchVision that works on the Pi's Cortex-A72. It reuses the project's
installed Ultralytics, OpenCV, and other Python dependencies. If the environment
needs to be recreated, run on the Pi:
`python3 -m venv --system-site-packages .venv-pi`
`.venv-pi/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu`.
The Signal command uses the linked account and group ID in the YAML configuration.

1. Copy `config.example.yaml` to `config.yaml`.
2. Set the disk path, remount command, and Signal account in the YAML file.
3. Validate the YAML without running external commands:
   `./.venv-pi/bin/python video_monitor.py --config config.yaml --check-config`
4. Start the monitor: `./run_monitor.sh` (or `./run_monitor.sh config.yaml`).

The default model is `runs/detect/train-4/weights/best.pt`, which runs on the CPU.
A model test without Signal or remounting can be run with
`python3 video_analyzer.py --no-isolation --self-test`.
Analyze a single video with `.venv-pi/bin/python video_analyzer.py /path/to/video.mp4`.
The analyzer reports the number of processed frames every ten frames and when
the video is complete.

## Running a TFLite model

Install the LiteRT runtime in the project virtual environment on the Raspberry Pi:
`.venv-pi/bin/python -m pip install -r requirements-tflite.txt`.
Then analyze a video directly, for example:

```sh
.venv-pi/bin/python video_analyzer.py /path/to/video.mp4 \
  --model runs/detect/train-8/weights/best_saved_model/best_float32.tflite
```

For the monitor, set `analysis.model` in `config.yaml` to the desired `.tflite`
file; the existing default model is not changed automatically. Ultralytics
exports use the model metadata for class names and image size. For other models,
you can provide a class names file with `analysis.data` or `--data`. Regular
`.tflite` files run on the CPU. Files with `edgetpu` in the name require the
Coral installation described below.

## Running a model on Coral Edge TPU

The Pi already detects the USB accelerator. To use it, the model file must be
compiled for Edge TPU and have `edgetpu` in its filename. The example model in
the project is `runs/detect/train-4/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite`.
Install the Python runtime in `.venv-pi`:

```sh
.venv-pi/bin/python -m pip install -r requirements-edgetpu.txt
```

The requirements file pins NumPy below 2 and OpenCV 4.11 because
`tflite-runtime 2.14.0` uses the NumPy 1.x ABI.

The locally built throttled runtime is located here:

`tools/edge-tpu-build/libedgetpu/out/throttled/k8/libedgetpu.so.1.0`

For Python to find `libedgetpu.so.1` at startup, the runtime directory must be
in `LD_LIBRARY_PATH`. Set it for each command that runs the Coral model:

```sh
LD_LIBRARY_PATH=/home/pelle/video_analyzer/tools/edge-tpu-build/libedgetpu/out/throttled/k8 \
  .venv-pi/bin/python video_analyzer.py /path/to/video.mp4 \
  --model runs/detect/train-4/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite
```

Use the same environment variable for the monitor:

```sh
LD_LIBRARY_PATH=/home/pelle/video_analyzer/tools/edge-tpu-build/libedgetpu/out/throttled/k8 \
  ./run_monitor.sh config.yaml
```

This applies when the runtime library is in the project build directory. If
`libedgetpu.so.1` is installed in a standard path by the system package manager,
`LD_LIBRARY_PATH` is usually not needed.

Then install the Coral system library and udev rule on the Raspberry Pi:

```sh
sudo install -d -m 755 /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/etc/apt/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install libedgetpu1-std
```

Unplug and reconnect the USB accelerator after package installation to activate
the udev rule. Check that `ldconfig -p` lists `libedgetpu.so.1` and that the
regular user can open the USB device. Run a single video with:

```sh
.venv-pi/bin/python video_analyzer.py /path/to/video.mp4 \
  --model runs/detect/train-4/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite
```

For the monitor, set the same model file in `analysis.model` in `config.yaml`.
Models without `edgetpu` in the name run on the CPU even when Coral is connected.

`remount.sh` assumes a disk image with partition 1 and uses `sudo -n` for
`umount`, `losetup`, `partx`, and `mount`. The user running the monitor must have
passwordless sudo permission for these commands. The rest of the monitor runs
as the regular user. Do not run the entire analysis program as root to resolve
mount permissions. Adapt the remount command to your existing environment.
The image is mounted read-only. No video is read while remounting is in progress.

## USB gadget service

The project includes the systemd unit file
[`systemd/usb-gadget.service`](systemd/usb-gadget.service) and its startup script
[`systemd/usb_gadget_start.sh`](systemd/usb_gadget_start.sh). The service exposes
the disk image `/video/virtual_disk.img` as USB mass storage. The startup script
uses ConfigFS, a loop device, and a USB Device Controller (UDC); systemd runs the
service as root. The script assumes that the image exists at the specified path.

From the project directory on the Raspberry Pi, install or update both files with:

```sh
sudo install -m 755 systemd/usb_gadget_start.sh /usr/local/bin/usb_gadget_start.sh
sudo install -m 644 systemd/usb-gadget.service /etc/systemd/system/usb-gadget.service
sudo systemctl daemon-reload
```

Enable the service at startup with
`sudo systemctl enable usb-gadget.service`; start or restart it with
`sudo systemctl start usb-gadget.service` or
`sudo systemctl restart usb-gadget.service`, respectively. Check it with
`systemctl status usb-gadget.service`. This project file documents the unit;
adding the files to the project does not automatically change the installed
systemd configuration.

## Local signal-cli installation

signal-cli 0.14.9 is installed in `tools/signal-cli-native` from the
[signal-cli-native ARM64 package](https://packaging.gitlab.io/signal-cli/).
This is the package project's experimental GraalVM variant without Java. No
system packages or services were installed. The package's SHA256 was checked
against the package index before extraction.

Run `./tools/signal-cli --version` from the project directory.
The startup script stores the account under `state/signal-cli` with directory
permissions set to 700. The example configuration uses this startup script.
The account still needs to be linked to Signal, and the example number must be
replaced before sending can work.

## Configuration and defaults

Relative file paths are resolved from the YAML file's directory. External
commands run from the project directory as argument lists without a shell.
Placeholders are replaced only when they make up an entire argument:
`{mountpoint}`, `{image}`, `{message}`, and `{group_id}`.

| Setting | Default value |
| --- | --- |
| watch.path | Required; the example uses /mnt/video |
| watch.scan_seconds | 10 seconds |
| watch.stable_seconds | 30 seconds without a change in size or modification time |
| watch.recursive | true; scans the mount and subdirectories |
| mount.remount_command | Required; the example uses remount.sh with /video/virtual_disk.img |
| mount.interval_seconds | 60 seconds, after remount completes |
| mount.timeout_seconds | 30 seconds |
| analysis.model | runs/detect/train-8/weights/best.pt |
| analysis.data | Empty string; optional class names file for TFLite |
| analysis.confidence | 0.45 |
| analysis.min_object_hits | 3 detections per class |
| analysis.max_bad_frames | 25; warns after the threshold but continues skipping unreadable frames |
| analysis.threads | 1; also used for the Edge TPU CPU fallback |
| analysis.iou | 0.7 |
| analysis.timeout_seconds | 1800 seconds |
| analysis.max_attempts | 3 |
| signal.command | signal-cli to the specified group; see YAML |
| signal.group_id | Required; Signal group ID, configured in YAML |
| signal.timeout_seconds | 120 seconds |
| storage.path | state, outside the monitored directory |
| storage.max_spool_mb | 256 MiB; stops new analyses from starting when the limit is reached |
| retry.initial_seconds | 30 seconds |
| retry.maximum_seconds | 3600 seconds |

Remounting waits until active analysis or sending has finished. The interval can
therefore be longer than configured. The spool limit is checked before a new
analysis and is not a hard limit on image writing already in progress.

## Recovery and logging

The SQLite queue in `state` stores processed files and pending sends across
restarts. File identity is based on path, size, and modification time. Failed
sends are retried with exponentially increasing delays. Images are deleted only
after a successful send. A process crash immediately after Signal receives an
image can result in a duplicate on the next attempt.

Before analysis, the MP4 file is copied to the job directory under `storage.path`.
A model-free check then verifies that the copy can be opened and that a frame
near the end can be decoded. If validation fails, the job waits and retries
without consuming an analysis attempt. The file identity is also checked again
after copying and validation. This reduces the chance of analyzing an MP4 while
Arlo is still writing it, but cannot guarantee that the hub has flushed all
data to disk. Analysis reads the local copy, so inference in progress is
independent of the USB mount. The copy is removed when analysis finishes; the
spool limit includes the video copy, so large files may wait until storage space
is available.

Errors are logged to standard error. After three failed analysis attempts, the
video is marked as failed; a changed file identity creates a new job. SIGINT and
SIGTERM stop the program and its active subprocess. A lock prevents multiple
monitors from using the same state directory. Run one analysis at a time on the
Pi with 1 GB of RAM.

## Tests

Run on the Raspberry Pi:
`PYTHONDONTWRITEBYTECODE=1 .venv-pi/bin/python -m unittest discover -s tests -v`.
The tests use temporary files under `tests` and simulate remounting and Signal.
A real model test is also needed to check the inference environment.

## Development

Most of the work with Codex was done via the Codex CLI on a laptop. The Raspberry
Pi project directory was mounted using SSHFS:
`sshfs user@rpi:/home/user/video_analyzer /home/user/rpi-project -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3`
