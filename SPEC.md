# Project specification

## 1. Purpose
When a new mp4 file is saved on an attached usb-disk, this project analyzes the video and sends images to a Signal group: one image per identified object class, or a frame from one second into the video if no objects are identified.


## 2. Functional requirements
- On first startup, the application shall inventory and ignore MP4 files already present; it shall analyze files that appear after that inventory
- The application shall recursively monitor the mounted share and its subdirectories for new mp4 files
- Before analysis, the application shall copy each video into local spool storage and analyze that snapshot
- The application shall try to identify objects in new mp4 files, use existing video_analyzer.py as a base
- The application shall support Ultralytics PyTorch `.pt` models, CPU `.tflite` models, and Coral Edge TPU `.tflite` models
- The application shall send one extracted image per identified object class to the configured Signal group
- If no objects are identified, the application shall send a frame from one second into the video and a text message to the configured Signal group that includes the video filename

## 3. Inputs
mp4-files on a local mountpoint
the mountpoint needs to be remounted on a configurable interval (default 1 minute)

## 4. Outputs
Signal message with one attached image per detected object class; when no objects are identified, a frame from one second into the video with a text message that includes the video filename

## 5. Configuration
- Configuration shall be stored in a YAML file.
- Default values shall be documented.

## 6. Error handling
- The application shall handle network interruptions.
- Errors shall be logged.
- The application shall recover automatically
  when possible.

## 7. Technical requirements
- Target platform: Raspberry Pi, Linux ARM64.
- The application shall run without a graphical interface.
- Memory consumption shall be minimized.

## 8. Acceptance criteria
- The application starts with a valid configuration.
- The application performs all required functions.
- The application handles invalid input gracefully.
- All tests pass on the Raspberry Pi.
