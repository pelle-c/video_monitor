import argparse
import faulthandler
import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import sys
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect objects in an MP4 video or folder of MP4 videos with a YOLO PyTorch model or TFLite export."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Path to an MP4 video or a folder containing MP4 videos.",
    )
    parser.add_argument(
        "--max-bad-frames",
        type=int,
        default=25,
        help="Warn after this many unreadable frames; continue skipping them. Default: 25.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum detection confidence to count an object. Default: 0.45.",
    )
    parser.add_argument(
        "--min-object-hits",
        type=int,
        default=3,
        help="Minimum detections needed before reporting an object type. Default: 3.",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/train-8/weights/best.pt",
        help="Path to a YOLO .pt model or .tflite export. Files named *edgetpu* use Coral.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional check of the fixed model size. By default use the model input size.",
    )
    parser.add_argument("--data", help="Dataset YAML with class names, if model metadata is missing.")
    parser.add_argument("--result-json", help="Write a machine-readable result (requires --no-isolation and one video).")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU inference and Edge TPU CPU-fallback threads. Default: 1.",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold. Default: 0.7.")
    parser.add_argument(
        "--annotated-dir",
        default="annotated_objects",
        help="Directory for annotated object images. Default: annotated_objects.",
    )
    parser.add_argument(
        "--no-save-annotated",
        action="store_true",
        help="Do not save one annotated JPG per detected object type.",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Run analysis in the main process instead of a protected child process.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate that the video opens and a frame near its end can be decoded; do not load a model.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Load the selected model and analyze a blank frame.",
    )
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1.")
    if args.min_object_hits < 1:
        parser.error("--min-object-hits must be positive.")
    if args.result_json and (not args.no_isolation or args.self_test
                             or not args.video_path or Path(args.video_path).is_dir()):
        parser.error("--result-json requires --no-isolation and a single video file")
    return args


def load_model(args):
    model_path = Path(args.model)
    suffix = model_path.suffix.lower()
    if suffix not in {".pt", ".tflite"}:
        raise ValueError(f"Unsupported model format: {suffix or '(no extension)'}; use .pt or .tflite")

    try:
        if suffix == ".tflite":
            runtime_module = (
                "tflite_runtime" if "edgetpu" in model_path.name.lower()
                else "ai_edge_litert"
            )
            try:
                __import__(runtime_module)
            except ImportError as exc:
                if runtime_module == "ai_edge_litert":
                    hint = "Install it with .venv-pi/bin/python -m pip install -r requirements-tflite.txt"
                else:
                    hint = "Install it with .venv-pi/bin/python -m pip install -r requirements-edgetpu.txt"
                raise RuntimeError(f"Missing {runtime_module} for this TFLite model. {hint}.") from exc

        from pytorch_detector import PyTorchDetector
        return PyTorchDetector(args.model, threads=args.threads, imgsz=args.imgsz,
                               iou=args.iou, data_path=args.data)
    except Exception as exc:
        if suffix == ".tflite" and is_edge_tpu_load_error(args.model, exc):
            print_edge_tpu_load_help(args.model, exc)
        else:
            print(f"Error: Could not load model: {exc}")
        raise SystemExit(1) from exc


def is_edge_tpu_load_error(model_path, exc):
    model_name = os.path.basename(str(model_path)).lower()
    error_text = str(exc).lower()
    return (
        "edgetpu" in model_name
        or "libedgetpu" in error_text
        or "failed to load delegate" in error_text
    )


def print_edge_tpu_load_help(model_path, exc):
    print("Error: Could not load the Edge TPU model.")
    print(f"Model: {model_path}")
    print(f"Details: {exc}")
    print("")
    print("This model requires a working Coral Edge TPU runtime and device.")
    print("Check that:")
    print("  1. The Coral device appears in lsusb.")
    print("  2. The Edge TPU runtime is installed, for example libedgetpu1-std.")
    print("  3. Your user can access the USB device, then unplug/replug the Coral.")
    print("")
    print("To run without Coral, use the non-Edge-TPU export instead:")
    print(
        "  --model runs/detect/train-8/weights/best_saved_model/"
        "best_full_integer_quant.tflite"
    )


def open_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' was not found.")
        raise SystemExit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open or read video file: {video_path}")
        raise SystemExit(1)

    return cap


def read_next_frame(cap, bad_frames, max_bad_frames):
    ret, frame = cap.read()
    if ret:
        return frame, bad_frames

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0 and current_frame >= total_frames:
        return None, bad_frames

    bad_frames += 1
    if bad_frames == max_bad_frames + 1:
        print(
            f"Warning: More than {max_bad_frames} unreadable frames; "
            "continuing analysis and skipping unreadable frames.",
            flush=True,
        )

    requested_frame = current_frame + 1
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, requested_frame):
        return None, bad_frames
    next_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if next_frame <= current_frame:
        return None, bad_frames
    return False, bad_frames


def safe_filename(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def get_model_class_name(model, class_id):
    return model.names.get(class_id, f"class_{class_id}")


def save_annotated_object(
    frame,
    box,
    object_name,
    video_key,
    model_key,
    frame_number,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box.xyxy.tolist()]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    confidence = float(box.confidence)
    label = f"{object_name} {confidence:.0%}"
    color = (0, 255, 0)

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
    text_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    text_width, text_height = text_size
    label_y = max(y1, text_height + baseline + 4)
    cv2.rectangle(
        annotated_frame,
        (x1, label_y - text_height - baseline - 4),
        (x1 + text_width + 6, label_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        annotated_frame,
        label,
        (x1 + 3, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    filename = (
        f"{video_key}_{safe_filename(object_name)}_frame-{frame_number:06d}"
        f"_conf-{confidence:.2f}_model-{model_key}.jpg"
    )
    output_path = os.path.join(output_dir, filename)
    if not cv2.imwrite(output_path, annotated_frame):
        return None

    return output_path


def save_no_object_frame(frame, video_key, model_key, frame_number, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = (
        f"{video_key}_no_objects_frame-{frame_number:06d}"
        f"_model-{model_key}.jpg"
    )
    output_path = os.path.join(output_dir, filename)
    if not cv2.imwrite(output_path, frame):
        return None
    return output_path


def analyze_video(args):
    started_at = time.perf_counter()
    video_path = args.video_path
    video_key = safe_filename(os.path.splitext(os.path.basename(video_path))[0])
    model_key = safe_filename(args.model)
    model = load_model(args)
    cap = open_video(video_path)
    object_counts = {}
    object_confidences = {}
    bad_frames = 0
    analyzed_frames = 0
    last_reported_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_index = max(round(fps), 0) if fps > 0 else 0
    if total_frames > 0:
        target_frame_index = min(target_frame_index, total_frames - 1)
    fallback_frame = None
    fallback_frame_number = None
    annotated_paths = {}

    print(f"Analyzing '{video_path}'... Please wait...", flush=True)

    try:
        while cap.isOpened():
            frame, bad_frames = read_next_frame(
                cap, bad_frames, args.max_bad_frames
            )
            if frame is None:
                break
            if frame is False:
                continue

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if fallback_frame is None and frame_number - 1 >= target_frame_index:
                fallback_frame = frame.copy()
                fallback_frame_number = frame_number
            try:
                detections = model.detect(frame, confidence=args.confidence)
            except Exception as exc:
                if is_edge_tpu_load_error(args.model, exc):
                    print_edge_tpu_load_help(args.model, exc)
                    raise SystemExit(1) from exc
                raise

            analyzed_frames += 1
            if analyzed_frames % 10 == 0:
                total = (str(total_frames) if total_frames >= analyzed_frames
                         else f"unknown (container reports {total_frames})")
                print(f"Frames analyzed: {analyzed_frames}/{total}", flush=True)
                last_reported_frames = analyzed_frames

            for box in detections:
                class_id = box.class_id
                confidence = float(box.confidence)
                if confidence < args.confidence:
                    continue

                object_name = get_model_class_name(model, class_id)

                object_counts[object_name] = object_counts.get(object_name, 0) + 1
                object_confidences[object_name] = max(
                    object_confidences.get(object_name, 0.0), confidence
                )

                if (
                    not args.no_save_annotated
                    and object_name not in annotated_paths
                    and object_counts[object_name] >= args.min_object_hits
                ):
                    output_path = save_annotated_object(
                        frame,
                        box,
                        object_name,
                        video_key,
                        model_key,
                        frame_number,
                        args.annotated_dir,
                    )
                    if output_path:
                        annotated_paths[object_name] = output_path

    finally:
        cap.release()

    if analyzed_frames != last_reported_frames:
        total = (str(total_frames) if total_frames >= analyzed_frames
                 else f"unknown (container reports {total_frames})")
        print(f"Frames analyzed: {analyzed_frames}/{total}", flush=True)

    detected_objects = {
        object_name
        for object_name in object_counts
        if object_counts.get(object_name, 0) >= args.min_object_hits
    }

    if not detected_objects and not args.no_save_annotated:
        if fallback_frame is None:
            raise RuntimeError("Could not read a frame at or near one second into the video")
        image_path = save_no_object_frame(
            fallback_frame,
            video_key,
            model_key,
            fallback_frame_number,
            args.annotated_dir,
        )
        if image_path is None:
            raise RuntimeError("Could not save the no-object preview image")
        annotated_paths["no_objects"] = image_path

    if not args.no_save_annotated:
        missing_images = detected_objects - annotated_paths.keys()
        if missing_images:
            raise RuntimeError(f"Could not save images for: {', '.join(sorted(missing_images))}")

    return {
        "detected_objects": detected_objects,
        "object_counts": object_counts,
        "object_confidences": object_confidences,
        "annotated_paths": annotated_paths,
        "annotated_dir": args.annotated_dir,
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def self_test(args):
    try:
        import numpy as np
    except ImportError as exc:
        print("Error: The 'numpy' package is not installed.")
        raise SystemExit(1) from exc

    model = load_model(args)
    test_frame = np.zeros((model.height, model.width, 3), dtype=np.uint8)
    model.detect(test_frame, confidence=args.confidence)
    print("Self-test passed: model loaded and analyzed a blank frame.")


def print_result(analysis_result):
    detected_objects = analysis_result["detected_objects"]
    object_counts = analysis_result["object_counts"]
    object_confidences = analysis_result["object_confidences"]
    annotated_paths = analysis_result["annotated_paths"]

    print("\n--- ANALYSIS COMPLETE ---")
    print(f"Analysis took {analysis_result['elapsed_seconds']:.2f} seconds.")
    if detected_objects:
        print("Objects detected in the video:")
        for object_name in sorted(detected_objects):
            print(
                f"  [X] {object_name} "
                f"({object_counts[object_name]} hits, "
                f"max {object_confidences[object_name]:.0%})"
            )
    else:
        print("No objects met the detection threshold.")

    if annotated_paths:
        if detected_objects:
            description = "annotated object image(s)"
        else:
            description = "one-second preview image"
        print(f"Saved {description} to '{analysis_result['annotated_dir']}'.")


def run_worker(result_queue, args):
    faulthandler.enable()
    try:
        if args.self_test:
            self_test(args)
            result_queue.put({"ok": True, "detected_objects": []})
            return

        analysis_result = analyze_video(args)
        result_queue.put(
            {
                "ok": True,
                "detected_objects": sorted(analysis_result["detected_objects"]),
                "object_counts": analysis_result["object_counts"],
                "object_confidences": analysis_result["object_confidences"],
                "annotated_paths": analysis_result["annotated_paths"],
                "annotated_dir": analysis_result["annotated_dir"],
                "elapsed_seconds": analysis_result["elapsed_seconds"],
            }
        )
    except SystemExit as exc:
        result_queue.put({"ok": False, "exit_code": exc.code})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def print_native_crash_help(exit_code):
    signal_number = -exit_code
    signal_name = signal.Signals(signal_number).name

    print(f"\nError: Analysis crashed in a native dependency ({signal_name}).")
    if signal_number == signal.SIGILL:
        print(
            "This usually means OpenCV, the TFLite interpreter, or another binary package was built "
            "with CPU instructions this machine does not support."
        )
    print("Try running:")
    print("  python video_analyzer.py --self-test --model /path/to/model.tflite")
    print("If the self-test also crashes, install a compatible TFLite runtime and OpenCV build.")
    print("If only this video crashes, re-encode or repair the MP4 before analysis.")


def run_isolated(args):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_worker, args=(result_queue, args))
    process.start()
    process.join()

    if process.exitcode and process.exitcode < 0:
        print_native_crash_help(process.exitcode)
        return 1

    if result_queue.empty():
        print("Error: Analysis stopped without returning a result.")
        return process.exitcode or 1

    result = result_queue.get()
    if not result["ok"]:
        if "error" in result:
            print(f"Error: {result['error']}")
        return int(result.get("exit_code") or 1)

    if args.self_test:
        return 0

    print_result(
        {
            "detected_objects": set(result["detected_objects"]),
            "object_counts": result["object_counts"],
            "object_confidences": result["object_confidences"],
            "annotated_paths": result["annotated_paths"],
            "annotated_dir": result["annotated_dir"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    )
    return 0


def run_single(args):
    if not args.no_isolation:
        return run_isolated(args)

    if args.self_test:
        self_test(args)
        return 0

    analysis_result = analyze_video(args)
    if args.result_json:
        result_path = Path(args.result_json)
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        payload = dict(analysis_result)
        payload["detected_objects"] = sorted(payload["detected_objects"])
        with temporary.open("w") as output:
            json.dump(payload, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(result_path)
    print_result(analysis_result)
    return 0


def validate_video(path):
    """Reject incomplete/unreadable MP4 snapshots before spending time on inference."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("video container cannot be opened")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError("video reports no frames")
        # Seeking to the final reported frame checks that the MP4 index reaches
        # the tail, where an interrupted recording is most likely to be truncated.
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            raise ValueError("the final indexed video frame cannot be decoded")
    finally:
        cap.release()


def main():
    args = parse_args()

    if args.validate_only:
        if not args.video_path or Path(args.video_path).is_dir():
            print("Error: --validate-only requires one video file.")
            return 2
        try:
            validate_video(args.video_path)
            print(f"Video is readable: {args.video_path}")
            return 0
        except (OSError, ValueError) as exc:
            print(f"Video is not ready for analysis: {exc}")
            return 1

    if not args.self_test and not args.video_path:
        print("Error: You must provide an MP4 file or folder.")
        print("Usage: python video_analyzer.py <file-or-folder> --model model.tflite")
        return 2

    if args.self_test or not Path(args.video_path).is_dir():
        return run_single(args)

    videos = sorted(
        path for path in Path(args.video_path).iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        print(f"Error: No MP4 files found in '{args.video_path}'.")
        return 1

    failed = 0
    for index, video in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] Analyzing '{video}'", flush=True)
        video_args = argparse.Namespace(**vars(args))
        video_args.video_path = str(video)
        try:
            exit_code = run_single(video_args)
        except SystemExit as exc:
            exit_code = exc.code
        except Exception as exc:
            print(f"Error: {exc}")
            exit_code = 1
        if exit_code:
            failed += 1
            print(f"Failed to analyze '{video}'. Continuing with the next file.")

    print(f"\nFolder analysis complete: {len(videos) - failed} succeeded, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAnalysis interrupted.")
        sys.exit(130)
