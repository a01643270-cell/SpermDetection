#!/usr/bin/env python3
"""Baseline YOLOv8 + DeepSort real-time detection/tracking pipeline."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2

MIN_TIME_DELTA = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSort pipeline")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Path to YOLO model")
    parser.add_argument("--source", type=str, default="0", help="Video source: file path or webcam index")
    parser.add_argument("--output", type=Path, default=Path("outputs/tracked_output.mp4"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="0", help="CUDA device id like '0', or 'cpu'")
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--nn-budget", type=int, default=100)
    parser.add_argument("--no-show", action="store_true", help="Disable preview window")
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> None:
    args = parse_args()
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for tracking pipeline. Install with: pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    tracker = DeepSort(max_age=args.max_age, nn_budget=args.nn_budget)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame size from source {args.source}: width={width}, height={height}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, (width, height)
    )

    last = time.time()
    fps_window = deque(maxlen=30)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                if cls_id != args.class_id:
                    continue
                left = float(x1)
                top = float(y1)
                box_width = float(x2 - x1)
                box_height = float(y2 - y1)
                if box_width <= 0 or box_height <= 0:
                    continue
                detections.append(([left, top, box_width, box_height], float(conf), str(cls_id)))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            track_id = track.track_id

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if not args.no_show:
            now = time.time()
            current_fps = 1.0 / max(MIN_TIME_DELTA, now - last)
            fps_window.append(current_fps)
            smoothed_fps = sum(fps_window) / len(fps_window)
            last = now
            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}",
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLOv8 + DeepSort", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Tracking output saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
