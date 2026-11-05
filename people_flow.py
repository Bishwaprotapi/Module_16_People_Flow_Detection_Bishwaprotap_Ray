"""People detection, ByteTrack tracking, line crossing counts, and a presence heatmap."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
                        help="Input video path or URL")
    parser.add_argument("--output", default="output/people_flow.mp4", help="Annotated output video")
    parser.add_argument("--heatmap", default="output/heatmap.png", help="Final heatmap image")
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics YOLO weights (COCO person class)")
    parser.add_argument("--upper", type=float, default=0.40, help="Upper line y position as frame-height fraction")
    parser.add_argument("--lower", type=float, default=0.60, help="Lower line y position as frame-height fraction")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold (keeps ByteTrack low-score matches)")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size; larger helps distant/small people")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu, 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.upper < args.lower <= 1:
        raise SystemExit("Line fractions must satisfy 0 <= upper < lower <= 1")

    output_path, heatmap_path = Path(args.output), Path(args.heatmap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise SystemExit("Video has invalid dimensions")

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {output_path}")

    model = YOLO(args.model)
    tracker_config = Path(__file__).with_name("bytetrack_people.yaml")
    upper_y, lower_y = int(height * args.upper), int(height * args.lower)
    heat = np.zeros((height, width), dtype=np.float32)
    # Keep the previous bounding-box center for each live track ID.
    previous: dict[int, tuple[int, int]] = {}
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=24))
    counted: set[tuple[int, str]] = set()
    armed_in: set[int] = set()
    armed_out: set[int] = set()
    in_count = out_count = 0
    reference_frame = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if reference_frame is None:
                reference_frame = frame.copy()
            result = model.track(frame, persist=True, tracker=str(tracker_config), classes=[0],
                                 conf=args.conf, imgsz=args.imgsz, iou=args.iou,
                                 device=args.device, verbose=False)[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box.tolist()
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # Spread each center visit slightly before global smoothing so the
                    # result reads as presence intensity instead of isolated hot pixels.
                    cv2.circle(heat, (cx, cy), 7, 1.0, thickness=-1)
                    trails[int(track_id)].append((cx, cy))
                    tid = int(track_id)
                    old = previous.get(tid)
                    previous[tid] = (cx, cy)
                    # Arm only after seeing the track clearly on the approach side.
                    # This tolerates brief detector gaps and line jitter while avoiding
                    # counts for tracks that first appear inside the counting band.
                    margin = max(2, int(height * 0.002))
                    if cy <= upper_y - margin:
                        armed_in.add(tid)
                    elif cy >= upper_y + margin and old is not None and cy > old[1] and tid in armed_in and (tid, "IN") not in counted:
                        in_count += 1
                        counted.add((tid, "IN"))
                        armed_in.discard(tid)
                    if cy >= lower_y + margin:
                        armed_out.add(tid)
                    elif cy <= lower_y - margin and old is not None and cy < old[1] and tid in armed_out and (tid, "OUT") not in counted:
                        out_count += 1
                        counted.add((tid, "OUT"))
                        armed_out.discard(tid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
                    points = list(trails[int(track_id)])
                    for a, b in zip(points, points[1:]):
                        cv2.line(frame, a, b, (255, 180, 0), 2)

            cv2.line(frame, (0, upper_y), (width, upper_y), (0, 220, 255), 2)
            cv2.line(frame, (0, lower_y), (width, lower_y), (255, 100, 0), 2)
            cv2.putText(frame, "IN line", (12, max(22, upper_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(frame, "OUT line", (12, max(22, lower_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.rectangle(frame, (10, 10), (235, 82), (25, 25, 25), -1)
            cv2.putText(frame, f"IN: {in_count}", (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(frame, f"OUT: {out_count}", (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 170, 255), 2)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    # Smooth center visits, scale against a robust percentile (rather than one outlier),
    # and overlay the activity on the original scene for spatial context.
    smooth = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(5, width / 70), sigmaY=max(5, height / 70))
    if smooth.max() > 0:
        scale = float(np.percentile(smooth[smooth > 0], 99.5))
        scale = max(scale, 1e-6)
        normalized = np.clip(smooth * (255.0 / scale), 0, 255).astype(np.uint8)
        normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        base = reference_frame if reference_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
        strength = (np.clip(smooth / scale, 0, 1) * 0.72).astype(np.float32)[..., None]
        heatmap = (base.astype(np.float32) * (1 - strength) + normalized.astype(np.float32) * strength).astype(np.uint8)
    else:
        heatmap = reference_frame if reference_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(heatmap, "People presence heatmap (blue = lower, red = higher)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(heatmap_path), heatmap)
    print(f"IN: {in_count} | OUT: {out_count}")
    print(f"Video: {output_path}\nHeatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
