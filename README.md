# Module 16 Assignment: People Flow Detection

**Project:** People Flow Detection using Object Tracking and Heatmap Visualization

This project processes a video to detect and track people, count directional movement across two horizontal lines, annotate the video with IDs and live counts, and save a heatmap of where people were present.

## Assignment objective and features

- Detect people with a pre-trained YOLO model.
- Track detections across frames with persistent IDs using ByteTrack.
- Keep previous positions by track ID and use trajectory direction to count entries and exits.
- Draw two color-coded horizontal lines and live IN/OUT counts on the output video.
- Accumulate people’s bounding-box center positions and save a heatmap overlay.

## Input video

Default sample: [people-walking.mp4](https://media.roboflow.com/supervision/video-examples/people-walking.mp4)

The script also accepts a local video file or another URL supported by OpenCV.

## Detection and tracking method

- **Detector:** Ultralytics YOLO11 small (`yolo11s.pt`), using COCO class `0` (person). The small model and `--imgsz 960` improve recall for distant people over the faster nano/640 defaults. Confidence threshold is `0.15` so ByteTrack can use low score detections to maintain tracks through brief occlusion; class filtering limits detections to people.
- **Tracker:** ByteTrack (`bytetrack_people.yaml`) with persistent tracking and a 60-frame track buffer. Lower-score matching and the longer buffer help keep IDs stable through short detector gaps. Ultralytics installs the `lap` dependency used by ByteTrack as needed.
- **Track state:** A dictionary stores each track ID’s previous bounding-box center, and short trails are drawn for visual direction.
- **Heatmap samples:** The center of each detected bounding box is accumulated once per observed frame, as specified in the assignment. Visits are smoothed, colorized, and overlaid on the video’s first frame for scene context. Red/yellow means higher observed presence; blue means lower presence.

The PolygonZone site can help inspect or draw regions on a video frame. This assignment uses full-width horizontal lines, so the line y-coordinates are configured directly as frame-height fractions in the script. No polygon package or manually copied polygon coordinates are required.

## Lines and IN/OUT counting logic

For the supplied 1920 × 1080 sample video, defaults are:

| Line | Pixel coordinates | Default fraction | Color | Event |
| --- | --- | --- | --- | --- |
| Upper | `(0, 432)` to `(1920, 432)` | `0.40 × frame height` | Yellow | IN, top to bottom |
| Lower | `(0, 648)` to `(1920, 648)` | `0.60 × frame height` | Blue | OUT, bottom to top |

Coordinates scale automatically with the input video dimensions. Change `--upper` and `--lower` to tune the lines for a different camera or entrance/exit area. Keep `0 <= upper < lower <= 1`.

For each persistent track ID, the script compares the current bounding-box center with its previous position and keeps per-ID armed state:

- **IN:** First arm the ID after observing it above the upper line; count once when its center moves downward past that line. A small pixel margin filters box jitter around the line.
- **OUT:** First arm the ID after observing it below the lower line; count once when its center moves upward past that line. The same margin is used.
- The assignment defines these as two separate directional events. A person does not have to cross both lines to count; one ID may count once as IN and once as OUT if both events occur during the clip.

## Setup and run

Use **Python 3.12** and a project-local `.venv` on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python people_flow.py
```

If PowerShell blocks activation, run the script with the environment's Python directly instead: `.\.venv\Scripts\python.exe people_flow.py`.

The first run downloads the YOLO weights. Example with custom input, output, and lines:

```powershell
python people_flow.py --source "videos/shop.mp4" --output "output/shop_tracked.mp4" --heatmap "output/shop_heatmap.png" --upper 0.38 --lower 0.62 --conf 0.30 --imgsz 960
```

Use `--device cpu` for CPU inference or `--device 0` for the first CUDA GPU.

## Deliverables

- **Python script:** `people_flow.py`
- **Dependencies:** `requirements.txt`
- **Annotated video:** `output/people_flow.mp4` (bounding boxes, IDs, short trails, both lines, live counters)
- **Final heatmap:** `output/heatmap.png` (presence overlay)
- **This README:** detection method, line coordinates, and IN/OUT explanation

The earlier YOLO11n run on the supplied sample reported **IN: 11, OUT: 7**. The current script uses YOLO11s at 960 pixels, lower-score detections, and a tuned ByteTrack config, so rerun it to generate output files that match this higher-recall configuration. Counts can vary with model weights, confidence threshold, tracker behavior, and line placement. For the best result on a particular camera, review the annotated video and tune the line fractions and confidence threshold to the actual counting zone.

## Accuracy notes

Tracking quality depends on visibility, lighting, crowd overlap, and detector confidence. ByteTrack helps maintain IDs through short detection gaps, while per-ID approach-side arming and a small jitter margin reduce missed crossings and false line events. Long occlusions can still cause an ID switch, and a track that begins inside the counting area cannot establish whether it crossed the line before the clip began. Review the output video before treating counts as ground truth.
