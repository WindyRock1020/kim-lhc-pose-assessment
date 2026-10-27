# KIM-LHC Pose Assessment

Research code for pose-based KIM-LHC body posture assessment using MediaPipe. This repository provides video analysis, coordinate-based pose estimation evaluation, temporal threshold sensitivity analysis, and a runnable landmark example with reference outputs.

The implemented total is the **basic body posture score plus posture-related additional points**. It does not include the remaining KIM-LHC components needed to calculate an overall task risk value. The complete mobile application and original participant videos are not included.

## Repository contents

| Component | Files | Purpose |
|---|---|---|
| Video analysis | `main.py`, `src/`, `templates/`, `utils/` | Upload a video through a local web page, estimate landmarks, classify postures, filter the posture sequence, and calculate scores |
| Reference comparison | `evaluate_pose_estimation.py` | Calculate MPJPE and MPJAE from prediction/ground-truth JSON pairs and generate workbooks corresponding to the S1 Data evaluation structure |
| Temporal sensitivity analysis | `temporal_sensitivity_analysis.py` | Run the parameter sweeps described in manuscript Section 3.5, using the embedded V1–V10 posture sequences |
| Landmark demonstration | `poseAnalysisExample.py`, `example_input/`, `example_output/` | Analyze 57 frames of landmark coordinates and compare the results with supplied reference outputs |
| Environment and license | `requirements.txt`, `LICENSE` | Record dependency versions and MIT license terms |

## Installation

The current development environment is Windows with Python 3.11.15. `requirements.txt` records the current package versions, including MediaPipe 0.10.35; it is not a record of the manuscript's historical software environment.

Clone the repository and install the dependencies. The following commands use PowerShell:

```powershell
git clone https://github.com/WindyRock1020/kim-lhc-pose-assessment.git
cd kim-lhc-pose-assessment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the commands below from the repository root. Each script uses settings defined in its source file; no command-line arguments are required.

The requirements use `opencv-contrib-python`. Avoid installing additional OpenCV distributions in the same environment because they share the `cv2` namespace.

## Quick start: landmark example

The example runs without downloading a model or supplying a video:

```powershell
.\.venv\Scripts\python.exe poseAnalysisExample.py
```

The default input directory is `example_input/`. Its 57 JSON files represent frames 0–56. These coordinates were extracted from a MediaPipe landmark TXT export of a V2 sequence; they are not synthetic coordinates. The example shares landmark coordinates rather than the source video.

Each file contains exactly these 12 joints, with finite numeric `[x, y, z]` coordinates:

```text
LWrist, RWrist, LElbow, RElbow,
LShoulder, RShoulder, LHip, RHip,
LKnee, RKnee, LAnkle, RAnkle
```

The input uses metric world coordinates with **upward-positive Y**, rather than normalized image coordinates. The script converts Y to the convention expected by the shared analyzer. It does not align, center, or normalize the coordinates. Filename suffixes identify frame numbers, for example `exampleInput_0000.json`; all files in one run must belong to a single sequence.

The example obtains geometry and posture classifications from `src/mediapipe_lib/base.py`. Temporal processing and score aggregation are implemented directly in the example script. It does not import `main.py` or start a server. Shared modules still require the installed dependencies, including MediaPipe and OpenCV, even though model inference is not performed.

### Expected result

| Output | Expected value |
|---|---:|
| Frames | 57 |
| Basic body posture score | 3 |
| Additional points | 3 |
| Combined body posture score | **6** |
| First maximum-scoring transition | **3 → 1** |

The additional points in this example come from the load-away-from-body condition. Reference files are supplied in `example_output/`. Each execution writes a new timestamped subdirectory containing:

- `response.json`: compact score summary.
- `posture.json`: original/filtered labels, transitions, and posture scores.
- `additional.json`: frequencies, valid-frame counts, and additional points.
- `per_frame.csv`: one field per row and one frame per column; blank values represent unavailable results.
- `run.json`: settings and execution metadata.

No `per_frame.json` is produced. Timestamps and local paths vary between runs and should be excluded when comparing reference outputs.

## Video analysis through the web page

### Download the model

Download the **Pose Landmarker Full** model from the [official MediaPipe model page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models). The current [Full float16 model download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task) uses Google's `latest` alias; record the downloaded model's version or checksum when documenting an experiment.

Place the file at:

```text
models/pose_landmarker_full.task
```

The model is downloaded separately and is not included in this repository.

### Run the service

```powershell
.\.venv\Scripts\python.exe main.py
```

Open `http://127.0.0.1:9999` and upload an MP4 or MOV video. The API also accepts `POST /upload` requests with a multipart form field named `file`.

By default, the service listens on localhost with debugging disabled. It uses MediaPipe's VIDEO mode and analyzes one person. This Flask development service is intended for local research use.

Uploaded files are stored in `Uploads/`. Each analysis creates `output/<run_id>/` with an Excel workbook, landmark TXT, posture/additional/response JSON files, and `run.json`. The workbook includes frame-wise results, additional-point details, and a score summary. Optional visualization output is controlled by `SAVE_VISUALS`.

### Response fields

| Field | Meaning |
|---|---|
| `totalFrame` | Number of decoded video frames, or supplied frames in the landmark example |
| `BodyPosturePoint` | Basic body posture score |
| `Addition Point` | Additional points after applying the cap |
| `twistAndLanternal` | Points for the combined twisting/lateral-bending condition |
| `twist`, `laternal` | Separate frequency-based scores, reported for inspection |
| `distance of body`, `arm raise`, `above shoulder` | Additional points for the corresponding conditions |
| `total_score` | Basic body posture score plus capped additional points |
| `start`, `end` | Posture labels in the first maximum-scoring transition; not frame indices or times |
| `checkPersonInScreen` | Video: a person was detected in at least one frame. Landmark example: landmark frames were supplied; no image detection is performed |
| `timeStamp` | Analysis start time |

Field names preserve the existing API spelling. Unavailable scores may be `null`; this is different from a score of zero. Twisting and lateral bending are combined before scoring and are not added twice to the total.

## Research parameters and implementation settings

The first six rows below correspond to the temporal and spatial/angle parameters listed in manuscript Table 3. Geometry rules are implemented in `src/mediapipe_lib/base.py`; temporal settings are defined in the entry scripts.

| Parameter | Value / implemented rule |
|---|---|
| Effective-posture threshold K | 12 frames; 0.40 s at 30 FPS |
| Sustained-posture criterion | 3K = 36 frames; 1.20 s at 30 FPS |
| Trunk twisting | Shoulder–hip line angle in the XZ plane > 15° |
| Lateral bending | Absolute shoulder height difference > 0.16 m |
| Load away from body | Either wrist's XZ distance from the hip midpoint > 0.27 m |
| Raised arms | Either shoulder angle > 60°, with wrist height strictly between elbow and shoulder heights on that side |
| Hands above shoulders | Either wrist strictly higher than its corresponding shoulder |
| Frequency boundaries | Seldom: r ≤ 1/9; occasional: 1/9 < r ≤ 1/3; frequent: r > 1/3 |
| Additional-point cap | 6 points |

### Posture classification

The six classification angles are bilateral shoulder, hip, and knee angles. Rules are evaluated in the following order; a qualifying angle on either side is sufficient:

| Label | Rule |
|---|---|
| A5 | Knee angle < 90° |
| A4 | Otherwise, hip angle < 120° |
| A3 | Otherwise, hip angle < 160° |
| A2 | Otherwise, shoulder angle > 90° |
| A1 | Otherwise |

### Temporal rules

The first and last segments of each valid block are preserved. Intermediate segments shorter than K are removed. Identical labels becoming adjacent after filtering may merge; their lengths count retained frames. A retained run of at least 3K frames is represented by a repeated label for sustained-posture scoring. The highest score among the resulting adjacent label pairs becomes the basic body posture score. Invalid frames or gaps in frame numbering split the sequence into independent blocks.

K and 3K are frame counts and do not automatically scale with FPS. Change them manually to preserve the intended durations at other frame rates. `FPS_OVERRIDE` changes timestamp conversion without resampling the video; otherwise, video analysis uses the FPS reported by the input. The example uses `FPS = 30.0`.

### Additional-point calculation

For each item, r is the fraction of positive frames among that item's valid frames, before temporal filtering. Invalid frames are excluded from that item's denominator.

| Item | Seldom | Occasional | Frequent |
|---|---:|---:|---:|
| Trunk twisting OR lateral bending | 0 | 1 | 3 |
| Load away from body | 0 | 1 | 3 |
| Raised arms | 0 | 0.5 | 1 |
| Hands above shoulders | 0 | 1 | 2 |

Twisting and lateral bending use a frame-wise logical OR. Additional points are summed and capped at 6. These ratios describe temporal prevalence, not the count of discrete events.

## Pose estimation evaluation

`evaluate_pose_estimation.py` reads precomputed prediction and ground-truth JSON files; it does not generate predictions from videos. Inputs use the 12 named joints listed above. Both sources must already have matching coordinate axes, units, and frame correspondence.

Configure:

```python
PRED_FOLDER = BASE_DIR / 'data' / 'pred'
GT_FOLDER = BASE_DIR / 'data' / 'gt'
OUTPUT_FOLDER = BASE_DIR / 'metrics_output'
```

Files such as `Video1_0001_pred.json` and `Video1_1_GT.json` are paired by video identifier and numeric frame index, ignoring case and zero padding. Run:

```powershell
.\.venv\Scripts\python.exe evaluate_pose_estimation.py
```

- **MPJPE:** mean Euclidean position error across 12 joints; units follow the input coordinates.
- **MPJAE:** mean absolute angular error across eight angles: bilateral shoulders, elbows, hips, and knees; units are degrees.
- Per-video summaries contain the mean, minimum, and maximum of frame-wise metrics. `Overall` is the equally weighted mean of video means.
- Outputs: `metrics_per_file_by_video.xlsx`, `metrics_per_video.xlsx`, and `pairing_report.json` under `metrics_output/`.
- Unmatched files are rejected and existing outputs are not overwritten by default (`ALLOW_UNMATCHED = False`, `OVERWRITE = False`).

No translation, rotation, scale, axis, or temporal alignment is performed. This script was reconstructed from the existing evaluation workbooks; it is not the recovered original generator. Exact reproduction of historical MPJPE values is not guaranteed. The included posture demonstration does not provide an independent prediction/ground-truth evaluation dataset.

## Temporal threshold sensitivity analysis

```powershell
.\.venv\Scripts\python.exe temporal_sensitivity_analysis.py
```

The script reprocesses the embedded V1–V10 posture sequences, rather than re-estimating landmarks from videos. It runs:

1. K = 6, 9, 12, 15, 18, with the sustained criterion equal to 3K.
2. Fixed K = 12, with sustained criteria of 24, 36, 48, 60 frames.

The baseline is K = 12 and a sustained criterion of 36 frames. Stability is assessed by counting videos whose basic body posture score matches the baseline.

Outputs are `ablation.xlsx` in the repository root, and plots plus `results.json` under `addition/ablation/`. These retain the existing output names; the experiment is a temporal threshold sensitivity analysis.

## Reproducibility scope

This repository supports the supplied landmark demonstration, coordinate-based error calculation when prediction/GT data are provided, and temporal reprocessing of the embedded posture sequences. Original participant videos are not distributed here.

The Python video service is a research implementation and is not the mobile application's on-device runtime. Current code preserves valid-block endpoint segments and may merge labels after short-segment removal; these details should be considered alongside the manuscript's temporal description. Software versions, input FPS, visibility settings, and model files must be recorded when comparing results.

## License and versioning

Original source code is licensed under the [MIT License](LICENSE), unless otherwise noted. Third-party libraries and downloaded models remain subject to their respective licenses.

For manuscript-related use, cite the publication when available and identify the exact Git commit or published release used. Versioned downloads, when published, are available on the [Releases page](https://github.com/WindyRock1020/kim-lhc-pose-assessment/releases).
