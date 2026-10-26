"""Analyze per-frame upward-positive XYZ JSON using src; no server/model inference.
Run: python analyze_landmark_example.py
Configure folders, FPS and thresholds below; no command-line arguments.
"""
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# Scoring and temporal filtering are implemented locally below.
def build_segments(sequence):
    """切出連續區間，start/end 為原始的零起算索引（end 不包含）。"""
    segments = []
    for index, label in enumerate(sequence):
        if not segments or segments[-1]["label"] != label:
            segments.append({"start": index, "end": index + 1, "label": label})
        else:
            segments[-1]["end"] = index + 1
    return segments


def score_changes(changes, pose_scores):
    """對相鄰值計分；查表不分方向，紀錄仍保留實際方向及所有並列。"""
    scored = []
    for left, right in zip(changes, changes[1:]):
        scored.append((left, right, pose_scores[tuple(sorted((left, right)))]))
    maximum = max((score for _, _, score in scored), default=None)
    best = list(dict.fromkeys((a, b) for a, b, score in scored if score == maximum))
    return maximum, best


def analyze_sequence(sequence, con_k, csv_k, pose_scores):
    """逐一刪除中間短區間，再將有效影格合併成數字變化。"""
    if con_k <= 0 or csv_k <= 0:
        raise ValueError("con_k and csv_k must be positive")
    if any(label not in range(1, 6) for label in sequence):
        raise ValueError("pose labels must be integers from 1 to 5")
    segments = build_segments(sequence)
    original_changes = [segment["label"] for segment in segments]
    filtered_x, filtered_y = [], []
    for index, segment in enumerate(segments):
        length = segment["end"] - segment["start"]
        if index in (0, len(segments) - 1) or length >= con_k:
            filtered_x.extend(range(segment["start"], segment["end"]))
            filtered_y.extend([segment["label"]] * length)
    changes = []
    for segment in build_segments(filtered_y):
        changes.append(segment["label"])
        if segment["end"] - segment["start"] >= csv_k:
            changes.append(segment["label"])
    maximum, best = score_changes(changes, pose_scores)
    original_maximum, original_best = score_changes(original_changes, pose_scores)
    return {
        "original_changes": original_changes,
        "filtered_x": filtered_x,
        "filtered_y": filtered_y,
        "label_changes": changes,
        "max_score": maximum,
        "max_changes": best,
        "original_max_score": original_maximum,
        "original_max_changes": original_best,
        "valid_frames": len(filtered_y),
        "removed_frames": len(sequence) - len(filtered_y),
    }





pose_scores = {
    (1, 1): 0, (1, 2): 3, (1, 3): 3, (1, 4): 7, (1, 5): 9,
    (2, 2): 5, (2, 3): 5, (2, 4): 10, (2, 5): 13,
    (3, 3): 5, (3, 4): 10, (3, 5): 13,
    (4, 4): 15, (4, 5): 18, (5, 5): 20,
}


JOINTS = {
    'theta11': ('left_shoulder', (13, 23, 11)),
    'theta12': ('right_shoulder', (14, 24, 12)),
    'theta23': ('left_hip', (11, 25, 23)),
    'theta24': ('right_hip', (12, 26, 24)),
    'theta25': ('left_knee', (23, 27, 25)),
    'theta26': ('right_knee', (24, 28, 26)),
}


FIELDS = [*JOINTS, 'theta_twist', 'l1112Y', 'maxl1516XZ',
          *[f'P{i}Y' for i in (11, 12, 13, 14, 15, 16)],
          'twist', 'lateral', 'gravity', 'armsRaised', 'aboveShoulder']


def valid_landmarks(landmarks, indices, threshold):
    return all(i < len(landmarks) and
               all(v is not None and math.isfinite(v) for v in
                   (landmarks[i].x, landmarks[i].y, landmarks[i].z))
               and (threshold is None or
                    (landmarks[i].visibility is not None and
                     math.isfinite(landmarks[i].visibility) and
                     landmarks[i].visibility >= threshold)) for i in indices)


def analyze_result(result, index, timestamp_ms, label_field, threshold):
    row = {'frame': index, 'timestamp_ms': timestamp_ms,
           'pose_detected': bool(result.pose_world_landmarks), 'posture_valid': False,
           label_field: None, **dict.fromkeys(FIELDS)}
    if not result.pose_world_landmarks:
        return row
    from src.mediapipe_lib.base import PoseResult, ResultAnalyzer
    analyzer = ResultAnalyzer(PoseResult(result))
    landmarks = result.pose_world_landmarks[0]

    def evaluate(indices, action):
        if not valid_landmarks(landmarks, indices, threshold):
            return None
        value = action()
        return value if math.isfinite(float(value)) else None

    for field, (joint, indices) in JOINTS.items():
        row[field] = evaluate(indices, lambda joint=joint: analyzer.get_joint_angle_by_name(joint, True))
    if all(row[field] is not None for field in JOINTS):
        row[label_field] = {f'A{i}': i for i in range(1, 6)}.get(analyzer.get_lhc_label(True))
        row['posture_valid'] = row[label_field] is not None
    rules = [
        ('theta_twist', (11, 12, 23, 24), analyzer.get_pose_shoulder_hip_staggered_angle_xz),
        ('l1112Y', (11, 12), analyzer.get_pose_shoulder_height),
        ('maxl1516XZ', (15, 16, 23, 24), analyzer.get_pose_gravity_distance),
        ('twist', (11, 12, 23, 24), lambda: analyzer.check_if_trunk_is_twisted(True)),
        ('lateral', (11, 12), lambda: analyzer.check_if_trunk_is_lateral_inclination(True)),
        ('gravity', (15, 16, 23, 24), lambda: analyzer.check_if_hands_at_a_distance(True)),
        ('armsRaised', (11, 12, 13, 14, 15, 16, 23, 24), lambda: analyzer.check_if_arms_raised(True)),
        ('aboveShoulder', (11, 12, 15, 16), lambda: analyzer.check_if_hands_above_shoulder(True)),
    ]
    for field, indices, action in rules:
        row[field] = evaluate(indices, action)
    for i in (11, 12, 13, 14, 15, 16):
        row[f'P{i}Y'] = evaluate((i,), lambda i=i: -landmarks[i].y)
    return row


def build_response(rows, posture, additional, timestamp, person_in_screen):
    """Compact public response; keep research details in separate output files."""
    best = posture['max_changes'][0] if posture['max_changes'] else (None, None)
    return {
        'totalFrame': len(rows),
        'BodyPosturePoint': posture['max_score'],
        'Addition Point': additional['additional_points'],
        'twistAndLanternal': additional['items']['trunk_twist_or_lateral']['points'],
        'twist': score_frequency([row.get('twist') for row in rows], 1.0, 3.0)['points'],
        'laternal': score_frequency([row.get('lateral') for row in rows], 1.0, 3.0)['points'],
        'distance of body': additional['items']['load_away_from_body']['points'],
        'arm raise': additional['items']['arms_raised']['points'],
        'above shoulder': additional['items']['hands_above_shoulder']['points'],
        'total_score': additional['body_posture_total'],
        'start': best[0],
        'end': best[1],
        'checkPersonInScreen': bool(person_in_screen),
        'timeStamp': timestamp,
    }


def score_frequency(values, occasional, frequent):
    valid = [value for value in values if value is not None]
    positive = sum(bool(value) for value in valid)
    count = len(valid)
    if not count:
        frequency, points, ratio = 'unavailable', None, None
    else:
        ratio = positive / count
        if positive * 3 > count:
            frequency, points = 'frequent', frequent
        elif positive * 9 > count:
            frequency, points = 'occasional', occasional
        else:
            frequency, points = 'seldom', 0.0
    return {'positive_frames': positive, 'valid_frames': count,
            'invalid_frames': len(values) - count, 'ratio': ratio,
            'frequency': frequency, 'points': points}


def calculate_additional_points(rows, base_score):
    """Score prevalence over valid raw frames per item, before temporal filtering."""
    def combine_trunk(row):
        twist, lateral = row.get('twist'), row.get('lateral')
        # True OR unknown is known True; False OR unknown remains unknown.
        if (twist is not None and bool(twist)) or (lateral is not None and bool(lateral)):
            return True
        if twist is None or lateral is None:
            return None
        return False

    definitions = [
        ('trunk_twist_or_lateral', combine_trunk, 1.0, 3.0),
        ('load_away_from_body', lambda row: row.get('gravity'), 1.0, 3.0),
        ('arms_raised', lambda row: row.get('armsRaised'), 0.5, 1.0),
        ('hands_above_shoulder', lambda row: row.get('aboveShoulder'), 1.0, 2.0),
    ]
    items = {}
    for name, extract, occasional, frequent in definitions:
        values = [extract(row) for row in rows]
        items[name] = score_frequency(values, occasional, frequent)
    complete = all(item['points'] is not None for item in items.values())
    raw_total = sum(item['points'] for item in items.values()) if complete else None
    capped = min(raw_total, 6.0) if raw_total is not None else None
    return {'frame_scope': 'all supplied landmark frames before temporal filtering',
            'denominator': 'valid frames for each item; invalid frames excluded',
            'trunk_combination': 'logical OR; True with unknown is True, False with unknown is unknown',
            'status': 'complete' if complete else 'incomplete', 'items': items,
            'uncapped_total': raw_total, 'cap': 6.0, 'additional_points': capped,
            'base_posture_score': base_score,
            'body_posture_total': base_score + capped if base_score is not None and capped is not None else None}


def filter_postures(rows, label_field, con_k=12, csv_k=36):
    """Apply ablation rules separately to contiguous valid runs; preserve frame IDs."""
    if type(con_k) is not int or type(csv_k) is not int or con_k <= 0 or csv_k <= 0:
        raise ValueError('con_k and csv_k must be positive integers')
    blocks = []
    pending = []

    def flush():
        if not pending:
            return
        result = analyze_sequence([row[label_field] for row in pending], con_k, csv_k, pose_scores)
        frame_ids = [pending[i]['frame'] for i in result['filtered_x']]
        for i in result['filtered_x']:
            pending[i]['filtered_label'] = pending[i][label_field]
            pending[i]['temporal_status'] = 'retained'
        blocks.append({'block': len(blocks), 'start_frame': pending[0]['frame'],
                       'end_frame': pending[-1]['frame'], **result,
                       'filtered_x': frame_ids})
        pending.clear()

    previous_frame = None
    for row in rows:
        row['filtered_label'] = None
        row['temporal_status'] = 'invalid'
        valid = row['posture_valid'] and row[label_field] in range(1, 6)
        if not valid or (previous_frame is not None and row['frame'] != previous_frame + 1):
            flush()
        if valid:
            row['temporal_status'] = 'removed_short_segment'
            pending.append(row)
        previous_frame = row['frame']
    flush()
    scores = [block['max_score'] for block in blocks if block['max_score'] is not None]
    maximum = max(scores, default=None)
    best = list(dict.fromkeys(pair for block in blocks if block['max_score'] == maximum
                             for pair in block['max_changes']))
    return {'con_k': con_k, 'csv_k': csv_k, 'threshold_unit': 'frames',
            'invalid_frame_policy': 'split into independent valid blocks; preserve each block endpoints',
            'original_sequence': [row[label_field] for row in rows],
            'filtered_sequence': [row['filtered_label'] for row in rows],
            'retained_frames': sum(row['temporal_status'] == 'retained' for row in rows),
            'removed_frames': sum(row['temporal_status'] == 'removed_short_segment' for row in rows),
            'invalid_frames': sum(row['temporal_status'] == 'invalid' for row in rows),
            'max_score': maximum, 'max_changes': best, 'blocks': blocks}

BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = Path('./example') 
OUTPUT_FOLDER = BASE_DIR / 'example_output'
FPS = 30.0
CON_K = 12
CSV_K = 36
JOINT_IDS = {
    'LWrist': 15, 'RWrist': 16, 'LElbow': 13, 'RElbow': 14,
    'LShoulder': 11, 'RShoulder': 12, 'LHip': 23, 'RHip': 24,
    'LKnee': 25, 'RKnee': 26, 'LAnkle': 27, 'RAnkle': 28,
}


def read_frames(folder):
    frames = {}
    prefixes = set()
    for path in sorted(Path(folder).glob('*.json')):
        match = re.fullmatch(r'(.+)[_-](\d+)', path.stem)
        if not match:
            raise ValueError(f'Expected name_<frame>.json: {path.name}')
        prefixes.add(match[1])
        frame = int(match[2])
        if frame in frames:
            raise ValueError(f'Duplicate frame: {frame}')
        points = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(points, dict) or set(points) != set(JOINT_IDS):
            raise ValueError(f'{path.name}: expected exactly the 12 named joints')
        for name, point in points.items():
            if (not isinstance(point, list) or len(point) != 3 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(v) for v in point)):
                raise ValueError(f'{path.name}: invalid XYZ for {name}')
        frames[frame] = points
    if not frames:
        raise ValueError(f'No frame JSON files found in {folder}')
    if len(prefixes) != 1:
        raise ValueError('Input must contain only one sequence (one filename prefix)')
    return sorted(frames.items())


def as_pose_result(points):
    # Missing, unused landmarks stay invalid rather than becoming zero coordinates.
    landmarks = [SimpleNamespace(x=math.nan, y=math.nan, z=math.nan,
                                 visibility=None, presence=None) for _ in range(33)]
    for name, idx in JOINT_IDS.items():
        x, y_up, z = points[name]
        # Shared analyzer expects MediaPipe's Y-down coordinates and flips Y itself.
        landmarks[idx] = SimpleNamespace(x=x, y=-y_up, z=z,
                                        visibility=None, presence=None)
    return SimpleNamespace(pose_world_landmarks=[landmarks], pose_landmarks=[])


def analyze_frames(frames, fps=FPS, con_k=CON_K, csv_k=CSV_K):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite')
    rows = [analyze_result(as_pose_result(points), frame,
                           round(frame * 1000 / fps), 'posture_label', None)
            for frame, points in frames]
    # Shared geometry routines may return NumPy scalar booleans/numbers.
    rows = [{key: value.item() if hasattr(value, 'item') else value
             for key, value in row.items()} for row in rows]
    posture = filter_postures(rows, 'posture_label', con_k, csv_k)
    additional = calculate_additional_points(rows, posture['max_score'])
    additional['frame_scope'] = 'all supplied JSON frames before temporal filtering'
    return rows, posture, additional


def run():
    frames = read_frames(INPUT_FOLDER)
    rows, posture, additional = analyze_frames(frames)
    now = datetime.now()
    response = build_response(rows, posture, additional,
                              now.strftime('%Y-%m-%dT%H-%M-%S-%f'), bool(rows))
    # Unique run directory avoids overwriting earlier expected outputs.
    out = OUTPUT_FOLDER / now.strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        'input_folder': str(INPUT_FOLDER.resolve()), 'frames': len(frames),
        'frame_ids': [frame for frame, _ in frames], 'fps': FPS,
        'con_k': CON_K, 'csv_k': CSV_K, 'visibility_filter_enabled': False,
        'input_coordinates': 'XYZ, upward-positive Y; no alignment or normalization',
        'person_flag_meaning': 'landmark frames supplied; no image-based detection performed',
        'score_scope': 'body posture plus additional points, not the full KIM-LHC risk score',
    }
    for name, obj in [('response', response), ('posture', posture),
                      ('additional', additional), ('run', metadata)]:
        (out / f'{name}.json').write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
            encoding='utf-8')
    with (out / 'per_frame.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['field', *[row['frame'] for row in rows]])
        for field in rows[0]:
            writer.writerow([field, *[row.get(field) for row in rows]])
    print(json.dumps(response, ensure_ascii=False, indent=2))
    print(f'Output: {out}')
    return out


if __name__ == '__main__':
    run()
