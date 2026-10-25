"""Flask video upload and frame-wise pose analysis; see README.md."""
import json
import math
from uuid import uuid4
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Settings live here; no command-line arguments are required.
HOST = '127.0.0.1'
PORT = 9999
DEBUG = False
SAVE_VISUALS = False
VISIBILITY_THRESHOLD = None  # None disables visibility filtering; 0.50 enables it.
FPS_OVERRIDE = None
CON_K = 12
CSV_K = 3 * CON_K
MODEL = BASE_DIR / 'models/pose_landmarker_full.task'

server = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))
server.json.ensure_ascii = False
server.json.sort_keys = False
server.config.update(UPLOAD_FOLDER=BASE_DIR / 'Uploads',
                     OUTPUT_FOLDER=BASE_DIR / 'output',
                     ALLOWED_EXTENSIONS={'mp4', 'mov'})


def allowed_file(filename):
    return Path(filename).suffix.lower().lstrip('.') in server.config['ALLOWED_EXTENSIONS']


@server.route('/')
def index():
    return render_template('index.html')


@server.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(status='failed', error='No file part'), 400
    file = request.files['file']
    if not file.filename:
        return jsonify(status='failed', error='No selected file'), 400
    if not allowed_file(file.filename):
        return jsonify(status='failed', error='Only MP4 and MOV files are accepted'), 400
    original_name = file.filename.replace('\\', '/').rsplit('/', 1)[-1]
    stem = secure_filename(Path(original_name).stem) or 'video'
    run_id = f'{stem}_3D_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}'
    upload_dir = Path(server.config['UPLOAD_FOLDER'])
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / f'{run_id}{Path(original_name).suffix.lower()}'
        file.save(video_path)
        if video_path.stat().st_size == 0:
            return jsonify(status='failed', error='Uploaded file is empty'), 400
        result = process_video(video_path, server.config['OUTPUT_FOLDER'], MODEL,
                               SAVE_VISUALS, VISIBILITY_THRESHOLD, FPS_OVERRIDE,
                               CON_K, CSV_K, run_id=run_id, original_name=original_name)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify(status='failed', error=str(exc)), 400
    except Exception:
        server.logger.exception('Video analysis failed: %s', run_id)
        return jsonify(status='failed', error='Video analysis failed; check server logs',
                       run_id=run_id), 500
# Used by the ablation-style temporal scoring stage.
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
    return {'frame_scope': 'all decoded frames before ablation filtering',
            'denominator': 'valid frames for each item; invalid frames excluded',
            'trunk_combination': 'logical OR; True with unknown is True, False with unknown is unknown',
            'status': 'complete' if complete else 'incomplete', 'items': items,
            'uncapped_total': raw_total, 'cap': 6.0, 'additional_points': capped,
            'base_posture_score': base_score,
            'body_posture_total': base_score + capped if base_score is not None and capped is not None else None}


def filter_postures(rows, label_field, con_k=12, csv_k=36):
    """Apply ablation rules separately to contiguous valid runs; preserve frame IDs."""
    from src.temporal import analyze_sequence
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


def process_video(video_path, output_root=None, model_path=None, visuals=False,
                  visibility=None, fps_override=None, con_k=12, csv_k=None,
                  run_id=None, original_name=None):
    video_path = Path(video_path).resolve()
    model_path = Path(model_path or BASE_DIR / 'models/pose_landmarker_full.task').resolve()
    for path in (video_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f'File not found: {path}')
    if visibility is not None and (not math.isfinite(visibility) or not 0 <= visibility <= 1):
        raise ValueError('visibility must be between 0 and 1')
    csv_k = 3 * con_k if csv_k is None else csv_k
    if type(con_k) is not int or type(csv_k) is not int or con_k <= 0 or csv_k <= 0:
        raise ValueError('con_k and csv_k must be positive integers')
    import cv2
    import mediapipe as mp
    import pandas as pd
    import openpyxl  
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    if visuals:
        import matplotlib
        matplotlib.use('Agg')
        from src.draw_diagram import DrawKPT, DrawOverlay
    started_at = datetime.now()
    run_name = run_id or f'{video_path.stem}_3D_{started_at:%Y%m%d_%H%M%S_%f}'
    run_dir = Path(output_root or BASE_DIR / 'output').resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'input': str(video_path), 'model': str(model_path),
                'run_id': run_name, 'original_filename': original_name or video_path.name,
                'visibility_threshold': visibility, 'visibility_filter_enabled': visibility is not None,
                'visuals': visuals,
                'coordinates': 'TXT and P*Y use upward-positive Y; model Y is inverted.',
                'invalid_values': 'Blank cells are invalid, not zero or False.',
                'mediapipe_version': mp.__version__, 'opencv_version': cv2.__version__}
    def save_metadata():
        (run_dir / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    save_metadata()
    cap = landmarker = None
    writers = {}
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError('Cannot open input video')
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = float(fps_override) if fps_override is not None else source_fps
        if not math.isfinite(fps) or fps <= 0 or fps > 1000:
            raise ValueError('Invalid FPS; specify --fps between 0 (exclusive) and 1000')
        metadata.update(source_fps=source_fps if math.isfinite(source_fps) else None,
                        analysis_fps=fps, timestamp_mode='frame index / FPS; no resampling',
                        source_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                        source_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        landmarker = PoseLandmarker.create_from_options(options)
        rows = []
        label_field = f'{Path(original_name or video_path.name).stem}_label'
        txt_path = run_dir / f'{run_name}.txt'
        with txt_path.open('w', encoding='utf-8') as output:
            output.write('BlazePose 3D Landmarks (before visibility filtering)\nIndex, X, Y(up), Z, Visibility, Presence\n')
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                index = len(rows)
                timestamp_ms = round(index * 1000 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                rows.append(analyze_result(result, index, timestamp_ms, label_field, visibility))
                output.write(f'Frame: {index}; timestamp_ms: {timestamp_ms}\n')
                if result.pose_world_landmarks:
                    for i, lm in enumerate(result.pose_world_landmarks[0]):
                        output.write(f'{i}:({lm.x}, {-lm.y}, {lm.z}, {lm.visibility}, {lm.presence})\n')
                else:
                    output.write('NO_POSE\n')
                if visuals:
                    for kind in ('kpt', 'overlay'):
                        folder = run_dir / kind
                        folder.mkdir(exist_ok=True)
                        png = folder / f'{index:06d}.png'
                        if kind == 'kpt' and result.pose_world_landmarks:
                            DrawKPT(result.pose_world_landmarks[0], str(png))
                        elif kind == 'overlay' and result.pose_landmarks:
                            DrawOverlay(frame, result.pose_landmarks[0], str(png))
                        else:
                            import numpy as np
                            placeholder = frame.copy() if kind == 'overlay' else np.zeros_like(frame)
                            cv2.putText(placeholder, 'NO POSE', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            if not cv2.imwrite(str(png), placeholder):
                                raise OSError(f'Cannot write {png}')
                        rendered = cv2.imread(str(png))
                        if rendered is None:
                            raise OSError(f'Cannot read {png}')
                        if kind not in writers:
                            height, width = rendered.shape[:2]
                            writer = cv2.VideoWriter(str(folder / f'{run_name}_{kind}.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                            writers[kind] = (writer, (width, height))
                            if not writer.isOpened():
                                raise OSError(f'Cannot create {kind} video')
                        writer, size = writers[kind]
                        writer.write(cv2.resize(rendered, size))
        if not rows:
            raise ValueError('No decodable frames')
        xlsx_path = run_dir / f'{run_name}.xlsx'
        posture = filter_postures(rows, label_field, con_k, csv_k)
        additional = calculate_additional_points(rows, posture['max_score'])
        posture['additional_points'] = additional['additional_points']
        posture['body_posture_total'] = additional['body_posture_total']
        posture['effective_duration_seconds'] = con_k / fps
        posture['sustained_duration_seconds'] = csv_k / fps
        posture_path = run_dir / f'{run_name}_posture.json'
        posture_path.write_text(json.dumps(posture, ensure_ascii=False, indent=2), encoding='utf-8')
        additional_path = run_dir / f'{run_name}_additional.json'
        additional_path.write_text(json.dumps(additional, ensure_ascii=False, indent=2), encoding='utf-8')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as workbook:
            pd.DataFrame(rows).set_index('frame').T.to_excel(workbook, sheet_name='per_frame')
            pd.DataFrame([{'item': name, **item} for name, item in additional['items'].items()]).to_excel(
                workbook, sheet_name='additional_points', index=False)
            pd.DataFrame([{'metric': key, 'value': additional[key]} for key in
                          ('status', 'frame_scope', 'denominator', 'uncapped_total', 'cap',
                           'additional_points', 'base_posture_score', 'body_posture_total')]).to_excel(
                workbook, sheet_name='score_summary', index=False)
        metadata.update(status='complete', frames=len(rows),
                        valid_posture_frames=sum(row['posture_valid'] for row in rows),
                        excel=str(xlsx_path), landmarks=str(txt_path),
                        posture_result=str(posture_path), posture=posture,
                        additional_result=str(additional_path), additional=additional)
        response = build_response(rows, posture, additional,
                                  started_at.strftime('%Y-%m-%dT%H-%M-%S-%f'),
                                  any(row['pose_detected'] for row in rows))
        response_path = run_dir / f'{run_name}_response.json'
        response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding='utf-8')
        metadata['response_result'] = str(response_path)
        return response
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        for writer, _ in writers.values():
            writer.release()
        if cap is not None:
            cap.release()
        if landmarker is not None:
            landmarker.close()
        save_metadata()


if __name__ == '__main__':
    server.run(host=HOST, port=PORT, debug=DEBUG)
