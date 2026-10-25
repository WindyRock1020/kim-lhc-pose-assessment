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



