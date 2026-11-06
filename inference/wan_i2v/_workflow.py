"""Input validation and atomic result handling for the Wan workflows."""
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile


def prepare_track2_data(data, manifest, video_duration, video_fps):
    if not isinstance(data, dict) or not data:
        raise ValueError('Track 2 input must be a nonempty object keyed by numeric case IDs')
    if video_duration <= 0 or video_fps <= 0:
        raise ValueError('Video duration and FPS must be positive')
    parent = Path(manifest).expanduser().resolve().parent
    numeric_ids = set()
    for case_id, case in data.items():
        if not isinstance(case_id, str) or not case_id.isdecimal():
            raise ValueError('Track 2 case IDs must be nonnegative decimal strings')
        if int(case_id) in numeric_ids:
            raise ValueError('Track 2 case IDs must identify distinct numeric values')
        numeric_ids.add(int(case_id))
        if not isinstance(case, dict) or not isinstance(case.get('first_frame'), str):
            raise ValueError(f'Case {case_id} needs a first_frame path')
        frame = Path(case['first_frame']).expanduser()
        if not frame.is_absolute():
            frame = parent / frame
        case['first_frame'] = str(frame.resolve())
        if not frame.is_file():
            raise ValueError(f'Case {case_id} first frame does not exist')
        captions = case.get('temporal_captions')
        if not isinstance(captions, list) or not captions:
            raise ValueError(f'Case {case_id} needs temporal_captions')
        previous_end = 0.0
        for caption in captions:
            if not isinstance(caption, dict) or not isinstance(caption.get('description'), str) or not caption['description'].strip():
                raise ValueError(f'Case {case_id} has an empty segment description')
            try:
                start, end = float(caption['start']), float(caption['end'])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f'Case {case_id} needs numeric start and end values for every segment') from None
            if not math.isfinite(start) or not math.isfinite(end) or not (0 <= start < end <= 1) or start < previous_end:
                raise ValueError(f'Case {case_id} segment intervals must be ordered, nonoverlapping fractions in [0, 1]')
            caption['start'], caption['end'] = start, end
            previous_end = end
    return data


def track2_case_complete(output, case_id, case):
    root = Path(output)
    expected = [root / 'final' / f'{case_id}.mp4']
    expected.extend(root / str(case_id) / f'{index}.mp4' for index in range(len(case['temporal_captions'])))
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.' + path.name,
                            suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def merge_track1(destination, parts):
    destination = Path(destination)
    paths = ([destination] if destination.exists() else []) + sorted(parts)
    combined = {}
    for path in paths:
        records = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(records, list):
            raise ValueError(f'Invalid result list in {path.name}')
        for record in records:
            if not isinstance(record, dict) or any(k not in record for k in ('id_index', 'ref_img_path', 'video_path', 'prompt')):
                raise ValueError(f'Invalid result record in {path.name}')
            key = str(record['id_index']), str(record['video_path'])
            combined[key] = record
    write_json_atomic(destination, list(combined.values()))
