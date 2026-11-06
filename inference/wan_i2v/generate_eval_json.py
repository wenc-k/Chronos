"""Add generated Track 2 video paths to an evaluation metadata manifest."""
import argparse
import copy
import json
from pathlib import Path

from _workflow import write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Original evaluation metadata JSON')
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == manifest:
        parser.error('Choose a separate output path for the augmented manifest')
    original = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if not isinstance(original, dict):
        parser.error('The original Track 2 evaluation manifest must be an object keyed by case ID')
    result = copy.deepcopy(original)
    video_root = args.video_root.expanduser().resolve()
    missing = []
    for case_id, case in result.items():
        if not str(case_id).isdecimal() or not isinstance(case, dict):
            parser.error('Cases must have numeric IDs and object metadata')
        for field in ('img_paths', 'class_label', 'global_prompt', 'temporal_captions'):
            if field not in case:
                parser.error(f'Case {case_id} lacks required evaluation field {field}')
        # Keep reference-image paths valid even when the new manifest has a different parent.
        if not isinstance(case['img_paths'], list):
            parser.error(f'Case {case_id} img_paths must be a list')
        resolved_images = []
        for image in case['img_paths']:
            image = Path(image).expanduser()
            if not image.is_absolute():
                image = manifest.parent / image
            resolved_images.append(str(image.resolve()))
        case['img_paths'] = resolved_images
        paths = [video_root / str(case_id) / f'{index}.mp4' for index in range(len(case['temporal_captions']))]
        final = video_root / 'final' / f'{case_id}.mp4'
        case['video_path'] = str(final)
        case['segment_video_paths'] = [str(path) for path in paths]
        missing.extend(str(path) for path in [final, *paths] if not path.is_file() or path.stat().st_size == 0)
    write_json_atomic(output, result)
    print(f'Wrote {len(result)} cases. Missing or empty video files: {len(missing)}')
    if missing:
        for path in missing:
            print(path)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
