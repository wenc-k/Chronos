"""Make a Track 1 evaluation manifest using original identity images and prompts."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def build_rows(output_root, reference_root, manifest, image_name, prompt_name, video_name=None):
    rows = []
    seen = set()
    for case in sorted(output_root.iterdir()):
        if not case.is_dir():
            continue
        videos = [case / video_name] if video_name else sorted(case.glob("*.mp4"))
        videos = [path for path in videos if path.is_file() and path.stat().st_size]
        if not videos:
            continue
        if len(videos) != 1:
            raise ValueError(f"{case.name}: multiple MP4 files; specify --video_name")
        video = videos[0]
        meta = video.with_suffix(video.suffix + ".case.json")
        case_id = json.loads(meta.read_text(encoding="utf-8"))["id_index"] if meta.exists() else case.name
        original = (reference_root / case_id).resolve()
        if original.parent != reference_root.resolve():
            raise ValueError(f"{case.name}: invalid reference case ID")
        if not original.is_dir() and not meta.exists():
            # Legacy JSON batches numbered directories as 000001_<original ID>.
            number, sep, suffix = case.name.partition("_")
            if sep and len(number) == 6 and number.isdigit() and suffix:
                original = reference_root / suffix
                case_id = suffix
        ref = original / image_name
        text = original / prompt_name
        if not ref.is_file() or not text.is_file():
            raise ValueError(f"{case.name}: missing original image or prompt")
        if case_id in seen:
            raise ValueError(f"{case.name}: multiple videos map to the same reference case")
        seen.add(case_id)
        rows.append({
            "id_index": case_id,
            "ref_img_path": Path(os.path.relpath(ref.resolve(), manifest.parent.resolve())).as_posix(),
            "prompt": text.read_text(encoding="utf-8-sig").strip(),
            "video_path": Path(os.path.relpath(video.resolve(), manifest.parent.resolve())).as_posix(),
        })
    if not rows:
        raise ValueError("No generated MP4 files were found")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", required=True, type=Path)
    p.add_argument("--reference_root", required=True, type=Path)
    p.add_argument("--output_json", type=Path)
    p.add_argument("--image_name", default="image.png")
    p.add_argument("--prompt_name", default="prompt1.txt")
    p.add_argument("--video_name", help="Select an exact filename when a case contains multiple videos")
    args = p.parse_args()
    target = args.output_json or args.output_root / "result.json"
    try:
        rows = build_rows(args.output_root, args.reference_root, target, args.image_name, args.prompt_name, args.video_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as f:
            tmp = Path(f.name)
            try:
                json.dump(rows, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.close()
                tmp.replace(target)
            finally:
                f.close()
                tmp.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError) as error:
        p.exit(1, f"Could not build evaluation manifest: {error}\n")
    print(f"Wrote {len(rows)} cases")


if __name__ == "__main__":
    main()
