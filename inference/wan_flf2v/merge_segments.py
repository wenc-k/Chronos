"""Merge ordered Track 2 video segments using the experiment's FFmpeg settings."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def identify(name):
    legacy = re.fullmatch(r"(.+)_([0-9]+)_[0-9]+_\.mp4", name)
    match = legacy or re.fullmatch(r"(.+)_([0-9]+)\.mp4", name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def numeric_order(text):
    return (0, int(text)) if text.isdigit() else (1, text)


def segment_groups(folder, output_dir, expected_jobs=None):
    groups = {}
    for video in sorted(folder.rglob("*.mp4")):
        if output_dir.resolve() in video.resolve().parents:
            continue
        identity = identify(video.name)
        if identity is None:
            continue
        case, part = identity
        if not video.stat().st_size:
            raise ValueError(f"{video.name}: empty segment")
        parts = groups.setdefault(case, {})
        if part in parts:
            raise ValueError(f"{case}_{part}: duplicate segment versions")
        parts[part] = video
    if not groups:
        raise ValueError("No segment videos were found")
    expected = None
    if expected_jobs:
        expected = {}
        with expected_jobs.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                identity = identify(str(json.loads(line)["id"]) + ".mp4")
                if not identity:
                    raise ValueError("Manifest id must end with _<segment number>")
                case, part = identity
                parts = expected.setdefault(case, set())
                if part in parts:
                    raise ValueError("Manifest contains duplicate segments")
                parts.add(part)
        if set(groups) != set(expected):
            raise ValueError("Generated cases differ from the expected manifest")
    for case, parts in groups.items():
        found = set(parts)
        required = expected[case] if expected is not None else set(range(max(parts) + 1))
        if found != required or required != set(range(max(required) + 1)):
            raise ValueError(f"{case}: missing or unexpected segment index")
    return groups


def ffmpeg_escape(path):
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def merge(parts, target):
    # The original procedure keeps both endpoints at every segment boundary.
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=target.parent) as folder:
        folder = Path(folder)
        listing = folder / "segments.txt"
        listing.write_text("".join("file '" + ffmpeg_escape(parts[n]) + "'\n" for n in sorted(parts)), encoding="utf-8")
        temp = folder / "merged.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(temp)], check=True)
        if not temp.is_file() or not temp.stat().st_size:
            raise ValueError("FFmpeg produced no video")
        temp.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segments-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--jobs", type=Path, help="Original JSONL to verify that every expected segment is present")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not shutil.which("ffmpeg"):
        p.error("ffmpeg must be installed and available on PATH")
    if args.segments_dir.resolve() == args.output_dir.resolve():
        p.error("output-dir must differ from segments-dir")
    try:
        groups = segment_groups(args.segments_dir, args.output_dir, args.jobs)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for case in sorted(groups, key=numeric_order):
            target = args.output_dir / (case + ".mp4")
            if target.is_file() and target.stat().st_size and not args.overwrite:
                print(f"{case}: skipped")
                continue
            merge(groups[case], target)
            print(f"{case}: merged {len(groups[case])} segments")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        p.exit(1, "Merge failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
