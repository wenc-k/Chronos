# Adapted from the benchmark working tree for portable evaluation.
import os
import re
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_ROOT = SCRIPT_DIR.parent / "ckpt"

HUMAN_CLASSES = {
    "person", "human", "man", "woman", "boy", "girl", "male", "female",
    "child", "kid", "adult", "elderly man", "elderly woman", "young man", "young woman",
}

CORE_FIELDS = {
    "case_id", "id_index", "img_paths", "class_label", "global_prompt", "temporal_captions",
    "video_path", "full_video_path", "segment_video_paths", "segments", "error",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_link_or_copy(src: Path, dst: Path, overwrite: bool = False) -> None:
    ensure_dir(dst.parent)
    if overwrite and (dst.exists() or dst.is_symlink()):
        dst.unlink()
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def mean_valid(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def weighted_mean(values: List[Optional[float]], weights: List[float]) -> Optional[float]:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None and float(w) > 0]
    if not pairs:
        return None
    return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))


def is_human_class(label: Any) -> bool:
    return str(label).strip().lower() in HUMAN_CLASSES


def sanitize_key(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    return x.strip("_") or "case"


def normalize_cases(raw: Any) -> List[Dict[str, Any]]:
    """Support both Track2 dict-by-id and list-of-dicts inputs."""
    if isinstance(raw, dict):
        out = []
        for k, v in raw.items():
            if not isinstance(v, dict):
                raise TypeError(f"Case `{k}` must be a dict, got {type(v)}")
            row = dict(v)
            row.setdefault("case_id", str(k))
            row.setdefault("id_index", str(k))
            out.append(row)
        return out
    if isinstance(raw, list):
        out = []
        for i, v in enumerate(raw):
            if not isinstance(v, dict):
                raise TypeError(f"Case index {i} must be a dict, got {type(v)}")
            row = dict(v)
            case_id = row.get("case_id", row.get("id_index", f"item_{i:06d}"))
            row.setdefault("case_id", str(case_id))
            row.setdefault("id_index", str(case_id))
            return_key = str(case_id)
            row["case_id"] = return_key
            out.append(row)
        return out
    raise TypeError("input_json must be either a dict keyed by case id or a list of case dicts")


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_path(raw: Optional[str], root: Optional[Path] = None) -> Optional[Path]:
    if raw is None or raw == "":
        return None
    p = Path(str(raw))
    if p.is_absolute():
        return p
    if root is not None:
        return root / p
    return p


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def extract_segment_with_ffmpeg(full_video: Path, dst: Path, start_ratio: float, end_ratio: float, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        return
    ensure_dir(dst.parent)
    dur = ffprobe_duration(full_video)
    start = max(0.0, min(1.0, float(start_ratio))) * dur
    end = max(0.0, min(1.0, float(end_ratio))) * dur
    if end <= start:
        raise ValueError(f"Invalid segment start/end: {start_ratio}, {end_ratio}")
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-ss", f"{start:.6f}", "-to", f"{end:.6f}", "-i", str(full_video),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def get_case_full_video(case: Dict[str, Any], args, json_root: Path) -> Path:
    root = Path(args.full_video_dir) if args.full_video_dir else None
    for key in ("video_path", "full_video_path"):
        p = resolve_path(case.get(key), root=json_root)
        if p is not None and p.exists():
            return p
    case_id = sanitize_key(case["case_id"])
    candidates = []
    if root is not None:
        candidates.extend([
            root / f"{case_id}.mp4",
            root / case_id / "full.mp4",
            root / case_id / f"{case_id}.mp4",
        ])
    found = first_existing(candidates)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"Full video not found for case_id={case_id}. Provide `video_path`/`full_video_path` in json or --full_video_dir."
    )


def get_case_segment_videos(case: Dict[str, Any], full_video: Path, args, case_key: str, json_root: Path) -> List[Path]:
    temporal = case.get("temporal_captions", []) or []
    seg_root = Path(args.segment_video_dir) if args.segment_video_dir else None
    paths: List[Path] = []

    if isinstance(case.get("segment_video_paths"), list):
        for p in case["segment_video_paths"]:
            rp = resolve_path(p, root=json_root)
            if rp is None or not rp.exists():
                raise FileNotFoundError(f"Segment video not found for case {case['case_id']}: {p}")
            paths.append(rp)
        return paths

    if isinstance(case.get("segments"), list):
        for seg in case["segments"]:
            if isinstance(seg, dict) and seg.get("video_path"):
                rp = resolve_path(seg.get("video_path"), root=json_root)
                if rp is None or not rp.exists():
                    raise FileNotFoundError(f"Segment video not found for case {case['case_id']}: {seg.get('video_path')}")
                paths.append(rp)
        if paths:
            return paths

    if seg_root is not None:
        case_id = sanitize_key(case["case_id"])
        for i, _ in enumerate(temporal):
            candidates = [
                seg_root / case_id / f"seg_{i:03d}.mp4",
                seg_root / case_id / f"segment_{i:03d}.mp4",
                seg_root / f"{case_id}_seg_{i:03d}.mp4",
                seg_root / f"{case_id}_segment_{i:03d}.mp4",
            ]
            found = first_existing(candidates)
            if found is None:
                break
            paths.append(found)
        if len(paths) == len(temporal):
            return paths

    if args.extract_segments_from_full:
        out_dir = Path(args.output_dir) / "_track2_work" / "extracted_segments" / case_key
        for i, seg in enumerate(temporal):
            dst = out_dir / f"seg_{i:03d}.mp4"
            extract_segment_with_ffmpeg(full_video, dst, seg.get("start", 0.0), seg.get("end", 1.0), overwrite=args.overwrite_work)
            paths.append(dst)
        return paths

    raise FileNotFoundError(
        f"Segment videos not found for case_id={case['case_id']}. Provide segment_video_paths / segments[].video_path / --segment_video_dir, "
        "or use --extract_segments_from_full."
    )


def get_subject_pairs(case: Dict[str, Any]) -> List[Tuple[str, str]]:
    img_paths = case.get("img_paths", []) or []
    labels = case.get("class_label", []) or []
    pairs = []
    for i, img_path in enumerate(img_paths):
        label = labels[i] if i < len(labels) else "object"
        pairs.append((str(img_path), str(label)))
    return pairs


class Track2Eval:
    def __init__(self, args):
        self.args = args
        self.json_root = (
            Path(args.path_base_dir).expanduser().resolve()
            if args.path_base_dir
            else Path(args.input_json).expanduser().resolve().parent
        )
        self.output_dir = Path(args.output_dir)
        self.work_dir = self.output_dir / "_track2_work"
        self.raw_dir = self.work_dir / "raw_outputs"
        self.full_video_dir = self.work_dir / "input_videos_full"
        self.segment_video_dir = self.work_dir / "input_videos_segment"
        self.human_ref_dir = self.work_dir / "input_refs_human"
        self.nonhuman_ref_dir = self.work_dir / "input_refs_nonhuman"
        self.full_prompt_json = self.work_dir / "gmescore_full_prompts.json"
        self.segment_prompt_json = self.work_dir / "gmescore_segment_prompts.json"
        self.facesim_json = self.work_dir / "facesim_human_input.json"
        self.nexus_json = self.work_dir / "nexus_nonhuman_input.json"
        self.meta_json = self.work_dir / "case_meta_track2.json"

    def run(self):
        stages = [s.strip() for s in self.args.stages.split(",") if s.strip()]
        if "all" in stages:
            stages = ["prepare", "core", "motion_smoothness", "nexus", "merge"]
        if "prepare" in stages:
            self.prepare()
        if "core" in stages:
            self.run_core_metrics()
        if "motion_smoothness" in stages:
            self.run_motion_smoothness()
        if "nexus" in stages:
            self.run_nexus()
        if "merge" in stages:
            self.merge()

    def subprocess_run(self, cmd: List[str], name: str) -> None:
        print(f"[Track2Eval] Running {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def prepare(self) -> None:
        ensure_dir(self.output_dir)
        ensure_dir(self.full_video_dir)
        ensure_dir(self.segment_video_dir)
        ensure_dir(self.human_ref_dir)
        ensure_dir(self.nonhuman_ref_dir)

        raw_cases = load_json(Path(self.args.input_json))
        cases = normalize_cases(raw_cases)
        input_image_root = Path(self.args.input_image_root) if self.args.input_image_root else None

        full_prompt_map: Dict[str, Any] = {}
        seg_prompt_map: Dict[str, Any] = {}
        facesim_map: Dict[str, Any] = {}
        nexus_map: Dict[str, Any] = {}
        meta: List[Dict[str, Any]] = []

        for idx, case in enumerate(cases):
            if "img_paths" not in case or "class_label" not in case or "global_prompt" not in case:
                raise KeyError(f"Case {case.get('case_id', idx)} must contain img_paths/class_label/global_prompt")
            if "temporal_captions" not in case or not isinstance(case["temporal_captions"], list):
                raise KeyError(f"Case {case.get('case_id', idx)} must contain temporal_captions as a list")

            case_id = sanitize_key(str(case["case_id"]))
            case_key = f"case_{idx:04d}_{case_id}"
            full_video_src = get_case_full_video(case, self.args, self.json_root)
            full_video_dst = self.full_video_dir / f"{case_key}.mp4"
            safe_link_or_copy(full_video_src, full_video_dst, overwrite=self.args.overwrite_work)

            temporal = case.get("temporal_captions", []) or []
            seg_srcs = get_case_segment_videos(case, full_video_src, self.args, case_key, self.json_root)
            if len(seg_srcs) != len(temporal):
                raise ValueError(
                    f"Case {case_id}: num segment videos ({len(seg_srcs)}) != num temporal captions ({len(temporal)})"
                )

            segment_meta = []
            for seg_idx, (seg, seg_src) in enumerate(zip(temporal, seg_srcs)):
                seg_key = f"{case_key}__seg_{seg_idx:03d}"
                seg_dst = self.segment_video_dir / f"{seg_key}.mp4"
                safe_link_or_copy(seg_src, seg_dst, overwrite=self.args.overwrite_work)
                start = safe_float(seg.get("start"))
                end = safe_float(seg.get("end"))
                if start is None:
                    start = 0.0
                if end is None:
                    end = 1.0
                weight = max(0.0, float(end) - float(start))
                desc = str(seg.get("description", ""))
                seg_prompt_map[seg_key] = {"prompt": desc}
                segment_meta.append({
                    "segment_id": seg_idx,
                    "segment_key": seg_key,
                    "start": start,
                    "end": end,
                    "duration_weight": weight,
                    "description": desc,
                    "video_path": str(seg_src),
                    "work_video_path": str(seg_dst),
                })

            full_prompt_map[case_key] = {"prompt": str(case.get("global_prompt", ""))}

            human_img_paths = []
            nonhuman_img_paths = []
            human_labels = []
            nonhuman_labels = []
            for ref_idx, (img_rel, label) in enumerate(get_subject_pairs(case)):
                ref_src = resolve_path(img_rel, input_image_root)
                if ref_src is None or not ref_src.exists():
                    raise FileNotFoundError(
                        f"Reference image not found for case={case_id}, ref={img_rel}. "
                        "Use --input_image_root for relative img_paths."
                    )
                suffix = ref_src.suffix or ".png"
                if is_human_class(label):
                    dst_name = f"{case_key}__human_{ref_idx:02d}{suffix}"
                    dst = self.human_ref_dir / dst_name
                    safe_link_or_copy(ref_src, dst, overwrite=self.args.overwrite_work)
                    human_img_paths.append(dst_name)
                    human_labels.append(label)
                else:
                    dst_name = f"{case_key}__obj_{ref_idx:02d}{suffix}"
                    dst = self.nonhuman_ref_dir / dst_name
                    safe_link_or_copy(ref_src, dst, overwrite=self.args.overwrite_work)
                    nonhuman_img_paths.append(dst_name)
                    nonhuman_labels.append(label)

            if human_img_paths:
                facesim_map[case_key] = {"img_paths": human_img_paths, "class_label": human_labels}
            if nonhuman_img_paths:
                nexus_map[case_key] = {"img_paths": nonhuman_img_paths, "class_label": nonhuman_labels}

            meta.append({
                "index": idx,
                "case_id": str(case["case_id"]),
                "id_index": str(case.get("id_index", case["case_id"])),
                "case_key": case_key,
                "original": case,
                "full_video_path": str(full_video_src),
                "work_full_video_path": str(full_video_dst),
                "has_human_subject": bool(human_img_paths),
                "has_nonhuman_subject": bool(nonhuman_img_paths),
                "human_ref_count": len(human_img_paths),
                "nonhuman_ref_count": len(nonhuman_img_paths),
                "num_refs": len(case.get("img_paths", []) or []),
                "num_segments": len(segment_meta),
                "segments": segment_meta,
            })

        dump_json(full_prompt_map, self.full_prompt_json)
        dump_json(seg_prompt_map, self.segment_prompt_json)
        dump_json(facesim_map, self.facesim_json)
        dump_json(nexus_map, self.nexus_json)
        dump_json(meta, self.meta_json)
        print(json.dumps({
            "meta": str(self.meta_json),
            "full_videos": str(self.full_video_dir),
            "segment_videos": str(self.segment_video_dir),
            "full_gme_prompts": str(self.full_prompt_json),
            "segment_gme_prompts": str(self.segment_prompt_json),
            "facesim_input": str(self.facesim_json),
            "nexus_input": str(self.nexus_json),
        }, indent=2, ensure_ascii=False))

    def run_core_metrics(self) -> None:
        if not self.meta_json.exists():
            raise FileNotFoundError(f"Prepare first: {self.meta_json}")
        self.run_aesthetic()
        self.run_gmescore_full()
        self.run_gmescore_segment()
        self.run_motion_amplitude()
        if not self.args.skip_facesim and json_has_items(self.facesim_json):
            self.run_facesim()
        else:
            ensure_dir(self.raw_dir / "facesim")
            dump_json({}, self.raw_dir / "facesim" / "facesim.json")
        if not self.args.skip_naturalscore:
            self.run_naturalscore()

    def run_aesthetic(self) -> None:
        out_dir = self.raw_dir / "aesthetic"
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(Path(self.args.aesthetic_script)),
            "--input_video_folder", str(self.full_video_dir),
            "--output_json_folder", str(out_dir),
            "--aes_clip_path", self.args.aes_clip_path,
            "--aes_main_path", self.args.aes_main_path,
            "--num_frames", str(self.args.aesthetic_num_frames),
        ]
        self.subprocess_run(cmd, "aesthetic/full")

    def run_gmescore_full(self) -> None:
        out_dir = self.raw_dir / "gmescore_full"
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(Path(self.args.gmescore_script)),
            "--input_video_folder", str(self.full_video_dir),
            "--input_json_file", str(self.full_prompt_json),
            "--output_json_folder", str(out_dir),
            "--model_path", self.args.gme_model_path,
            "--num_frames", str(self.args.gme_num_frames),
        ]
        self.subprocess_run(cmd, "gmescore/full")

    def run_gmescore_segment(self) -> None:
        out_dir = self.raw_dir / "gmescore_segment"
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(Path(self.args.gmescore_script)),
            "--input_video_folder", str(self.segment_video_dir),
            "--input_json_file", str(self.segment_prompt_json),
            "--output_json_folder", str(out_dir),
            "--model_path", self.args.gme_model_path,
            "--num_frames", str(self.args.gme_segment_num_frames),
        ]
        self.subprocess_run(cmd, "gmescore/segment")

    def run_motion_amplitude(self) -> None:
        out_dir = self.raw_dir / "motion_amplitude"
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(Path(self.args.motion_amplitude_script)),
            "--input_video_folder", str(self.full_video_dir),
            "--output_json_folder", str(out_dir),
            "--num_workers", str(self.args.motion_num_workers),
        ]
        self.subprocess_run(cmd, "motion_amplitude/full")

    def run_facesim(self) -> None:
        out_dir = self.raw_dir / "facesim"
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(Path(self.args.facesim_script)),
            "--input_video_folder", str(self.full_video_dir),
            "--input_image_folder", str(self.human_ref_dir),
            "--input_json_file", str(self.facesim_json),
            "--output_json_folder", str(out_dir),
            "--model_path", self.args.facesim_model_path,
            "--num_frames", str(self.args.facesim_num_frames),
        ]
        self.subprocess_run(cmd, "facesim/human/full")

    def run_naturalscore(self) -> None:
        out_dir = self.raw_dir / "naturalscore"
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(Path(self.args.naturalscore_script)),
            "--input_video_folder", str(self.full_video_dir),
            "--output_json_folder", str(out_dir),
            "--model_name", self.args.natural_model_name,
            "--num_workers", str(self.args.natural_num_workers),
        ]
        self.subprocess_run(cmd, "naturalscore/full")

    def run_motion_smoothness(self) -> None:
        out_dir = self.raw_dir / "motion_smoothness"
        ensure_dir(out_dir)
        cmd = [
            sys.executable, str(Path(self.args.motion_smoothness_script)),
            "--input_video_folder", str(self.full_video_dir),
            "--output_json_folder", str(out_dir),
            "--model_path", self.args.motion_smoothness_model_path,
            "--window_size", str(self.args.motion_smoothness_window_size),
        ]
        self.subprocess_run(cmd, "motion_smoothness/full")

    def run_nexus(self) -> None:
        out_dir = self.raw_dir / "nexusscore"
        ensure_dir(out_dir)
        if not json_has_items(self.nexus_json):
            dump_json({}, out_dir / "nexusscore.json")
            print("[Track2Eval] No non-human subject refs; wrote empty nexusscore.json")
            return
        cmd = [
            sys.executable, str(Path(self.args.nexusscore_script)),
            "--input_video_folder", str(self.full_video_dir),
            "--input_image_folder", str(self.nonhuman_ref_dir),
            "--input_json_file", str(self.nexus_json),
            "--output_json_folder", str(out_dir),
            "--model_config", self.args.nexus_model_config,
            "--yolo_model_path", self.args.yolo_model_path,
            "--yolo_clip_model_path", self.args.yolo_clip_model_path,
            "--gme_model_path", self.args.gme_model_path,
            "--num_frames", str(self.args.nexus_num_frames),
            "--overwrite",
        ]
        self.subprocess_run(cmd, "nexusscore/nonhuman/full")

    def merge(self) -> None:
        meta = load_json(self.meta_json)
        aes = load_json_if_exists(self.raw_dir / "aesthetic" / "aesthetic_score.json")
        gme_full = load_json_if_exists(self.raw_dir / "gmescore_full" / "gmescore.json")
        gme_segment = load_json_if_exists(self.raw_dir / "gmescore_segment" / "gmescore.json")
        motion_amp = load_json_if_exists(self.raw_dir / "motion_amplitude" / "motion_amplitude.json")
        facesim = load_json_if_exists(self.raw_dir / "facesim" / "facesim.json")
        nexus = load_json_if_exists(self.raw_dir / "nexusscore" / "nexusscore.json")
        motion_smooth = load_json_if_exists(self.raw_dir / "motion_smoothness" / "motion_smoothness.json")
        naturals = []
        if not self.args.skip_naturalscore:
            for i in [1, 2, 3]:
                p = self.raw_dir / "naturalscore" / f"naturalscore_{i}.json"
                if p.exists():
                    naturals.append(load_json(p))

        merged: List[Dict[str, Any]] = []
        for m in meta:
            key = m["case_key"]
            original = dict(m["original"])
            row = {
                **original,
                "case_id": m["case_id"],
                "id_index": m.get("id_index", m["case_id"]),
                "case_key": key,
                "video_path": m["full_video_path"],
                "work_video_path": m["work_full_video_path"],
                "num_refs": m["num_refs"],
                "num_segments": m["num_segments"],
                "has_human_subject": m["has_human_subject"],
                "has_nonhuman_subject": m["has_nonhuman_subject"],
            }

            update_prefixed(row, aes.get(key, {}), prefix=None)
            update_prefixed(row, motion_amp.get(key, {}), prefix=None)
            update_prefixed(row, motion_smooth.get(key, {}), prefix=None)
            if m["has_human_subject"]:
                update_prefixed(row, facesim.get(key, {}), prefix=None)
            else:
                row["cur_score"] = None
                row["arc_score"] = None
            if m["has_nonhuman_subject"]:
                update_prefixed(row, nexus.get(key, {}), prefix=None)
            else:
                row["nexus_score"] = None

            full_gme_row = gme_full.get(key, {})
            row["gme_score_full"] = safe_float(metric_from_row(full_gme_row, "gme_score"))

            seg_scores = []
            seg_weights = []
            seg_out = []
            for seg in m["segments"]:
                seg_key = seg["segment_key"]
                seg_row = gme_segment.get(seg_key, {})
                score = safe_float(metric_from_row(seg_row, "gme_score"))
                seg_scores.append(score)
                seg_weights.append(float(seg.get("duration_weight", 0.0)))
                seg_item = dict(seg)
                seg_item["gme_score"] = score
                seg_out.append(seg_item)
            row["segment_results"] = seg_out
            row["gme_score_segment_avg"] = weighted_mean(seg_scores, seg_weights)
            # For Track2 final text score, use segment text by default; keep full score separately.
            row["gme_score"] = row["gme_score_segment_avg"]

            natural_vals = []
            for run_idx, natural_map in enumerate(naturals, start=1):
                raw_score = natural_map.get(key)
                score = safe_float(raw_score)
                row[f"naturalscore_{run_idx}"] = score
                natural_vals.append(score)
            if natural_vals:
                row["naturalscore"] = mean_valid(natural_vals)
                row["natural_score"] = row["naturalscore"]

            row["subject_score"] = combine_subject_score(row)
            merged.append(row)

        id_results = aggregate_by_id(merged)
        summary = aggregate_summary(merged, id_results)

        case_out = self.output_dir / "case_results_track2.json"
        by_case_out = self.output_dir / "case_results_track2_by_case.json"
        id_out = self.output_dir / "id_results_track2.json"
        summary_out = self.output_dir / "summary_track2.json"
        official_out = self.output_dir / "official_merge_track2.json"

        dump_json(merged, case_out)
        dump_json({c["case_id"]: c for c in merged}, by_case_out)
        dump_json(id_results, id_out)
        dump_json(summary, summary_out)
        dump_json(summary, official_out)
        print(json.dumps({
            "case_results": str(case_out),
            "case_results_by_case": str(by_case_out),
            "id_results": str(id_out),
            "summary_results": str(summary_out),
            "official_merge": str(official_out),
        }, indent=2, ensure_ascii=False))


def json_has_items(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        obj = load_json(path)
        return isinstance(obj, dict) and len(obj) > 0
    except Exception:
        return False


def load_json_if_exists(path: Path):
    if path.exists():
        return load_json(path)
    return {}


def metric_from_row(row: Any, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return None


def update_prefixed(dst: Dict[str, Any], src: Any, prefix: Optional[str] = None) -> None:
    if not isinstance(src, dict):
        return
    for k, v in src.items():
        dst[f"{prefix}_{k}" if prefix else k] = v


def extract_natural_score(case: Dict[str, Any]) -> Optional[float]:
    val = safe_float(case.get("natural_score"))
    if val is None:
        val = safe_float(case.get("naturalscore"))
    if val is None:
        val = mean_valid([safe_float(case.get(f"naturalscore_{i}")) for i in (1, 2, 3)])
    return val


def combine_subject_score(case: Dict[str, Any]) -> Optional[float]:
    vals = []
    if case.get("has_human_subject"):
        vals.append(safe_float(case.get("cur_score")))
    if case.get("has_nonhuman_subject"):
        vals.append(safe_float(case.get("nexus_score")))
    return mean_valid(vals)


def numeric_metric_names(case_results: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for c in case_results:
        for k, v in c.items():
            if k in CORE_FIELDS or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                names.add(k)
    return sorted(names)


def aggregate_by_id(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for case in case_results:
        grouped[str(case.get("id_index", case.get("case_id", "unknown")))].append(case)
    metric_names = numeric_metric_names(case_results)
    out = {}
    for id_key, cases in sorted(grouped.items(), key=lambda x: str(x[0])):
        row = {
            "id_index": id_key,
            "num_cases": len(cases),
            "case_ids": [c.get("case_id") for c in cases],
        }
        for m in metric_names:
            row[f"{m}_raw_avg"] = mean_valid([safe_float(c.get(m)) for c in cases])
        row.update(process_scores_track2_from_cases(cases))
        out[id_key] = row
    return out


def aggregate_summary(case_results: List[Dict[str, Any]], id_results: Dict[str, Any]) -> Dict[str, Any]:
    metric_names = numeric_metric_names(case_results)
    raw_avg = {f"{m}_raw_avg": mean_valid([safe_float(c.get(m)) for c in case_results]) for m in metric_names}
    official = process_scores_track2_from_cases(case_results)
    return {
        "num_cases": len(case_results),
        "num_valid_cases": sum(1 for c in case_results if "error" not in c),
        "num_ids": len(id_results),
        "track2_weights": {
            "aes_score": 0.15,
            "motion_smoothness": 0.08,
            "motion_amplitude": 0.02,
            "subject_score": 0.25,
            "gme_score_segment_avg": 0.15,
            "gme_score_full": 0.05,
            "natural_score": 0.30,
        },
        "case_average_raw": raw_avg,
        "case_average_norm": official["case_average_norm"],
        "total_score": official["total_score"],
        "metric_counts": official["metric_counts"],
    }


def avg_clipped(cases: List[Dict[str, Any]], key: str, lo: float, hi: float, abs_value: bool = False) -> Tuple[float, int]:
    vals = []
    for c in cases:
        v = safe_float(c.get(key))
        if v is None:
            continue
        if abs_value:
            v = abs(v)
        vals.append(max(min(v, hi), lo))
    if not vals:
        return 0.0, 0
    return float(sum(vals) / len(vals)), len(vals)


def norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def process_scores_track2_from_cases(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Ranges inherit the existing Track1/Human-Domain convention where possible.
    aes, c_aes = avg_clipped(case_results, "aes_score", 4.0, 7.0)
    motion_amp, c_amp = avg_clipped(case_results, "motion_fb", 0.0, 1.0, abs_value=True)
    motion_smooth, c_ms = avg_clipped(case_results, "motion_smoothness", 0.0, 1.0, abs_value=True)
    subject, c_subj = avg_clipped(case_results, "subject_score", 0.0, 1.0)
    gme_seg, c_gme_seg = avg_clipped(case_results, "gme_score_segment_avg", 0.0, 1.0)
    gme_full, c_gme_full = avg_clipped(case_results, "gme_score_full", 0.0, 1.0)

    # NaturalScore follows Track1: GPT raw score is in [1, 5].
    natural_vals = []
    for c in case_results:
        v = extract_natural_score(c)
        if v is not None:
            natural_vals.append(max(min(v, 5.0), 1.0))
    natural = float(sum(natural_vals) / len(natural_vals)) if natural_vals else 0.0
    c_nat = len(natural_vals)

    n = {
        "aes_score_norm": norm(aes, 4.0, 7.0),
        "motion_smoothness_norm": norm(motion_smooth, 0.0, 1.0),
        "motion_amplitude_norm": norm(motion_amp, 0.0, 1.0),
        "subject_score_norm": norm(subject, 0.0, 1.0),
        "gme_score_segment_norm": norm(gme_seg, 0.0, 1.0),
        "gme_score_full_norm": norm(gme_full, 0.0, 1.0),
        "natural_score_norm": norm(natural, 1.0, 5.0),
    }
    total = (
        0.15 * n["aes_score_norm"]
        + 0.08 * n["motion_smoothness_norm"]
        + 0.02 * n["motion_amplitude_norm"]
        + 0.25 * n["subject_score_norm"]
        + 0.15 * n["gme_score_segment_norm"]
        + 0.05 * n["gme_score_full_norm"]
        + 0.30 * n["natural_score_norm"]
    )
    return {
        "case_average_norm": n,
        "total_score": total,
        "metric_counts": {
            "aes_score": c_aes,
            "motion_amplitude": c_amp,
            "motion_smoothness": c_ms,
            "subject_score": c_subj,
            "gme_score_segment_avg": c_gme_seg,
            "gme_score_full": c_gme_full,
            "natural_score": c_nat,
        },
    }


def build_parser():
    p = argparse.ArgumentParser(description="Track2 evaluation wrapper: full-video quality/subject/text + segment-level text.")
    p.add_argument("--input_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--path_base_dir",
        type=str,
        default=None,
        help=(
            "Base directory for relative paths in input_json. "
            "If omitted, relative img_paths/video_path/segment_video_paths are resolved against input_json's parent directory."
        ),
    )
    p.add_argument("--stages", type=str, default="prepare,core,merge", help="Comma stages: prepare,core,motion_smoothness,nexus,merge,all")

    p.add_argument("--input_image_root", type=str, default=None, help="Root for relative img_paths in track2 json.")
    p.add_argument("--full_video_dir", type=str, default=None, help="Root containing <case_id>.mp4 or <case_id>/full.mp4 if video_path is absent.")
    p.add_argument("--segment_video_dir", type=str, default=None, help="Root containing <case_id>/seg_000.mp4 or <case_id>_seg_000.mp4 if segment paths are absent.")
    p.add_argument("--extract_segments_from_full", action="store_true", help="Fallback: cut segments from full video using temporal_captions start/end ratios.")
    p.add_argument("--overwrite_work", action="store_true")

    p.add_argument("--aesthetic_script", type=str, default=str(SCRIPT_DIR / "get_aesscore.py"))
    p.add_argument("--facesim_script", type=str, default=str(SCRIPT_DIR / "get_facesim.py"))
    p.add_argument("--gmescore_script", type=str, default=str(SCRIPT_DIR / "get_gmescore.py"))
    p.add_argument("--motion_amplitude_script", type=str, default=str(SCRIPT_DIR / "get_motion_amplitude.py"))
    p.add_argument("--naturalscore_script", type=str, default=str(SCRIPT_DIR / "get_naturalscore.py"))
    p.add_argument("--motion_smoothness_script", type=str, default=str(SCRIPT_DIR / "get_motion_smoothness.py"))
    p.add_argument("--nexusscore_script", type=str, default=str(SCRIPT_DIR / "get_nexusscore_track2.py"))

    p.add_argument("--ckpt_root", type=str, default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--aes_clip_path", type=str, default='openai/clip-vit-large-patch14')
    p.add_argument("--aes_main_path", type=str, default="")
    p.add_argument("--facesim_model_path", type=str, default="")
    p.add_argument("--gme_model_path", type=str, default='Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')

    p.add_argument("--aesthetic_num_frames", type=int, default=32)
    p.add_argument("--facesim_num_frames", type=int, default=32)
    p.add_argument("--gme_num_frames", type=int, default=32)
    p.add_argument("--gme_segment_num_frames", type=int, default=16)
    p.add_argument("--motion_num_workers", type=int, default=32)

    p.add_argument("--skip_facesim", action="store_true")
    p.add_argument("--skip_naturalscore", action="store_true")
    p.add_argument("--natural_model_name", type=str, default="gpt-5.4")
    p.add_argument("--natural_num_workers", type=int, default=32)

    p.add_argument("--motion_smoothness_model_path", type=str, default="q-future/one-align")
    p.add_argument("--motion_smoothness_window_size", type=int, default=3)

    p.add_argument("--nexus_model_config", type=str, default=str(SCRIPT_DIR / "utils/yoloworld/configs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py"))
    p.add_argument("--yolo_model_path", type=str, default="BestWishYsh/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth")
    p.add_argument("--yolo_clip_model_path", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--nexus_num_frames", type=int, default=32)
    return p


def fill_default_paths(args):
    ckpt_root = Path(args.ckpt_root)
    if not args.aes_main_path:
        args.aes_main_path = str(ckpt_root / "aesthetic-model.pth")
    if not args.facesim_model_path:
        args.facesim_model_path = str(ckpt_root)
    return args


def main():
    args = build_parser().parse_args()
    args = fill_default_paths(args)
    Track2Eval(args).run()


if __name__ == "__main__":
    main()
