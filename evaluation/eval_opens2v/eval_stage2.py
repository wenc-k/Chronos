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
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def mean_valid(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def extract_natural_score(case: Dict[str, Any]) -> Optional[float]:
    val = safe_float(case.get("natural_score"))
    if val is None:
        val = safe_float(case.get("naturalscore"))
    if val is None:
        val = mean_valid([safe_float(case.get(f"naturalscore_{i}")) for i in (1, 2, 3)])
    return val


def make_case_key(idx: int, case: Dict[str, Any]) -> str:
    stem = Path(case["video_path"]).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return f"case_{idx:04d}_{stem}"

def process_scores_official_from_cases(case_results: List[Dict[str, Any]]) -> Dict[str, float]:
    total_aes_score = 0.0
    total_motion_amplitude = 0.0
    total_motion_smoothness = 0.0
    total_facesim_cur = 0.0
    total_gme_score = 0.0
    total_natural_score = 0.0

    count_aes_score = 0
    count_motion_amplitude = 0
    count_motion_smoothness = 0
    count_facesim_cur = 0
    count_gme_score = 0
    count_natural_score = 0

    min_aes_score = 4.0
    max_aes_score = 7.0
    min_motion_amplitude = 0.0
    max_motion_amplitude = 1.0
    min_motion_smoothness = 0.0
    max_motion_smoothness = 1.0
    min_facesim_cur = 0.0
    max_facesim_cur = 1.0
    min_gme_score = 0.0
    max_gme_score = 1.0
    min_natural_score = 1.0
    max_natural_score = 5.0

    for case in case_results:
        aes_score = safe_float(case.get("aes_score"))
        if aes_score is not None:
            aes_score = max(min(aes_score, max_aes_score), min_aes_score)
            total_aes_score += aes_score
            count_aes_score += 1

        motion_amplitude = safe_float(case.get("motion_fb"))
        if motion_amplitude is not None:
            motion_amplitude = min(abs(motion_amplitude), max_motion_amplitude)
            total_motion_amplitude += motion_amplitude
            count_motion_amplitude += 1

        motion_smoothness = safe_float(case.get("motion_smoothness"))
        if motion_smoothness is not None:
            motion_smoothness = min(abs(motion_smoothness), max_motion_smoothness)
            total_motion_smoothness += motion_smoothness
            count_motion_smoothness += 1

        facesim_cur = safe_float(case.get("cur_score"))
        if facesim_cur is not None:
            facesim_cur = min(facesim_cur, max_facesim_cur)
            total_facesim_cur += facesim_cur
            count_facesim_cur += 1

        gme_score = safe_float(case.get("gme_score"))
        if gme_score is not None:
            gme_score = min(gme_score, max_gme_score)
            total_gme_score += gme_score
            count_gme_score += 1

        natural_score = extract_natural_score(case)
        if natural_score is not None:
            natural_score = max(min(natural_score, max_natural_score), min_natural_score)
            total_natural_score += natural_score
            count_natural_score += 1

    avg_aes_score = total_aes_score / count_aes_score if count_aes_score else 0.0
    avg_motion_amplitude = total_motion_amplitude / count_motion_amplitude if count_motion_amplitude else 0.0
    avg_motion_smoothness = total_motion_smoothness / count_motion_smoothness if count_motion_smoothness else 0.0
    avg_facesim_cur = total_facesim_cur / count_facesim_cur if count_facesim_cur else 0.0
    avg_gme_score = total_gme_score / count_gme_score if count_gme_score else 0.0
    avg_natural_score = total_natural_score / count_natural_score if count_natural_score else 0.0

    aes_score_norm = (avg_aes_score - min_aes_score) / (max_aes_score - min_aes_score)
    motion_amplitude_norm = (avg_motion_amplitude - min_motion_amplitude) / (max_motion_amplitude - min_motion_amplitude)
    motion_smoothness_norm = (avg_motion_smoothness - min_motion_smoothness) / (max_motion_smoothness - min_motion_smoothness)
    facesim_cur_norm = (avg_facesim_cur - min_facesim_cur) / (max_facesim_cur - min_facesim_cur)
    gme_score_norm = (avg_gme_score - min_gme_score) / (max_gme_score - min_gme_score)
    natural_score_norm = (avg_natural_score - min_natural_score) / (max_natural_score - min_natural_score)

    total_score = (
        0.18 * aes_score_norm
        + 0.09 * motion_smoothness_norm
        + 0.03 * motion_amplitude_norm
        + 0.25 * facesim_cur_norm
        + 0.15 * gme_score_norm
        + 0.30 * natural_score_norm
    )

    return {
        "aes_score": aes_score_norm,
        "motion_smoothness": motion_smoothness_norm,
        "motion_amplitude": motion_amplitude_norm,
        "facesim_cur": facesim_cur_norm,
        "gme_score": gme_score_norm,
        "natural_score": natural_score_norm,
        "total_score": total_score,
    }

def aggregate_by_id(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for case in case_results:
        grouped[str(case.get("id_index", "unknown"))].append(case)

    out = {}
    for id_key, cases in sorted(grouped.items(), key=lambda x: str(x[0])):
        natural_vals = [extract_natural_score(c) for c in cases]

        row = {
            "id_index": id_key,
            "num_cases": len(cases),
            "ref_img_path": sorted({c.get("ref_img_path", "") for c in cases}),
            "aes_score_raw_avg": mean_valid([safe_float(c.get("aes_score")) for c in cases]),
            "motion_smoothness_raw_avg": mean_valid([safe_float(c.get("motion_smoothness")) for c in cases]),
            "motion_fb_raw_avg": mean_valid([safe_float(c.get("motion_fb")) for c in cases]),
            "cur_score_raw_avg": mean_valid([safe_float(c.get("cur_score")) for c in cases]),
            "arc_score_raw_avg": mean_valid([safe_float(c.get("arc_score")) for c in cases]),
            "gme_score_raw_avg": mean_valid([safe_float(c.get("gme_score")) for c in cases]),
            "natural_score_raw_avg": mean_valid(natural_vals),
        }

        official_merge = process_scores_official_from_cases(cases)
        row.update(official_merge)

        out[id_key] = row
    return out


def aggregate_summary(case_results: List[Dict[str, Any]], id_results: Dict[str, Any]) -> Dict[str, Any]:
    natural_vals = [extract_natural_score(c) for c in case_results]

    avg_case = {
        "aes_score_raw_avg": mean_valid([safe_float(c.get("aes_score")) for c in case_results]),
        "motion_smoothness_raw_avg": mean_valid([safe_float(c.get("motion_smoothness")) for c in case_results]),
        "motion_fb_raw_avg": mean_valid([safe_float(c.get("motion_fb")) for c in case_results]),
        "cur_score_raw_avg": mean_valid([safe_float(c.get("cur_score")) for c in case_results]),
        "arc_score_raw_avg": mean_valid([safe_float(c.get("arc_score")) for c in case_results]),
        "gme_score_raw_avg": mean_valid([safe_float(c.get("gme_score")) for c in case_results]),
        "natural_score_raw_avg": mean_valid(natural_vals),
    }

    official_merge = process_scores_official_from_cases(case_results)

    return {
        "num_cases": len(case_results),
        "num_ids": len(id_results),
        "human_domain_weights": {
            "aes_score": 0.18,
            "motion_smoothness": 0.09,
            "motion_amplitude": 0.03,
            "facesim_cur": 0.25,
            "gme_score": 0.15,
            "natural_score": 0.30,
        },
        "case_average_raw": avg_case,
        "case_average_norm": {
            "aes_score_norm": official_merge["aes_score"],
            "motion_smoothness_norm": official_merge["motion_smoothness"],
            "motion_amplitude_norm": official_merge["motion_amplitude"],
            "facesim_cur_norm": official_merge["facesim_cur"],
            "gme_score_norm": official_merge["gme_score"],
            "natural_score_norm": official_merge["natural_score"],
        },
        "total_score": official_merge["total_score"],
    }


class MotionSmoothnessMerger:
    def __init__(self, args):
        self.args = args
        self.work_dir = Path(args.output_dir) / "_merge_ms_work"
        self.video_dir = (
            Path(args.reuse_input_video_dir)
            if args.reuse_input_video_dir is not None
            else (self.work_dir / "input_videos")
        )
        self.ms_out_dir = self.work_dir / "motion_smoothness"

    def prepare_videos(self, input_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        meta = []

        # Reuse existing input_videos: check that the files exist instead of copying again
        if self.args.reuse_input_video_dir is not None:
            self.video_dir = Path(self.args.reuse_input_video_dir)
            if not self.video_dir.exists():
                raise FileNotFoundError(f"reuse_input_video_dir not found: {self.video_dir}")

            for idx, case in enumerate(input_cases):
                for req in ("id_index", "ref_img_path", "video_path", "prompt"):
                    if req not in case:
                        raise KeyError(f"Missing key `{req}` in case index {idx}")

                case_key = make_case_key(idx, case)
                expected_video = self.video_dir / f"{case_key}.mp4"
                if not expected_video.exists():
                    raise FileNotFoundError(f"Missing reused video: {expected_video}")

                meta.append({"index": idx, "case_key": case_key})

            return meta

        # Default: link or copy input_videos again
        ensure_dir(self.video_dir)
        for idx, case in enumerate(input_cases):
            for req in ("id_index", "ref_img_path", "video_path", "prompt"):
                if req not in case:
                    raise KeyError(f"Missing key `{req}` in case index {idx}")

            video_src = (Path(self.args.input_json).resolve().parent / case['video_path'])
            if not video_src.exists():
                raise FileNotFoundError(f"Video not found: {video_src}")

            case_key = make_case_key(idx, case)
            video_dst = self.video_dir / f"{case_key}.mp4"
            safe_link_or_copy(video_src, video_dst)
            meta.append({"index": idx, "case_key": case_key})

        return meta

    def run_motion_smoothness(self) -> Path:
        ensure_dir(self.ms_out_dir)
        cmd = [
            sys.executable,
            str(Path(self.args.motion_smoothness_script)),
            "--input_video_folder", str(self.video_dir),
            "--output_json_folder", str(self.ms_out_dir),
            "--model_path", self.args.motion_smoothness_model_path,
            "--window_size", str(self.args.motion_smoothness_window_size),
        ]
        print(f"[MotionSmoothnessMerger] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return self.ms_out_dir / "motion_smoothness.json"

    def run(self):
        input_cases = load_json(Path(self.args.input_json))
        existing_cases = load_json(Path(self.args.existing_case_json))
        if len(input_cases) != len(existing_cases):
            raise ValueError("input_json and existing_case_json must have the same number of cases")

        meta = self.prepare_videos(input_cases)
        ms_json_path = self.run_motion_smoothness()
        ms_map = load_json(ms_json_path)

        merged_cases = []
        for case, m in zip(existing_cases, meta):
            row = dict(case)
            ms_row = ms_map.get(m["case_key"], {})
            if isinstance(ms_row, dict):
                row.update(ms_row)
            merged_cases.append(row)

        id_results = aggregate_by_id(merged_cases)
        summary = aggregate_summary(merged_cases, id_results)

        case_out = Path(self.args.output_dir) / "case_results_with_qalign_ms.json"
        id_out = Path(self.args.output_dir) / "id_results_with_qalign_ms.json"
        summary_out = Path(self.args.output_dir) / "summary_human_domain_total.json"

        dump_json(merged_cases, case_out)
        dump_json(id_results, id_out)
        dump_json(summary, summary_out)

        print(json.dumps({
            "motion_smoothness_json": str(ms_json_path),
            "case_results": str(case_out),
            "id_results": str(id_out),
            "summary_results": str(summary_out),
        }, indent=2, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(description="Run qalign motion_smoothness then merge with existing metrics using Human-Domain weights.")
    p.add_argument("--input_json", type=str, required=True, help="Original unified input case json.")
    p.add_argument("--existing_case_json", type=str, required=True, help="Existing case_results_v2.json containing other metrics.")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--motion_smoothness_script", type=str, default=str(SCRIPT_DIR / "get_motion_smoothness.py"))
    p.add_argument("--motion_smoothness_model_path", type=str, default="q-future/one-align")
    p.add_argument("--motion_smoothness_window_size", type=int, default=3)

    p.add_argument(
        "--reuse_input_video_dir",
        type=str,
        default=None,
        help="Reuse an existing _v2_work/input_videos directory instead of copying videos again.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    merger = MotionSmoothnessMerger(args)
    merger.run()


if __name__ == "__main__":
    main()
