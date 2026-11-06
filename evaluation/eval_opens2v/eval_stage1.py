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

CORE_CASE_FIELDS = {"id_index", "ref_img_path", "video_path", "prompt", "error"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_ROOT = SCRIPT_DIR.parent / "ckpt"


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


def parse_optional_float(value):
    try:
        return float(value)
    except Exception:
        return None


def mean_valid(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def numeric_metric_names(case_results: List[Dict[str, Any]]) -> List[str]:
    metric_names = set()
    for case in case_results:
        for k, v in case.items():
            if k in CORE_CASE_FIELDS:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metric_names.add(k)
    return sorted(metric_names)


def aggregate_by_id(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for case in case_results:
        grouped[str(case.get("id_index", "unknown"))].append(case)

    metric_names = numeric_metric_names(case_results)
    out = {}
    for id_key, cases in sorted(grouped.items(), key=lambda x: str(x[0])):
        ref_paths = sorted({c.get("ref_img_path", "") for c in cases})
        row = {
            "id_index": id_key,
            "num_cases": len(cases),
            "ref_img_path": ref_paths[0] if len(ref_paths) == 1 else ref_paths,
        }
        for m in metric_names:
            row[m] = mean_valid([c.get(m) for c in cases])
        out[id_key] = row
    return out


def aggregate_all(case_results: List[Dict[str, Any]], id_results: Dict[str, Any]) -> Dict[str, Any]:
    metric_names = numeric_metric_names(case_results)
    return {
        "num_cases": len(case_results),
        "num_valid_cases": sum(1 for c in case_results if "error" not in c),
        "num_ids": len(id_results),
        "case_average": {
            m: mean_valid([c.get(m) for c in case_results])
            for m in metric_names
        },
    }


class EvalSystemV2NewOnly:
    def __init__(self, args):
        self.args = args
        self.work_dir = Path(args.output_dir) / "_v2_work"
        self.raw_dir = self.work_dir / "raw_outputs"
        self.temp_video_dir = self.work_dir / "input_videos"
        self.temp_ref_dir = self.work_dir / "input_refs"
        self.temp_prompt_json = self.work_dir / "gmescore_prompts.json"
        self.temp_facesim_json = self.work_dir / "facesim_input.json"
        self.case_map_json = self.work_dir / "case_key_map.json"

    def run(self):
        ensure_dir(Path(self.args.output_dir))
        cases = load_json(Path(self.args.input_json))
        if not isinstance(cases, list):
            raise ValueError("input_json must be a list of case dicts.")

        case_meta = self.prepare_inputs(cases)
        dump_json(case_meta, self.case_map_json)

        self.run_aesthetic()
        self.run_facesim()
        self.run_gmescore()
        self.run_motion_amplitude()
        if not self.args.skip_naturalscore:
            self.run_naturalscore()

        merged_cases = self.merge_results(cases, case_meta)
        id_results = aggregate_by_id(merged_cases)
        summary_results = aggregate_all(merged_cases, id_results)

        case_out = Path(self.args.output_dir) / "case_results_v2.json"
        id_out = Path(self.args.output_dir) / "id_results_v2.json"
        summary_out = Path(self.args.output_dir) / "summary_results_v2.json"
        dump_json(merged_cases, case_out)
        dump_json(id_results, id_out)
        dump_json(summary_results, summary_out)

        print(json.dumps({
            "case_results": str(case_out),
            "id_results": str(id_out),
            "summary_results": str(summary_out),
        }, indent=2, ensure_ascii=False))

    def make_case_key(self, idx: int, case: Dict[str, Any]) -> str:
        stem = (Path(self.args.input_json).resolve().parent / case['video_path']).stem
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
        return f"case_{idx:04d}_{stem}"

    def prepare_inputs(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ensure_dir(self.temp_video_dir)
        ensure_dir(self.temp_ref_dir)

        prompt_map = {}
        facesim_map = {}
        meta = []

        for idx, case in enumerate(cases):
            for req in ("id_index", "ref_img_path", "video_path", "prompt"):
                if req not in case:
                    raise KeyError(f"Missing key `{req}` in case index {idx}")

            video_src = (Path(self.args.input_json).resolve().parent / case['video_path'])
            ref_src = (Path(self.args.input_json).resolve().parent / case['ref_img_path'])
            if not video_src.exists():
                raise FileNotFoundError(f"Video not found: {video_src}")
            if not ref_src.exists():
                raise FileNotFoundError(f"Reference image not found: {ref_src}")

            case_key = self.make_case_key(idx, case)
            video_dst = self.temp_video_dir / f"{case_key}.mp4"
            ref_suffix = ref_src.suffix if ref_src.suffix else ".png"
            ref_dst = self.temp_ref_dir / f"{case_key}{ref_suffix}"

            safe_link_or_copy(video_src, video_dst)
            safe_link_or_copy(ref_src, ref_dst)

            prompt_map[case_key] = {"prompt": case["prompt"]}
            facesim_map[case_key] = {"img_paths": [f"{case_key}{ref_suffix}"]}

            meta.append({
                "index": idx,
                "case_key": case_key,
                "video_dst": str(video_dst),
                "ref_dst": str(ref_dst),
            })

        dump_json(prompt_map, self.temp_prompt_json)
        dump_json(facesim_map, self.temp_facesim_json)
        return meta

    def subprocess_run(self, cmd: List[str], name: str) -> None:
        print(f"[EvalSystemV2NewOnly] Running {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def run_aesthetic(self) -> None:
        out_dir = self.raw_dir / "aesthetic"
        ensure_dir(out_dir)
        cmd = [
            sys.executable,
            str(Path(self.args.aesthetic_script)),
            "--input_video_folder", str(self.temp_video_dir),
            "--output_json_folder", str(out_dir),
            "--aes_clip_path", self.args.aes_clip_path,
            "--aes_main_path", self.args.aes_main_path,
            "--num_frames", str(self.args.aesthetic_num_frames),
        ]
        self.subprocess_run(cmd, "aesthetic")

    def run_facesim(self) -> None:
        out_dir = self.raw_dir / "facesim"
        ensure_dir(out_dir)
        cmd = [
            sys.executable,
            str(Path(self.args.facesim_script)),
            "--input_video_folder", str(self.temp_video_dir),
            "--input_image_folder", str(self.temp_ref_dir),
            "--input_json_file", str(self.temp_facesim_json),
            "--output_json_folder", str(out_dir),
            "--model_path", self.args.facesim_model_path,
            "--num_frames", str(self.args.facesim_num_frames),
        ]
        self.subprocess_run(cmd, "facesim")

    def run_gmescore(self) -> None:
        out_dir = self.raw_dir / "gmescore"
        ensure_dir(out_dir)
        cmd = [
            sys.executable,
            str(Path(self.args.gmescore_script)),
            "--input_video_folder", str(self.temp_video_dir),
            "--input_json_file", str(self.temp_prompt_json),
            "--output_json_folder", str(out_dir),
            "--model_path", self.args.gme_model_path,
            "--num_frames", str(self.args.gme_num_frames),
        ]
        self.subprocess_run(cmd, "gmescore")

    def run_motion_amplitude(self) -> None:
        out_dir = self.raw_dir / "motion_amplitude"
        ensure_dir(out_dir)
        cmd = [
            sys.executable,
            str(Path(self.args.motion_amplitude_script)),
            "--input_video_folder", str(self.temp_video_dir),
            "--output_json_folder", str(out_dir),
            "--num_workers", str(self.args.motion_num_workers),
        ]
        self.subprocess_run(cmd, "motion_amplitude")

    def run_naturalscore(self) -> None:
        out_dir = self.raw_dir / "naturalscore"
        ensure_dir(out_dir)
        cmd = [
            sys.executable,
            str(Path(self.args.naturalscore_script)),
            "--input_video_folder", str(self.temp_video_dir),
            "--output_json_folder", str(out_dir),
            "--model_name", self.args.natural_model_name,
            "--num_workers", str(self.args.natural_num_workers),
        ]
        self.subprocess_run(cmd, "naturalscore")

    def merge_results(self, original_cases: List[Dict[str, Any]], case_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        aes = load_json(self.raw_dir / "aesthetic" / "aesthetic_score.json")
        facesim = load_json(self.raw_dir / "facesim" / "facesim.json")
        gme = load_json(self.raw_dir / "gmescore" / "gmescore.json")
        motion_amp = load_json(self.raw_dir / "motion_amplitude" / "motion_amplitude.json")

        naturals = []
        if not self.args.skip_naturalscore:
            for i in [1, 2, 3]:
                p = self.raw_dir / "naturalscore" / f"naturalscore_{i}.json"
                if p.exists():
                    naturals.append(load_json(p))

        merged = []
        for case, meta in zip(original_cases, case_meta):
            key = meta["case_key"]
            row = dict(case)

            aes_row = aes.get(key, {})
            if isinstance(aes_row, dict):
                row.update(aes_row)

            face_row = facesim.get(key, {})
            if isinstance(face_row, dict):
                row.update(face_row)

            gme_row = gme.get(key, {})
            if isinstance(gme_row, dict):
                row.update(gme_row)

            motion_row = motion_amp.get(key, {})
            if isinstance(motion_row, dict):
                row.update(motion_row)

            natural_vals = []
            for run_idx, natural_map in enumerate(naturals, start=1):
                raw_score = natural_map.get(key)
                score = parse_optional_float(raw_score)
                row[f"naturalscore_{run_idx}"] = score
                natural_vals.append(score)
            if natural_vals:
                row["naturalscore"] = mean_valid(natural_vals)

            merged.append(row)

        return merged


def build_parser():
    p = argparse.ArgumentParser(description="Evaluation system v2 wrapper for new metrics only.")
    p.add_argument("--input_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--aesthetic_script", type=str, default=str(SCRIPT_DIR / "get_aesscore.py"))
    p.add_argument("--facesim_script", type=str, default=str(SCRIPT_DIR / "get_facesim.py"))
    p.add_argument("--gmescore_script", type=str, default=str(SCRIPT_DIR / "get_gmescore.py"))
    p.add_argument("--motion_amplitude_script", type=str, default=str(SCRIPT_DIR / "get_motion_amplitude.py"))
    p.add_argument("--naturalscore_script", type=str, default=str(SCRIPT_DIR / "get_naturalscore.py"))

    p.add_argument("--ckpt_root", type=str, default=str(DEFAULT_CKPT_ROOT))

    p.add_argument("--aes_clip_path", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--aes_main_path", type=str, default="")
    p.add_argument("--facesim_model_path", type=str, default="")
    p.add_argument("--gme_model_path", type=str, default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")

    p.add_argument("--aesthetic_num_frames", type=int, default=32)
    p.add_argument("--facesim_num_frames", type=int, default=32)
    p.add_argument("--gme_num_frames", type=int, default=32)
    p.add_argument("--motion_num_workers", type=int, default=32)

    p.add_argument("--skip_naturalscore", action="store_true")
    p.add_argument("--natural_model_name", type=str, default="gpt-5.4")
    p.add_argument("--natural_num_workers", type=int, default=1)
    return p


def fill_default_paths(args):
    ckpt_root = Path(args.ckpt_root)
    if not args.aes_main_path:
        args.aes_main_path = str(ckpt_root / "aesthetic-model.pth")
    if not args.facesim_model_path:
        args.facesim_model_path = str(ckpt_root)
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = fill_default_paths(args)
    runner = EvalSystemV2NewOnly(args)
    runner.run()


if __name__ == "__main__":
    main()
