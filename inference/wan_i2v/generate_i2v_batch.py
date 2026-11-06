import _bootstrap
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import copy
import json
import logging
import os
import sys
import warnings
import re
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument('--task', type=str, help='The task to run.', default='i2v-A14B', choices=['i2v-A14B'])
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument('--ckpt_dir', type=str, help='The path to the checkpoint directory.', default=None, required=True)
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )

    # batch i2v for vip200k-style directory:
    # batch_root/id001/{first_frame.png,prompt1_aug.txt,prompt1.txt,image.png}
    parser.add_argument('--batch_root', type=str, help='Root directory containing idxxx subdirectories. If set, run batch i2v generation.', default=None)
    parser.add_argument('--batch_output_dir', type=str, help='Directory to save generated videos in batch mode.', default='outputs/wan_track1')
    parser.add_argument('--batch_json', type=str, help='Path to input json for batch i2v. Each item should contain id_index, ref_img_path, video_path, prompt. If set, use json records instead of reading prompt/image files from batch_root.', default=None)
    parser.add_argument('--batch_output_json', type=str, help='Path to save json metadata in batch mode. Default: batch_output_dir/results.json', default=None)
    parser.add_argument(
        "--batch_image_name",
        type=str,
        default="first_frame.png",
        help="Image filename used as i2v input inside each id directory."
    )
    parser.add_argument(
        "--batch_prompt_name",
        type=str,
        default="prompt1_aug.txt",
        help="Prompt filename used for i2v generation inside each id directory."
    )
    parser.add_argument(
        "--batch_eval_prompt_name",
        type=str,
        default="prompt1.txt",
        help="Original prompt filename written into json for evaluation."
    )
    parser.add_argument(
        "--batch_json_ref_img_name",
        type=str,
        default="image.png",
        help="Reference image filename written to json ref_img_path. Fallback to batch_image_name if missing."
    )
    parser.add_argument(
        "--auto_size_from_image",
        action="store_true",
        default=False,
        help="In batch i2v mode, choose landscape/portrait size according to first_frame image aspect ratio."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip generation if the target video already exists, but still write json metadata."
    )
    parser.add_argument(
        "--batch_shard_by_rank",
        action="store_true",
        default=False,
        help=(
            "Shard batch ids across torchrun ranks/GPU processes in contiguous chunks. "
            "Example with 200 ids and 4 ranks: rank0->id001-id050, rank1->id051-id100, "
            "rank2->id101-id150, rank3->id151-id200. "
            "Use this for multi-GPU throughput; in this mode please keep --ulysses_size 1."
        )
    )
    parser.add_argument(
        "--start_id",
        type=str,
        default=None,
        help=(
            "Start id for local batch filtering, e.g. id001. "
            "Useful for launching one independent python process per GPU."
        )
    )
    parser.add_argument(
        "--end_id",
        type=str,
        default=None,
        help=(
            "End id for local batch filtering, e.g. id050. "
            "Useful for launching one independent python process per GPU."
        )
    )
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _atomic_write_json(records, json_path):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)


def _select_i2v_size_from_image(image_path, default_size, task):
    """
    Wan I2V internally follows the input image aspect ratio. Here we only select
    a landscape/portrait max-area bucket so LTX/other first frames with different
    aspect ratios are mapped to a more suitable Wan size config.
    """
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception as e:
        logging.warning(f"Failed to read image size from {image_path}: {e}. Keep size={default_size}")
        return default_size

    candidates = []
    if w >= h:
        candidates = ["1280*720", "832*480", default_size]
    else:
        candidates = ["720*1280", "480*832", default_size]

    supported = set(SUPPORTED_SIZES.get(task, []))
    for size in candidates:
        if size in SIZE_CONFIGS and size in supported:
            return size
    return default_size


def _split_list_contiguous(items, rank, world_size):
    """Split a list into contiguous chunks and return the chunk for this rank."""
    n = len(items)
    start = (n * rank) // world_size
    end = (n * (rank + 1)) // world_size
    return items[start:end], start, end




def _id_to_int(id_str):
    if id_str is None:
        return None
    s = str(id_str).strip()
    if not s.startswith("id"):
        raise ValueError(f"Invalid id format: {id_str}. Expected like id001")
    try:
        return int(s[2:])
    except ValueError as e:
        raise ValueError(f"Invalid id format: {id_str}. Expected like id001") from e


def _filter_id_dirs_by_range(id_dirs, start_id=None, end_id=None):
    start_num = _id_to_int(start_id)
    end_num = _id_to_int(end_id)
    if start_num is not None and end_num is not None and start_num > end_num:
        raise ValueError(f"start_id {start_id} is larger than end_id {end_id}")

    filtered = []
    for p in id_dirs:
        try:
            cur = _id_to_int(p.name)
        except ValueError:
            continue
        if start_num is not None and cur < start_num:
            continue
        if end_num is not None and cur > end_num:
            continue
        filtered.append(p)
    return filtered


def _setup_generation_runtime_once(args):
    """Initialize distributed runtime once. Used by batch mode to avoid rebuilding it per id."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if args.batch_shard_by_rank and args.ulysses_size > 1:
        raise ValueError(
            "--batch_shard_by_rank means each GPU/rank runs different ids independently, "
            "so --ulysses_size must be 1 in this mode."
        )

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), "sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, \
            "The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, \
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    return rank, world_size, local_rank, device, cfg


def _create_i2v_pipeline_once(args, cfg, device, rank):
    """Create WanI2V only once for the whole batch."""
    logging.info("Creating WanI2V pipeline once for batch generation.")
    return wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )


def _generate_i2v_with_reused_pipeline(args, wan_i2v, cfg, rank):
    """Generate one I2V sample with an already-created WanI2V pipeline."""
    logging.info(f"Input prompt: {args.prompt}")
    img = Image.open(args.image).convert("RGB")
    logging.info(f"Input image: {args.image}")

    logging.info("Generating video with reused WanI2V pipeline ...")
    video = wan_i2v.generate(
        args.prompt,
        img,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)

    if rank == 0 or getattr(args, "batch_shard_by_rank", False):
        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    del video
    torch.cuda.synchronize()

def _record_unique_key(record):
    """
    Key a record by ID and video path so several prompts per ID do not overwrite each other.
    """
    return f"{record.get('id_index', '')}::{record.get('video_path', '')}"


def _record_sort_key(record):
    """
    Sort order:
    id001_prompt1.mp4
    id001_prompt2.mp4
    id002_prompt1.mp4
    ...
    """
    id_index = str(record.get("id_index", "id000"))
    try:
        id_num = _id_to_int(id_index)
    except Exception:
        id_num = 10**9

    video_path = str(record.get("video_path", ""))
    match = re.search(r"_prompt(\d+)", video_path)
    prompt_num = int(match.group(1)) if match else 0

    return id_num, prompt_num, video_path


def _load_batch_json_records(batch_json):
    manifest_path = Path(batch_json).expanduser().resolve()
    with manifest_path.open('r', encoding='utf-8-sig') as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError('Track 1 input must be a JSON list')
    cleaned = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'Record {index} must be an object')
        item = {}
        for field in ('id_index', 'ref_img_path', 'video_path', 'prompt'):
            if field not in record or record[field] is None or not str(record[field]).strip():
                raise ValueError(f'Record {index} needs a nonempty {field}')
            if field != 'id_index' and not isinstance(record[field], str):
                raise ValueError(f'Record {index} {field} must be a string')
            item[field] = str(record[field]).strip()
        for field in ('ref_img_path', 'video_path'):
            path = Path(item[field]).expanduser()
            if not path.is_absolute():
                path = manifest_path.parent / path
            item[field] = str(path.resolve())
        cleaned.append(item)
    return cleaned


def _filter_records_by_range(records, start_id=None, end_id=None):
    start_num = _id_to_int(start_id)
    end_num = _id_to_int(end_id)

    if start_num is not None and end_num is not None and start_num > end_num:
        raise ValueError(f"start_id {start_id} is larger than end_id {end_id}")

    filtered = []
    for record in records:
        cur = _id_to_int(record["id_index"])

        if start_num is not None and cur < start_num:
            continue
        if end_num is not None and cur > end_num:
            continue

        filtered.append(record)

    return filtered


def _merge_json_records_from_paths(json_paths, output_json):
    """
    Merge result records keyed by id_index + video_path, so an ID with
    several prompts keeps one record per video.
    """
    merged = {}

    for json_path in json_paths:
        json_path = Path(json_path)
        if not json_path.exists():
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load shard json {json_path}: {e}")
            continue

        if not isinstance(data, list):
            logging.warning(f"Shard json {json_path} is not a list, skip it.")
            continue

        for record in data:
            if isinstance(record, dict) and "id_index" in record:
                merged[_record_unique_key(record)] = record

    merged_records = sorted(merged.values(), key=_record_sort_key)
    _atomic_write_json(merged_records, output_json)

    return merged_records

def run_batch_i2v_from_json(args):
    assert args.task == "i2v-A14B", "JSON batch mode is for Wan I2V. Please add: --task i2v-A14B"

    if args.use_prompt_extend:
        raise NotImplementedError(
            "JSON batch mode directly uses prompt field in json and does not support --use_prompt_extend."
        )

    rank, world_size, local_rank, device, cfg = _setup_generation_runtime_once(args)

    records_all = _load_batch_json_records(args.batch_json)

    output_dir = Path(args.batch_output_dir).expanduser()
    output_json = (
        Path(args.batch_output_json).expanduser()
        if args.batch_output_json is not None
        else output_dir / "results.json"
    )

    # Optional ID range filter, e.g. --start_id id001 --end_id id050
    if args.start_id is not None or args.end_id is not None:
        if args.batch_shard_by_rank:
            raise ValueError("Please use either --batch_shard_by_rank or --start_id/--end_id, not both.")

        records_all = _filter_records_by_range(records_all, args.start_id, args.end_id)

        if len(records_all) == 0:
            raise RuntimeError(
                f"No records remain after applying range filter: "
                f"start_id={args.start_id}, end_id={args.end_id}"
            )

    # Multi-GPU torchrun: shard cases by rank
    shard_json_path = output_json
    if args.batch_shard_by_rank:
        records_to_run, start_idx, end_idx = _split_list_contiguous(records_all, rank, world_size)
        shard_json_path = output_json.with_name(f"{output_json.stem}.rank{rank}{output_json.suffix}")
    else:
        records_to_run = records_all
        start_idx, end_idx = 0, len(records_all)

        # Independent processes over different ID ranges write separate JSON files
        if args.start_id is not None or args.end_id is not None:
            start_tag = args.start_id if args.start_id is not None else records_to_run[0]["id_index"]
            end_tag = args.end_id if args.end_id is not None else records_to_run[-1]["id_index"]
            shard_json_path = output_json.with_name(
                f"{output_json.stem}.{start_tag}-{end_tag}{output_json.suffix}"
            )

    records_done = []
    if shard_json_path.exists():
        try:
            with open(shard_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                records_done = loaded
        except Exception as e:
            logging.warning(f"Failed to load existing json {shard_json_path}: {e}. Will rewrite it.")
            records_done = []

    if rank == 0:
        logging.info(f"Batch json input: {args.batch_json}")
        logging.info(f"Found {len(records_all)} json records in total")
        logging.info(f"Final merged json target: {output_json}")

    if args.batch_shard_by_rank:
        first_name = records_to_run[0]["id_index"] if len(records_to_run) > 0 else "N/A"
        last_name = records_to_run[-1]["id_index"] if len(records_to_run) > 0 else "N/A"
        logging.info(
            f"[rank {rank} | gpu {local_rank}] assigned {len(records_to_run)} records "
            f"(global slice: [{start_idx}:{end_idx})) | first={first_name}, last={last_name}"
        )
        logging.info(f"[rank {rank}] shard json: {shard_json_path}")
    else:
        if rank == 0:
            logging.info(f"Batch output json: {output_json}")
            if args.start_id is not None or args.end_id is not None:
                first_name = records_to_run[0]["id_index"] if len(records_to_run) > 0 else "N/A"
                last_name = records_to_run[-1]["id_index"] if len(records_to_run) > 0 else "N/A"
                logging.info(
                    f"Explicit id range enabled: start_id={args.start_id}, end_id={args.end_id}, "
                    f"count={len(records_to_run)}, first={first_name}, last={last_name}"
                )
                logging.info(f"Range-shard json: {shard_json_path}")

    wan_i2v = _create_i2v_pipeline_once(args, cfg, device, rank)

    try:
        for item in records_to_run:
            id_index = item["id_index"]
            i2v_image_path = Path(item["ref_img_path"]).expanduser()
            save_file = Path(item["video_path"]).expanduser()
            prompt = item["prompt"]

            missing_files = []
            if not i2v_image_path.exists():
                missing_files.append(str(i2v_image_path))

            if missing_files:
                logging.warning(f"[rank {rank}] Skip {id_index}, missing files: {missing_files}")
                if dist.is_initialized() and not args.batch_shard_by_rank:
                    dist.barrier()
                continue

            save_file.parent.mkdir(parents=True, exist_ok=True)

            run_args = copy.deepcopy(args)
            run_args.prompt = prompt
            run_args.image = str(i2v_image_path)
            run_args.save_file = str(save_file)

            if args.auto_size_from_image:
                run_args.size = _select_i2v_size_from_image(i2v_image_path, args.size, args.task)

            logging.info("=" * 80)
            logging.info(f"[rank {rank}] [{id_index}] image for i2v: {run_args.image}")
            logging.info(f"[rank {rank}] [{id_index}] prompt from json")
            logging.info(f"[rank {rank}] [{id_index}] size config: {run_args.size}")
            logging.info(f"[rank {rank}] [{id_index}] save video: {run_args.save_file}")

            if args.skip_existing and save_file.exists():
                logging.info(f"[rank {rank}] [{id_index}] video exists, skip generation: {save_file}")
            else:
                _generate_i2v_with_reused_pipeline(run_args, wan_i2v, cfg, rank)

                if dist.is_initialized() and not args.batch_shard_by_rank:
                    dist.barrier()

            record = {
                "id_index": id_index,
                "ref_img_path": item["ref_img_path"],
                "video_path": item["video_path"],
                "prompt": prompt,
            }

            # In distributed mode without batch_shard_by_rank, only rank 0 writes the JSON
            if rank == 0 or args.batch_shard_by_rank:
                records_done = [
                    x for x in records_done
                    if _record_unique_key(x) != _record_unique_key(record)
                ]
                records_done.append(record)
                records_done = sorted(records_done, key=_record_sort_key)

                _atomic_write_json(records_done, shard_json_path)
                logging.info(f"[rank {rank}] [{id_index}] json updated: {shard_json_path}")

            if dist.is_initialized() and not args.batch_shard_by_rank:
                dist.barrier()

        if dist.is_initialized() and args.batch_shard_by_rank:
            dist.barrier()

        if args.batch_shard_by_rank:
            if rank == 0:
                shard_jsons = [
                    output_json.with_name(f"{output_json.stem}.rank{r}{output_json.suffix}")
                    for r in range(world_size)
                ]
                merged_records = _merge_json_records_from_paths(shard_jsons, output_json)
                logging.info(
                    f"Merged {len(shard_jsons)} shard json files into final json: "
                    f"{output_json} total records={len(merged_records)}"
                )

            if dist.is_initialized():
                dist.barrier()

    finally:
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    logging.info("Finished JSON batch i2v generation.")

def run_batch_i2v(args):
    assert args.task == "i2v-A14B", "Batch mode is for Wan I2V. Please add: --task i2v-A14B"
    if args.use_prompt_extend:
        raise NotImplementedError(
            "Batch mode uses prompt1_aug.txt directly and does not support --use_prompt_extend."
        )

    rank, world_size, local_rank, device, cfg = _setup_generation_runtime_once(args)

    batch_root = Path(args.batch_root).expanduser().resolve()
    output_dir = Path(args.batch_output_dir).expanduser().resolve()
    output_json = (
        Path(args.batch_output_json).expanduser().resolve()
        if args.batch_output_json is not None
        else output_dir / "results.json"
    )

    id_dirs_all = sorted([p for p in batch_root.iterdir() if p.is_dir() and p.name.startswith("id")])
    if len(id_dirs_all) == 0:
        raise RuntimeError(f"No idxxx directories found under {batch_root}")

    # Optional explicit local range filter, e.g. --start_id id001 --end_id id050.
    # This is useful when you launch one independent python process per GPU with CUDA_VISIBLE_DEVICES.
    if args.start_id is not None or args.end_id is not None:
        if args.batch_shard_by_rank:
            raise ValueError("Please use either --batch_shard_by_rank or --start_id/--end_id, not both.")
        id_dirs_all = _filter_id_dirs_by_range(id_dirs_all, args.start_id, args.end_id)
        if len(id_dirs_all) == 0:
            raise RuntimeError(
                f"No id directories remain after applying range filter: "
                f"start_id={args.start_id}, end_id={args.end_id}"
            )

    shard_json_path = output_json
    if args.batch_shard_by_rank:
        id_dirs, start_idx, end_idx = _split_list_contiguous(id_dirs_all, rank, world_size)
        shard_json_path = output_json.with_name(f"{output_json.stem}.rank{rank}{output_json.suffix}")
    else:
        id_dirs = id_dirs_all
        start_idx, end_idx = 0, len(id_dirs_all)
        if args.start_id is not None or args.end_id is not None:
            start_tag = args.start_id if args.start_id is not None else id_dirs[0].name
            end_tag = args.end_id if args.end_id is not None else id_dirs[-1].name
            shard_json_path = output_json.with_name(
                f"{output_json.stem}.{start_tag}-{end_tag}{output_json.suffix}"
            )

    records = []
    if shard_json_path.exists():
        try:
            with open(shard_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                records = loaded
        except Exception as e:
            logging.warning(f"Failed to load existing json {shard_json_path}: {e}. Will rewrite it.")
            records = []

    if rank == 0:
        logging.info(f"Batch root: {batch_root}")
        logging.info(f"Found {len(id_dirs_all)} id directories in total")
        logging.info(f"Final merged json target: {output_json}")

    if args.batch_shard_by_rank:
        first_name = id_dirs[0].name if len(id_dirs) > 0 else "N/A"
        last_name = id_dirs[-1].name if len(id_dirs) > 0 else "N/A"
        logging.info(
            f"[rank {rank} | gpu {local_rank}] assigned {len(id_dirs)} ids "
            f"(global slice: [{start_idx}:{end_idx})) | first={first_name}, last={last_name}"
        )
        logging.info(f"[rank {rank}] shard json: {shard_json_path}")
    else:
        if rank == 0:
            logging.info(f"Batch json: {output_json}")
            if args.start_id is not None or args.end_id is not None:
                first_name = id_dirs[0].name if len(id_dirs) > 0 else "N/A"
                last_name = id_dirs[-1].name if len(id_dirs) > 0 else "N/A"
                logging.info(
                    f"Explicit id range enabled: start_id={args.start_id}, end_id={args.end_id}, "
                    f"count={len(id_dirs)}, first={first_name}, last={last_name}"
                )
                logging.info(f"Range-shard json: {shard_json_path}")
                logging.info(
                    "Note: when you launch multiple independent python processes with different "
                    "CUDA_VISIBLE_DEVICES, each process writes its own range json to avoid overwrite."
                )
            else:
                logging.info(f"All ranks cooperate on the same sample sequence (ulysses/SP mode).")

    wan_i2v = _create_i2v_pipeline_once(args, cfg, device, rank)

    try:
        for id_dir in id_dirs:
            id_index = id_dir.name
            i2v_image_path = id_dir / args.batch_image_name
            i2v_prompt_path = id_dir / args.batch_prompt_name
            eval_prompt_path = id_dir / args.batch_eval_prompt_name
            json_ref_img_path = id_dir / args.batch_json_ref_img_name
            if not json_ref_img_path.exists():
                json_ref_img_path = i2v_image_path

            required_files = [i2v_image_path, i2v_prompt_path, eval_prompt_path]
            missing_files = [str(x) for x in required_files if not x.exists()]
            if missing_files:
                logging.warning(f"[rank {rank}] Skip {id_index}, missing files: {missing_files}")
                if dist.is_initialized() and not args.batch_shard_by_rank:
                    dist.barrier()
                continue

            video_name = f"{Path(args.batch_prompt_name).stem}__{Path(args.batch_image_name).stem}__wan2_2_i2v.mp4"
            save_file = output_dir / id_index / video_name
            save_file.parent.mkdir(parents=True, exist_ok=True)

            run_args = copy.deepcopy(args)
            run_args.prompt = _read_text_file(i2v_prompt_path)      # i2v generation prompt: prompt1_aug.txt
            eval_prompt = _read_text_file(eval_prompt_path)         # json/evaluation prompt: prompt1.txt
            run_args.image = str(i2v_image_path)                    # i2v reference image: first_frame.png
            run_args.save_file = str(save_file)

            if args.auto_size_from_image:
                run_args.size = _select_i2v_size_from_image(i2v_image_path, args.size, args.task)

            logging.info("=" * 80)
            logging.info(f"[rank {rank}] [{id_index}] image for i2v: {run_args.image}")
            logging.info(f"[rank {rank}] [{id_index}] prompt for i2v: {i2v_prompt_path}")
            logging.info(f"[rank {rank}] [{id_index}] prompt for json/eval: {eval_prompt_path}")
            logging.info(f"[rank {rank}] [{id_index}] size config: {run_args.size}")
            logging.info(f"[rank {rank}] [{id_index}] save video: {run_args.save_file}")

            if args.skip_existing and save_file.exists():
                logging.info(f"[rank {rank}] [{id_index}] video exists, skip generation: {save_file}")
            else:
                _generate_i2v_with_reused_pipeline(run_args, wan_i2v, cfg, rank)
                if dist.is_initialized() and not args.batch_shard_by_rank:
                    dist.barrier()

            # write per-rank json immediately so reruns are resumable
            record = {
                "ref_img_path": str(json_ref_img_path),
                "prompt": eval_prompt,
                "id_index": id_index,
                "video_path": str(save_file),
            }
            records = [x for x in records if x.get("id_index") != id_index]
            records.append(record)
            records = sorted(records, key=lambda x: x.get("id_index", ""))
            _atomic_write_json(records, shard_json_path)
            logging.info(f"[rank {rank}] [{id_index}] json updated: {shard_json_path}")

            if dist.is_initialized() and not args.batch_shard_by_rank:
                dist.barrier()

        if dist.is_initialized() and args.batch_shard_by_rank:
            dist.barrier()

        if args.batch_shard_by_rank:
            if rank == 0:
                shard_jsons = [
                    output_json.with_name(f"{output_json.stem}.rank{r}{output_json.suffix}")
                    for r in range(world_size)
                ]
                merged_records = _merge_json_records_from_paths(shard_jsons, output_json)
                logging.info(
                    f"Merged {len(shard_jsons)} shard json files into final json: "
                    f"{output_json} (total records={len(merged_records)})"
                )
            if dist.is_initialized():
                dist.barrier()

    finally:
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    logging.info("Finished batch i2v generation.")

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "animate" in args.task:
        logging.info("Creating Wan-Animate pipeline.")
        wan_animate = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora
        )

        logging.info(f"Generating video ...")
        video = wan_animate.generate(
            src_root_path=args.src_root_path,
            replace_flag=args.replace_flag,
            refert_num = args.refert_num,
            clip_len=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info("Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()

    if args.batch_json is not None:
        run_batch_i2v_from_json(args)
    elif args.batch_root is not None:
        run_batch_i2v(args)
    else:
        generate(args)