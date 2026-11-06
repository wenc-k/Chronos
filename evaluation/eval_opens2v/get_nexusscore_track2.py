# Adapted from the benchmark working tree for portable evaluation.
import argparse
import os
import os.path as osp
import json
from typing import Any

import decord
import numpy as np
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS

from utils.gme.gme_model import GmeQwen2VL

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HUMAN_CLASSES = {
    "person", "human", "man", "woman", "boy", "girl", "male", "female",
    "child", "kid", "adult", "elderly man", "elderly woman", "young man", "young woman",
}

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)


def is_human_class(label: Any) -> bool:
    return str(label).strip().lower() in HUMAN_CLASSES


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_video_frames(video_path, num_frames=16):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames <= 0:
        return []
    if num_frames is None or num_frames <= 0:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        frame = vr[int(idx)].asnumpy()
        frames.append(Image.fromarray(frame))
    return frames


def generate_image_embeddings(prompt_image, vision_encoder, vision_processor, projector, device="cuda"):
    prompt_image = prompt_image.convert("RGB")
    inputs = vision_processor(images=[prompt_image], return_tensors="pt", padding=True).to(device)
    image_outputs = vision_encoder(**inputs)
    img_feats = image_outputs.image_embeds.view(1, -1)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    if projector is not None:
        img_feats = projector(img_feats)
    return img_feats


def run_image(
    runner,
    vision_encoder,
    vision_processor,
    padding_token,
    image,
    text,
    prompt_image,
    add_padding,
    max_num_boxes,
    score_thr,
    nms_thr,
    device,
):
    image = image.convert("RGB")
    if prompt_image is not None:
        texts = [["object"], [" "]]
        projector = None
        if hasattr(runner.model, "image_prompt_encoder"):
            projector = runner.model.image_prompt_encoder.projector
        prompt_embeddings = generate_image_embeddings(
            prompt_image,
            vision_encoder=vision_encoder,
            vision_processor=vision_processor,
            projector=projector,
            device=device,
        )
        if add_padding == "padding":
            prompt_embeddings = torch.cat([prompt_embeddings, padding_token], dim=0)
        prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(p=2, dim=-1, keepdim=True)
        runner.model.num_test_classes = prompt_embeddings.shape[0]
        runner.model.setembeddings(prompt_embeddings[None])
    else:
        runner.model.setembeddings(None)
        texts = [[t.strip()] for t in text.split(",")]

    data_info = {"img_id": 0, "img": np.array(image), "texts": texts}
    data_info = runner.pipeline(data_info)
    data_batch = {
        "inputs": data_info["inputs"].unsqueeze(0),
        "data_samples": [data_info["data_samples"]],
    }

    with autocast(enabled=False), torch.no_grad():
        if (prompt_image is not None) and ("texts" in data_batch["data_samples"][0]):
            del data_batch["data_samples"][0]["texts"]
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    return pred_instances.cpu().numpy()


def yoloworld_inference(
    runner,
    vision_encoder,
    vision_processor,
    padding_embed,
    image,
    prompt_image,
    input_text=None,
    add_padding="none",
    max_num_boxes=100,
    score_thr=0.5,
    nms_thr=0.7,
    device="cuda",
):
    return run_image(
        runner,
        vision_encoder,
        vision_processor,
        padding_embed,
        image,
        input_text,
        prompt_image,
        add_padding,
        max_num_boxes,
        score_thr,
        nms_thr,
        device,
    )

def patch_mmengine_skip_log_env():
    """
    Workaround:
    mmengine Runner.from_cfg() calls Runner._log_env(),
    which calls collect_env() -> torch.cuda.get_device_name().
    In opens2v_yoloworld this may segfault during CUDA lazy init.
    This patch only skips environment logging; it does not change model inference.
    """
    try:
        import mmengine.runner.runner as runner_mod

        def _skip_log_env(self, env_cfg):
            try:
                self.logger.info(
                    "Skip mmengine collect_env to avoid torch.cuda lazy-init SIGSEGV."
                )
            except Exception:
                pass

        runner_mod.Runner._log_env = _skip_log_env
    except Exception as e:
        print(f"[WARN] Failed to patch mmengine Runner._log_env: {e}")

def warmup_cuda_before_runner():
    """
    Force CUDA initialization before MMEngine Runner moves YOLOWorld to GPU.
    With CUDA_VISIBLE_DEVICES=4, the visible CUDA index inside this process is cuda:0.
    """
    import os
    import torch

    print(f"[INFO] warmup_cuda_before_runner: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[INFO] warmup_cuda_before_runner: torch={torch.__version__}, torch_cuda={torch.version.cuda}")

    if not torch.cuda.is_available():
        print("[WARN] warmup_cuda_before_runner: torch.cuda.is_available() is False")
        return

    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    print(f"[INFO] warmup_cuda_before_runner OK: device={torch.cuda.get_device_name(0)}")

def load_model_and_config(args, clip_model_path, device):
    cfg = Config.fromfile(args.model_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.model_config))[0])

    cfg.load_from = args.yolo_model_path
    cfg.text_model_name = clip_model_path
    cfg.model.vision_model = clip_model_path
    cfg.model.backbone.text_model.model_name = clip_model_path

    if "runner_type" not in cfg:
        try:
            from mmyolo.utils import register_all_modules
            register_all_modules(init_default_scope=True)
        except Exception as e:
            print(f"[WARN] Failed to register mmyolo modules: {e}")

        patch_mmengine_skip_log_env()
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = "mmdet.LoadImageFromNDArray"
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    yolo_processor = AutoProcessor.from_pretrained(clip_model_path)
    yolo_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_path).to(device)

    yolo_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, use_fast=True)
    yolo_text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_path).to(device)

    texts = yolo_tokenizer(text=[" "], return_tensors="pt", padding=True).to(device)
    text_outputs = yolo_text_model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1])[0].unsqueeze(0)

    return yolo_vision_model, yolo_processor, runner, txt_feats


def compute_case_nexus_score(
    video_path,
    case_json,
    input_image_folder,
    runner,
    yolo_vision_model,
    yolo_processor,
    txt_feats,
    gme,
    args,
    device,
):
    prompt_image_paths = case_json.get("img_paths", []) or []
    image_labels = case_json.get("class_label", []) or []
    pairs = []
    for i, p in enumerate(prompt_image_paths):
        label = image_labels[i] if i < len(image_labels) else "object"
        if is_human_class(label):
            continue
        pairs.append((p, str(label)))

    if not pairs:
        return {
            "nexus_score": None,
            "nexus_num_subject_refs": 0,
            "nexus_num_local_crops": 0,
            "nexus_detected_frame_count": 0,
        }

    frames = sample_video_frames(video_path, num_frames=args.num_frames)
    if not frames:
        return {
            "nexus_score": 0.0,
            "nexus_num_subject_refs": len(pairs),
            "nexus_num_local_crops": 0,
            "nexus_detected_frame_count": 0,
        }

    all_prompt_images = []
    all_image_labels = []
    all_local_images = []
    all_yolo_world_conf = []
    detected_frame_count = 0

    prompt_image_cache = {}
    for img_path, _ in pairs:
        full_path = os.path.join(input_image_folder, img_path)
        prompt_image_cache[img_path] = Image.open(full_path).convert("RGB")

    for frame in frames:
        frame_has_obj = False
        for prompt_image_path, image_label in pairs:
            prompt_image = prompt_image_cache[prompt_image_path]
            pred_instances = yoloworld_inference(
                runner,
                yolo_vision_model,
                yolo_processor,
                txt_feats,
                frame,
                prompt_image,
                score_thr=args.score_thr,
                nms_thr=args.nms_thr,
                max_num_boxes=args.max_num_boxes,
                device=device,
            )
            bboxes = pred_instances["bboxes"]
            confidences = pred_instances["scores"]
            if len(bboxes) != 0:
                frame_has_obj = True
            for bbox, conf in zip(bboxes, confidences):
                x1, y1, x2, y2 = [float(x) for x in bbox]
                if x2 <= x1 or y2 <= y1:
                    continue
                local_image = frame.crop((x1, y1, x2, y2))
                all_local_images.append(local_image)
                all_prompt_images.append(prompt_image)
                all_image_labels.append(image_label)
                all_yolo_world_conf.append(float(conf))
        if frame_has_obj:
            detected_frame_count += 1

    if not all_local_images or detected_frame_count <= 0:
        return {
            "nexus_score": 0.0,
            "nexus_num_subject_refs": len(pairs),
            "nexus_num_local_crops": 0,
            "nexus_detected_frame_count": detected_frame_count,
        }

    e_ref = gme.get_image_embeddings(images=all_prompt_images, is_query=False, show_progress_bar=False)
    e_text = gme.get_text_embeddings(
        texts=all_image_labels,
        instruction="Find an image that matches the given text.",
        show_progress_bar=False,
    )
    e_crop = gme.get_image_embeddings(images=all_local_images, is_query=False, show_progress_bar=False)

    gme_image_score = (e_ref * e_crop).sum(-1)
    gme_text_score = (e_text * e_crop).sum(-1)

    kept_scores = []
    for bbox_conf, text_conf, image_conf in zip(all_yolo_world_conf, gme_text_score, gme_image_score):
        text_conf_f = float(text_conf.detach().float().cpu().item())
        image_conf_f = float(image_conf.detach().float().cpu().item())
        if bbox_conf > args.bbox_conf_thr and text_conf_f > args.text_conf_thr and image_conf_f != 0.0:
            kept_scores.append(image_conf_f)

    if kept_scores:
        nexus_score = float(sum(kept_scores) / len(kept_scores)) / float(detected_frame_count)
    else:
        nexus_score = 0.0

    return {
        "nexus_score": nexus_score,
        "nexus_num_subject_refs": len(pairs),
        "nexus_num_local_crops": len(all_local_images),
        "nexus_detected_frame_count": detected_frame_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Track2 non-human subject consistency via OpenS2V NexusScore.")
    parser.add_argument("--input_video_folder", type=str, required=True)
    parser.add_argument("--input_image_folder", type=str, required=True)
    parser.add_argument("--input_json_file", type=str, required=True)
    parser.add_argument("--output_json_folder", type=str, required=True)
    parser.add_argument(
        "--model_config",
        default="utils/yoloworld/configs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py",
    )
    parser.add_argument(
        "--yolo_model_path",
        default="BestWishYsh/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth",
    )
    parser.add_argument("--yolo_clip_model_path", default="openai/clip-vit-base-patch32")
    parser.add_argument("--gme_model_path", default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")
    parser.add_argument("--work-dir", dest="work_dir", default=None)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--nms_thr", type=float, default=0.7)
    parser.add_argument("--max_num_boxes", type=int, default=100)
    parser.add_argument("--bbox_conf_thr", type=float, default=0.6)
    parser.add_argument("--text_conf_thr", type=float, default=0.30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gme_attn_model", type=str, default="flash_attention_2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_json_folder, exist_ok=True)
    output_json_file = os.path.join(args.output_json_folder, "nexusscore.json")
    if os.path.exists(output_json_file) and not args.overwrite:
        print(f"[get_nexusscore_track2] Existing output found, skip: {output_json_file}")
        return

    json_data = load_json(args.input_json_file)
    if not isinstance(json_data, dict):
        raise TypeError("input_json_file must be a dict keyed by video prefix/case_key")

    yolo_vision_model, yolo_processor, runner, txt_feats = load_model_and_config(
        args, args.yolo_clip_model_path, args.device
    )
    gme = GmeQwen2VL(args.gme_model_path, attn_model=args.gme_attn_model, device=args.device)

    video_files = sorted([f for f in os.listdir(args.input_video_folder) if f.endswith(".mp4")])
    results = {}
    for video_file in tqdm(video_files, desc="Processing videos"):
        prefix = os.path.splitext(video_file)[0]
        if prefix not in json_data:
            continue
        video_path = os.path.join(args.input_video_folder, video_file)
        try:
            results[prefix] = compute_case_nexus_score(
                video_path,
                json_data[prefix],
                args.input_image_folder,
                runner,
                yolo_vision_model,
                yolo_processor,
                txt_feats,
                gme,
                args,
                args.device,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {video_file}: {repr(e)}")
            results[prefix] = {
                "nexus_score": 0.0,
                "nexus_error": repr(e),
            }

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"All results have been saved to {output_json_file}")


if __name__ == "__main__":
    main()
