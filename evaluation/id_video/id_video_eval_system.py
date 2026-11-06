import os
import glob
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

import cv2
import clip
import torch
import open_clip
import numpy as np
import torch.nn as nn
from PIL import Image
from easydict import EasyDict as edict
from transformers import AutoImageProcessor, AutoModel
from omegaconf import OmegaConf

from third_party.RAFT.core.raft import RAFT
from third_party.RAFT.core.utils_core.utils import InputPadder as RAFTInputPadder
from third_party.amt.utils.build_utils import build_from_cfg
from third_party.amt.utils.utils import InputPadder as AMTInputPadder
from third_party.amt.utils.utils import img2tensor, tensor2img, check_dim_and_resize

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class DynamicDegree:
    def __init__(self, args, device: str):
        self.args = args
        self.device = device
        self.load_model()

    def load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location='cpu')
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()

    def get_score(self, flo: torch.Tensor) -> float:
        flo = flo[0].permute(1, 2, 0).cpu().numpy()
        u = flo[:, :, 0]
        v = flo[:, :, 1]
        rad = np.sqrt(np.square(u) + np.square(v))

        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = max(1, int(h * w * 0.05))
        max_rad = np.mean(np.abs(np.sort(-rad_flat))[:cut_index])
        return float(max_rad.item())

    def infer(self, video_path: str) -> float:
        with torch.no_grad():
            if video_path.endswith('.mp4'):
                frames = self.get_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.get_frames_from_img_folder(video_path)
            else:
                raise NotImplementedError(f'Unsupported video input: {video_path}')

            static_score: List[float] = []
            for image1, image2 in zip(frames[:-1], frames[1:]):
                padder = RAFTInputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(flow_up)
                static_score.append(max_rad)

            if not static_score:
                return 0.0
            return float(np.mean(static_score))

    def get_frames(self, video_path: str) -> List[torch.Tensor]:
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        interval = max(1, round(fps / 8)) if fps and fps > 0 else 1
        frame_idx = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            if frame_idx % interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            frame_idx += 1
        video.release()
        if not frame_list:
            raise ValueError(f'No frames found in video: {video_path}')
        return frame_list

    def get_frames_from_img_folder(self, img_folder: str) -> List[torch.Tensor]:
        frame_list = []
        imgs = sorted([
            p for p in glob.glob(os.path.join(img_folder, '*'))
            if Path(p).suffix.lower() in IMAGE_EXTS
        ])
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        if not frame_list:
            raise ValueError(f'No image frames found in folder: {img_folder}')
        return frame_list


class FrameProcess:
    def get_frames(self, video_path: str) -> List[np.ndarray]:
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
            else:
                break
        video.release()
        if not frame_list:
            raise ValueError(f'No frames found in video: {video_path}')
        return frame_list

    def get_frames_from_img_folder(self, img_folder: str) -> List[np.ndarray]:
        frame_list = []
        imgs = sorted([
            p for p in glob.glob(os.path.join(img_folder, '*'))
            if Path(p).suffix.lower() in IMAGE_EXTS
        ])
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        if not frame_list:
            raise ValueError(f'No image frames found in folder: {img_folder}')
        return frame_list

    def extract_frame(self, frame_list: Sequence[np.ndarray], start_from: int = 0) -> List[np.ndarray]:
        return [frame_list[i] for i in range(start_from, len(frame_list), 2)]


class MotionSmoothness:
    def __init__(self, config: str, ckpt: str, device: str):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.initialization()
        self.load_model()

    def load_model(self):
        network_cfg = OmegaConf.load(self.config).network
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(self.ckpt, map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def initialization(self):
        if str(self.device).startswith('cuda'):
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(self.device)
        self.fp = FrameProcess()

    def motion_score(self, video_path: str) -> float:
        if video_path.endswith('.mp4'):
            frames = self.fp.get_frames(video_path)
        elif os.path.isdir(video_path):
            frames = self.fp.get_frames_from_img_folder(video_path)
        else:
            raise NotImplementedError(f'Unsupported video input: {video_path}')

        frame_list = self.fp.extract_frame(frames, start_from=0)
        inputs = [img2tensor(frame).to(self.device) for frame in frame_list]
        if len(inputs) <= 1:
            return 0.0

        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        padding = int(16 / scale)
        padder = AMTInputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        outputs = [inputs[0]]
        for _ in range(int(self.niters)):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(out) for out in outputs]
        vfi_score = self.vfi_score(frames, outputs)
        return float((255.0 - vfi_score) / 255.0)

    def vfi_score(self, ori_frames: Sequence[np.ndarray], interpolate_frames: Sequence[np.ndarray]) -> float:
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        if not interpolate:
            return 255.0
        scores = [self.get_diff(ori[i], interpolate[i]) for i in range(len(interpolate))]
        return float(np.mean(np.array(scores)))

    @staticmethod
    def get_diff(img1: np.ndarray, img2: np.ndarray) -> float:
        img = cv2.absdiff(img1, img2)
        return float(np.mean(img))


class IDVideoEvaluator:
    def __init__(
        self,
        dino_model_path: str,
        dd_model_path: Optional[str],
        amt_config_path: Optional[str],
        amt_ckpt_path: Optional[str],
        device: str = 'cuda',
        use_laion400m_e32: bool = True,
        batch_size: int = 32,
        clip_cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.use_laion400m_e32 = use_laion400m_e32
        self.batch_size = batch_size
        self.clip_cache_dir = clip_cache_dir

        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_path)
        self.dino_model = AutoModel.from_pretrained(dino_model_path).to(device)
        self.dino_model.eval()

        self.clip_i_model, self.clip_i_preprocess = self._load_clip_image_model()
        
        self.clip_t_model, self.clip_t_preprocess = clip.load('ViT-B/32', device=device)
        self.clip_t_model.eval()

        self.dynamic_model = DynamicDegree(
            edict({'model': dd_model_path, 'small': False, 'mixed_precision': False, 'alternate_corr': False}),
            device,
        ) if dd_model_path else None

        self.motion_model = MotionSmoothness(amt_config_path, amt_ckpt_path, device) if amt_config_path and amt_ckpt_path else None

        self.ref_dino_cache: Dict[str, torch.Tensor] = {}
        self.ref_clip_i_cache: Dict[str, torch.Tensor] = {}

    def _load_clip_image_model(self):
        if self.use_laion400m_e32:
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32-quickgelu',
                pretrained='laion400m_e32',
                cache_dir=self.clip_cache_dir,
            )
            model = model.to(self.device).eval()
            return model, preprocess

        model, preprocess = clip.load('ViT-B/32', device=self.device)
        model.eval()
        return model, preprocess

    def evaluate_json(
        self,
        input_json_path: str,
        output_dir: str,
        case_output_name: str = 'case_results.json',
        id_output_name: str = 'id_results.json',
        summary_output_name: str = 'summary_results.json',
    ) -> Dict[str, str]:
        ensure_dir(output_dir)
        with open(input_json_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        if not isinstance(cases, list):
            raise ValueError('Input json must be a list of case dicts.')

        case_results = []
        id_to_cases: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for idx, case in enumerate(cases):
            result = dict(case)
            try:
                resolved_case = dict(case)
                for field in ('video_path', 'ref_img_path'):
                    path = Path(case[field]).expanduser()
                    if not path.is_absolute():
                        relative = Path(input_json_path).resolve().parent / path
                        path = relative if relative.exists() or not path.exists() else path
                    resolved_case[field] = str(path)
                metric_result = self.evaluate_case(resolved_case)
                result.update(metric_result)
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {str(exc)}'
            case_results.append(result)
            id_to_cases[str(case.get('id_index', 'unknown'))].append(result)
            if (idx + 1) % 20 == 0:
                print(f'[IDVideoEvaluator] Finished {idx + 1}/{len(cases)} cases')

        id_results = self.aggregate_by_id(id_to_cases)
        summary_results = self.aggregate_all(case_results, id_results)

        case_output_path = os.path.join(output_dir, case_output_name)
        id_output_path = os.path.join(output_dir, id_output_name)
        summary_output_path = os.path.join(output_dir, summary_output_name)

        with open(case_output_path, 'w', encoding='utf-8') as f:
            json.dump(case_results, f, indent=2, ensure_ascii=False)
        with open(id_output_path, 'w', encoding='utf-8') as f:
            json.dump(id_results, f, indent=2, ensure_ascii=False)
        with open(summary_output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)

        return {
            'case_results': case_output_path,
            'id_results': id_output_path,
            'summary_results': summary_output_path,
        }

    def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_case(case)

        ref_img_path = case['ref_img_path']
        video_path = case['video_path']
        prompt = case['prompt']

        ref_dino = self.get_ref_dino_features(ref_img_path)
        ref_clip_i = self.get_ref_clip_i_features(ref_img_path)

        frames = self.read_video_rgb_frames(video_path)
        video_dino = self.extract_video_dino_features_from_frames(frames)
        video_clip_i = self.extract_video_clip_i_features_from_frames(frames)
        video_clip_t = self.extract_video_clip_t_features_from_frames(frames)

        dino_i = self.compute_cosine_matrix_mean(video_dino, ref_dino)
        clip_i = self.compute_cosine_matrix_mean(video_clip_i, ref_clip_i)
        clip_t = self.compute_clip_t_from_video_features(video_clip_t, prompt)
        t_cons = self.compute_temporal_consistency(video_clip_i)
        dynamic_degree = self.dynamic_model.infer(video_path) if self.dynamic_model is not None else None
        motion_smoothness = self.motion_model.motion_score(video_path) if self.motion_model is not None else None
        temporal_flickering = self.temporal_flickering_from_frames(frames)

        id_consistency = self.mean_valid([dino_i, clip_i])
        text_consistency = clip_t
        video_quality = self.mean_valid([t_cons, motion_smoothness, temporal_flickering])

        return {
            'dino_i': dino_i,
            'clip_i': clip_i,
            'clip_t': clip_t,
            't_cons': t_cons,
            'dynamic_degree': dynamic_degree,
            'motion_smoothness': motion_smoothness,
            'temporal_flickering': temporal_flickering,
            'id_consistency': id_consistency,
            'text_consistency': text_consistency,
            'video_quality': video_quality,
        }

    def aggregate_by_id(self, id_to_cases: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        id_results: Dict[str, Any] = {}
        for id_key, cases in sorted(id_to_cases.items(), key=lambda x: str(x[0])):
            metric_names = self.metric_names()
            aggregate = {
                metric: self.mean_valid([case.get(metric) for case in cases])
                for metric in metric_names
            }
            ref_img_paths = sorted({case.get('ref_img_path', '') for case in cases})
            id_results[str(id_key)] = {
                'id_index': self.safe_to_int(id_key),
                'num_cases': len(cases),
                'ref_img_path': ref_img_paths[0] if len(ref_img_paths) == 1 else ref_img_paths,
                **aggregate,
            }
        return id_results

    def aggregate_all(self, case_results: List[Dict[str, Any]], id_results: Dict[str, Any]) -> Dict[str, Any]:
        metric_names = self.metric_names()
        return {
            'num_cases': len(case_results),
            'num_valid_cases': sum(1 for case in case_results if 'error' not in case),
            'num_ids': len(id_results),
            'case_average': {
                metric: self.mean_valid([case.get(metric) for case in case_results])
                for metric in metric_names
            },
        }

    @staticmethod
    def metric_names() -> List[str]:
        return [
            'dino_i',
            'clip_i',
            'clip_t',
            't_cons',
            'dynamic_degree',
            'motion_smoothness',
            'temporal_flickering',
            'id_consistency',
            'text_consistency',
            'video_quality',
        ]

    @staticmethod
    def safe_to_int(value: Any) -> Any:
        try:
            return int(value)
        except Exception:
            return value

    @staticmethod
    def _validate_case(case: Dict[str, Any]) -> None:
        required_keys = ['id_index', 'ref_img_path', 'video_path', 'prompt']
        missing = [k for k in required_keys if k not in case]
        if missing:
            raise KeyError(f'Missing keys in case: {missing}')

    @staticmethod
    def mean_valid(values: Sequence[Optional[float]]) -> Optional[float]:
        valid = [float(v) for v in values if v is not None]
        if not valid:
            return None
        return float(np.mean(valid))

    @staticmethod
    def normalize_features(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @staticmethod
    def compute_cosine_matrix_mean(video_features: torch.Tensor, ref_features: torch.Tensor) -> float:
        video_features = IDVideoEvaluator.normalize_features(video_features)
        ref_features = IDVideoEvaluator.normalize_features(ref_features)
        similarity_matrix = video_features @ ref_features.T
        frame_scores = similarity_matrix.mean(dim=1)
        return float(frame_scores.mean().item())

    @staticmethod
    def compute_temporal_consistency(video_features: torch.Tensor) -> Optional[float]:
        if video_features.shape[0] <= 1:
            return None
        video_features = IDVideoEvaluator.normalize_features(video_features)
        sim = (video_features[:-1] * video_features[1:]).sum(dim=-1)
        return float(sim.mean().item())

    def compute_clip_t_from_video_features(self, video_features: torch.Tensor, prompt: str) -> float:
        text_tokens = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_t_model.encode_text(text_tokens).float()
        text_features = self.normalize_features(text_features)
        video_features = self.normalize_features(video_features)
        similarity = text_features @ video_features.T
        return float(similarity.mean().item())

    def get_ref_dino_features(self, ref_img_path: str) -> torch.Tensor:
        cache_key = os.path.abspath(ref_img_path)
        if cache_key not in self.ref_dino_cache:
            images = self.load_reference_images(ref_img_path)
            self.ref_dino_cache[cache_key] = self.extract_dino_features_from_pil(images)
        return self.ref_dino_cache[cache_key]

    def get_ref_clip_i_features(self, ref_img_path: str) -> torch.Tensor:
        cache_key = os.path.abspath(ref_img_path)
        if cache_key not in self.ref_clip_i_cache:
            images = self.load_reference_images(ref_img_path)
            self.ref_clip_i_cache[cache_key] = self.extract_clip_i_features_from_pil(images)
        return self.ref_clip_i_cache[cache_key]

    def load_reference_images(self, ref_img_path: str) -> List[Image.Image]:
        if os.path.isdir(ref_img_path):
            image_paths = sorted([
                p for p in glob.glob(os.path.join(ref_img_path, '*'))
                if is_image_file(p)
            ])
        elif os.path.isfile(ref_img_path) and is_image_file(ref_img_path):
            image_paths = [ref_img_path]
        else:
            raise FileNotFoundError(f'Invalid reference image path: {ref_img_path}')

        if not image_paths:
            raise ValueError(f'No reference images found under: {ref_img_path}')
        return [Image.open(p).convert('RGB') for p in image_paths]

    def extract_video_dino_features(self, video_path: str) -> torch.Tensor:
        frames = self.read_video_rgb_frames(video_path)
        return self.extract_video_dino_features_from_frames(frames)

    def extract_video_dino_features_from_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        pil_images = [Image.fromarray(frame).convert('RGB') for frame in frames]
        return self.extract_dino_features_from_pil(pil_images)

    def extract_video_clip_i_features(self, video_path: str) -> torch.Tensor:
        frames = self.read_video_rgb_frames(video_path)
        return self.extract_video_clip_i_features_from_frames(frames)

    def extract_video_clip_i_features_from_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        pil_images = [Image.fromarray(frame).convert('RGB') for frame in frames]
        return self.extract_clip_i_features_from_pil(pil_images)

    def extract_video_clip_t_features(self, video_path: str) -> torch.Tensor:
        frames = self.read_video_rgb_frames(video_path)
        return self.extract_video_clip_t_features_from_frames(frames)

    def extract_video_clip_t_features_from_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        features = []
        for start in range(0, len(frames), self.batch_size):
            batch_frames = frames[start:start + self.batch_size]
            batch_tensors = []
            for frame in batch_frames:
                pil_image = Image.fromarray(frame).convert('RGB')
                batch_tensors.append(self.clip_t_preprocess(pil_image))
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                batch_features = self.clip_t_model.encode_image(batch_tensor).float()
            features.append(batch_features)
        return torch.cat(features, dim=0)

    def extract_dino_features_from_pil(self, images: List[Image.Image]) -> torch.Tensor:
        features = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start:start + self.batch_size]
            with torch.no_grad():
                inputs = self.dino_processor(images=batch_images, return_tensors='pt').to(self.device)
                outputs = self.dino_model(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :].float()
            features.append(batch_features)
        return torch.cat(features, dim=0)

    def extract_clip_i_features_from_pil(self, images: List[Image.Image]) -> torch.Tensor:
        features = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start:start + self.batch_size]
            batch_tensors = [self.clip_i_preprocess(img) for img in batch_images]
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                batch_features = self.clip_i_model.encode_image(batch_tensor).float()
            features.append(batch_features)
        return torch.cat(features, dim=0)

    @staticmethod
    def read_video_rgb_frames(video_path: str) -> List[np.ndarray]:
        if os.path.isdir(video_path):
            frame_paths = sorted([
                p for p in glob.glob(os.path.join(video_path, '*'))
                if is_image_file(p)
            ])
            frames = []
            for p in frame_paths:
                frame = cv2.imread(p, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if not frames:
                raise ValueError(f'No frames found in folder: {video_path}')
            return frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f'Unable to open video file: {video_path}')

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f'No frames found in video: {video_path}')
        return frames

    @staticmethod
    def temporal_flickering_single(video_path: str) -> Optional[float]:
        frames = IDVideoEvaluator.read_video_rgb_frames(video_path)
        return IDVideoEvaluator.temporal_flickering_from_frames(frames)

    @staticmethod
    def temporal_flickering_from_frames(frames: List[np.ndarray]) -> Optional[float]:
        if len(frames) <= 1:
            return None
        diffs = []
        for i in range(len(frames) - 1):
            img1 = np.array(frames[i], dtype=np.float32)
            img2 = np.array(frames[i + 1], dtype=np.float32)
            diffs.append(np.mean(cv2.absdiff(img1, img2)))
        return float((255.0 - np.mean(diffs)) / 255.0)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ID-preserving video evaluation system')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input json list.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save three output json files.')
    parser.add_argument('--dino_model_path', type=str, required=True)
    parser.add_argument('--dd_model_path', type=str, default=None)
    parser.add_argument('--amt_config_path', type=str, default=None)
    parser.add_argument('--amt_ckpt_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--disable_laion_clip', action='store_true')
    parser.add_argument('--clip_cache_dir', type=str, default=None)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    evaluator = IDVideoEvaluator(
        dino_model_path=args.dino_model_path,
        dd_model_path=args.dd_model_path,
        amt_config_path=args.amt_config_path,
        amt_ckpt_path=args.amt_ckpt_path,
        device=args.device,
        use_laion400m_e32=not args.disable_laion_clip,
        batch_size=args.batch_size,
        clip_cache_dir=args.clip_cache_dir,
    )
    output_paths = evaluator.evaluate_json(args.input_json, args.output_dir)
    print(json.dumps(output_paths, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
