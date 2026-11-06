# Chronos: Sequential Identity-Preserving Video Generation by Temporal Anchoring

<p align="center">
  <a href="https://arxiv.org/abs/2611.05893"><img src="https://img.shields.io/badge/arXiv-2611.05893-b31b1b.svg" alt="arXiv"></a>
  <a href="https://wenc-k.github.io/Chronos-page/"><img src="https://img.shields.io/badge/Project-Page-orange.svg" alt="Project page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

Official code of **Chronos** (ACM Multimedia 2026), a training-free framework for identity-preserving video generation.

## Entry points

Run from the repository root; every script supports `--help`.

| Entry point | Function |
| --- | --- |
| `pre_data_engineering/prompt_aug_pipeline.py` | Semantic prompt anchoring: extracts a stable identity summary and rewrites prompts |
| `pre_data_engineering/i2i/generate_i2v_first_frames_openai.py` | Visual frame anchoring: generates the first frame from the reference image |
| `api_i2v/generate.py` | API-based image-to-video generation (Track 1) |
| `api_i2v/build_result_json.py` | Builds the Track 1 evaluation manifest |
| `inference/wan_i2v/run.py track1` | Wan2.2 I2V generation (Track 1) |
| `inference/wan_i2v/run.py track2` | Wan2.2 I2V sequential multi-segment generation (Track 2) |
| `inference/wan_i2v/generate_eval_json.py` | Builds the Track 2 evaluation manifest |
| `inference/wan_flf2v/run.py` | Wan2.2 first/last-frame generation via ComfyUI (Track 2) |
| `inference/wan_flf2v/merge_segments.py` | Merges segments into full videos |
| `inference/environment/setup_environment.py` | Sets up the Wan / ComfyUI environments |
| `evaluation/eval_opens2v/eval_stage1.py`, `eval_stage2.py` | Track 1 evaluation |
| `evaluation/eval_opens2v/eval_track2.py` | Track 2 evaluation |
| `evaluation/id_video/id_video_eval_system.py` | DINO / CLIP identity and temporal metrics |

## Citation

```bibtex
@misc{huang2026chronossequentialidentitypreservingvideo,
  title={Chronos: Sequential Identity-Preserving Video Generation by Temporal Anchoring},
  author={Mengqi Huang and Wenchuan Wang and Zhendong Mao and Yijing Tu and Zhike Liu and Luoni Wang and Ning Tong and Xiaoxiao Sheng and Tianyu Huang and Xuecheng Qi and Yongdong Zhang},
  year={2026},
  eprint={2611.05893},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.05893},
}
```

## License

[Apache License 2.0](LICENSE). Bundled third-party code keeps its own license (see the `LICENSE` files in its directories).
