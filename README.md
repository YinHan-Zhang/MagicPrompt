# MagicPrompt: Ultra-Lightweight Prompt Tuning for Video Generation

<p align="center">
  <img src="asset/logo.png" alt="MagicPrompt Logo" width="400">
</p>

## 0. Environment setup

```
conda create -n wan python=3.10 -y
conda activate magicprompt
pip install -r requirements.txt
```

Download the required pretrained models with `bash models/download_models.sh`.

## 1. Data preparation

All datasets are described by a single **JSON metadata file** (`--train_data_meta`), while the actual video / image files live under `--train_data_dir`. Paths in the JSON are **relative to** `--train_data_dir`.

### 1.1 T2V

A flat list of samples. Each item needs `file_path`, `text` and `type`:

```json
[
  {
    "file_path": "./t2v_train_data/celebv_S6iFW-HoFwc_2.mp4",
    "text": "The video features a man with a dark complexion, wearing a red shirt ...",
    "type": "video"
  }
]
```

- `file_path`: path to the target video (or image) **relative to** `--train_data_dir`.
- `text`: the caption / prompt.
- `type`: `"video"` or `"image"`.



### 1.2 Control2V

A flat list. Each item references a **target video** plus a **control video** (e.g. pose / canny), and optionally a **reference image**:

```json
[
  {
    "file_path": "00001/video.mp4",
    "control_file_path": "00001/pose.mp4",
    "text": "A young woman with curly black hair ... is dancing in a room.",
    "type": "video"
  }
]
```

- `file_path`: the ground-truth target video, relative to `--train_data_dir`.
- `control_file_path`: the conditioning video (pose skeleton, canny edge, depth, etc.), relative to `--train_data_dir`.
- `text`: caption of the target video.
- `type`: `"video"`.

---



## 2. Training



### 2.1 Text-to-Video (`t2v`)

```bash
bash scripts/t2v/train.sh
```

Key arguments (see `scripts/t2v/train.sh`):


| Argument                                 | Meaning                                                                                                                                                                            |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--config_path`                          | `config/wan2.2/wan_civitai_t2v.yaml`                                                                                                                                               |
| `--pretrained_model_name_or_path`        | Wan2.2-T2V-A14B base model                                                                                                                                                         |
| `--train_data_dir` / `--train_data_meta` | dataset root and JSON metadata                                                                                                                                                     |
| `--train_mode`                           | `"normal"` (text-to-video)                                                                                                                                                         |
| `--boundary_type`                        | `"low"` (default) trains the **low-noise** expert; `"high"` trains the high-noise expert; `"full"` trains the low-noise expert but also keeps the high-noise expert for validation |
| `--video_sample_n_frames`                | number of frames sampled per clip                                                                                                                                                  |
| `--trainable_modules`                    | which transformer params are trainable, e.g. `soft_k soft_v soft_scale soft_bias text_soft_k text_soft_v`                                                                          |




### 2.2 Control2V

```bash
bash scripts/control2v/train_control.sh
```

Key arguments (see `scripts/control2v/train_control.sh`):


| Argument                                 | Meaning                                        |
| ---------------------------------------- | ---------------------------------------------- |
| `--config_path`                          | `config/wan2.2/wan_civitai_5b.yaml`            |
| `--train_mode`                           | `"control_ref"`                                |
| `--control_ref_image`                    | `"random"` (randomly sample a reference frame) |
| `--add_inpaint_info`                     | inject inpainting mask info into the sample    |
| `--add_full_ref_image_in_self_attention` | feed the reference image into self-attention   |
| `--boundary_type`                        | `"full"` (single 5B model, only one expert)    |




### 2.3 Control2V — reward training (`control2v`)

```bash
bash scripts/control2v/train_reward.sh
```

This variant (`train_reward.py`) adds a reward signal on top of the control objective. Relevant flags:


| Argument                                       | Meaning                                                                                                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--use_mps_reward`                             | enable the MPS (Multi-dimensional Preference Score) reward model                                                                                                                                  |
| `--mps_reward_weight`                          | weight of the MPS reward (default `0.5`)                                                                                                                                                          |
| `--latent_reward_weight`                       | weight of the latent reward (default `0.5`)                                                                                                                                                       |
| `--num_reward_steps`                           | number of **leading** denoising steps used to compute the latent reward; the gradient-tracking denoising loop breaks early after this many steps (default `10`, controlled from the command line) |
| `--num_decoded_latents`                        | number of decoded latents scored by the MPS reward                                                                                                                                                |
| `--backprop`                                   | backpropagate through the denoising steps                                                                                                                                                         |
| `--backprop_strategy` / `--backprop_num_steps` | which steps to backprop through (`tail`, `head`, ...)                                                                                                                                             |
| `--mps_reward_device`                          | device for the frozen MPS model (`cpu` to save VRAM)                                                                                                                                              |


---



## 3. Inference



### 3.1 Text-to-Video

Edit the top-level variables, then run:

```bash
python inference/predict_t2v.py
```

Main knobs:


| Variable                                                         | Meaning                                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `model_name`                                                     | Wan2.2-T2V-A14B directory                                                                   |
| `config_path`                                                    | `config/wan2.2/wan_civitai_t2v.yaml`                                                        |
| `transformer_path`                                               | checkpoint for the **low-noise** expert (e.g. `output_dir_t2v/checkpoint-1000/transformer`) |
| `transformer_high_path`                                          | checkpoint for the **high-noise** expert (`None` = use the original pretrained one)         |
| `prompt` / `negative_prompt`                                     | text inputs                                                                                 |
| `sample_size`, `video_length`, `fps`                             | output resolution / length                                                                  |
| `num_inference_steps`, `guidance_scale`, `shift`, `sampler_name` | generation settings                                                                         |
| `save_path`                                                      | output directory (auto-numbered `00000001.mp4`, ...)                                        |


Checkpoint loading supports a **single file** (`.safetensors` / `.pt`) **or a folder** (sharded safetensors with `model.safetensors.index.json`).

### 3.2 Control2V

```bash
python inference/predict_control2v.py
```

Main knobs:


| Variable                     | Meaning                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `model_name`                 | Wan2.2-Fun-5B-Control directory                                                   |
| `config_path`                | `config/wan2.2/wan_civitai_5b.yaml`                                               |
| `transformer_path`           | trained control transformer checkpoint (or folder)                                |
| `control_video`              | path to the control/conditioning video                                            |
| `ref_image`                  | reference image path (e.g. `<clip>/video_first_frame.jpg`)                        |
| `control_camera_txt`         | (optional) camera-pose text file; if set, pose is used instead of a control video |
| `prompt` / `negative_prompt` | text inputs                                                                       |
| `start_image` / `end_image`  | (optional) first/last frame conditioning                                          |
| `save_path`                  | output directory                                                                  |


