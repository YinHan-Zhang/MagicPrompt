"""Modified from VideoX-Fun/scripts/wan2.1_fun/train_lora.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import contextmanager
from typing import List, Optional

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from decord import VideoReader
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

import wan.reward.reward_fn as reward_fn
from wan.reward import LatentReward
from wan.models import (AutoencoderKLWan, AutoencoderKLWan3_8, CLIPModel, WanT5EncoderModel,
                               Wan2_2Transformer3DModel, Wan2_2Transformer3DModel_ori)
from wan.pipeline import Wan2_2FunInpaintPipeline
from wan.utils.lora_utils import create_network, merge_lora
from wan.utils.utils import get_image_to_video_latent, save_videos_grid
from wan.data.dataset_image_video import (ImageVideoControlDataset,
                                                 ImageVideoDataset,
                                                 ImageVideoSampler,
                                                 get_random_mask,
                                                 process_pose_file,
                                                 process_pose_params)
from wan.utils.discrete_sampler import DiscreteSampling
from wan.data.bucket_sampler import (ASPECT_RATIO_512,
                                            ASPECT_RATIO_RANDOM_CROP_512,
                                            ASPECT_RATIO_RANDOM_CROP_PROB,
                                            AspectRatioBatchImageVideoSampler,
                                            RandomSampler, get_closest_ratio)

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

@contextmanager
def video_reader(*args, **kwargs):
    """A context manager to solve the memory leak of decord.
    """
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def log_validation(
    vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, network, 
    loss_fn, config, args, accelerator, weight_dtype, global_step, validation_prompts_idx
):
    try:
        logger.info("Running validation... ")

        transformer3d_val = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        # Initialize a new vae if gradient checkpointing or model cpu offload is enabled.
        if args.vae_gradient_checkpointing or args.low_vram:
            vae = AutoencoderKLWan.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
                additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
            )
            vae.eval()
        # Initialize a new image encoder if model cpu offload is enabled.
        if args.low_vram:
            image_encoder_subpath = config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, image_encoder_subpath),
            )
            clip_image_encoder.eval()
        pipeline = Wan2_2FunInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer3d_val,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        pipeline = pipeline.to(dtype=weight_dtype)
        if args.low_vram:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device=accelerator.device)
        lora_state_dict = accelerator.unwrap_model(transformer3d).state_dict()
        pipeline = merge_lora(pipeline, None, 1, accelerator.device, state_dict=lora_state_dict, transformer_only=True)
        
        to_tensor = transforms.ToTensor()
        validation_loss, validation_reward = 0, 0
        for i in range(len(validation_prompts_idx)):
            validation_idx, validation_prompt = validation_prompts_idx[i]
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    temporal_compression_ratio = vae.config.temporal_compression_ratio
                    video_length = 1
                    if args.video_length != 1:
                        video_length += int((args.video_length - 1) // temporal_compression_ratio * temporal_compression_ratio)
                    sample_size = [args.validation_sample_height, args.validation_sample_width]
                    input_video, input_video_mask, clip_image = get_image_to_video_latent(
                        None, None, video_length=video_length, sample_size=sample_size
                    )

                    if args.seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                    sample = pipeline(
                        validation_prompt,
                        num_frames = video_length,
                        negative_prompt = "bad detailed",
                        height = args.validation_sample_height,
                        width = args.validation_sample_width,
                        guidance_scale = 7,
                        generator = generator,
                        video = input_video,
                        mask_video = input_video_mask,
                        clip_image = clip_image,
                    ).videos
                    sample_saved_name = f"validation_sample/sample-{global_step}-{validation_idx}.mp4"
                    sample_saved_path = os.path.join(args.output_dir, sample_saved_name)
                    save_videos_grid(sample, sample_saved_path, fps=16)

                    num_sampled_frames = 4
                    sampled_frames_list = []
                    with video_reader(sample_saved_path) as vr:
                        sampled_frame_idx_list = np.linspace(0, len(vr), num_sampled_frames, endpoint=False, dtype=int)
                        sampled_frame_list = vr.get_batch(sampled_frame_idx_list).asnumpy()
                        sampled_frames = torch.stack([to_tensor(frame) for frame in sampled_frame_list], dim=0)
                        sampled_frames_list.append(sampled_frames)
                    
                    sampled_frames = torch.stack(sampled_frames_list)
                    sampled_frames = rearrange(sampled_frames, "b t c h w -> b c t h w")
                    loss, reward = loss_fn(sampled_frames, [validation_prompt])
                    validation_loss, validation_reward = validation_loss + loss, validation_reward + reward
        
        validation_loss = validation_loss / len(validation_prompts_idx)
        validation_reward = validation_reward / len(validation_prompts_idx)

        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return validation_loss, validation_reward
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        return None, None


def load_prompts(prompt_path, prompt_column="prompt", start_idx=None, end_idx=None):
    prompt_list = []
    if prompt_path.endswith(".txt"):
        with open(prompt_path, "r") as f:
            for line in f:
                prompt_list.append(line.strip())
    elif prompt_path.endswith(".jsonl"):
        with open(prompt_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                prompt_list.append(item[prompt_column])
    else:
        raise ValueError("The prompt_path must end with .txt or .jsonl.")
    prompt_list = prompt_list[start_idx:end_idx]

    return prompt_list


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

def encode_prompt(
    tokenizer,
    text_encoder,
    prompt,
    negative_prompt,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            Whether to use classifier free guidance or not.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        device: (`torch.device`, *optional*):
            torch device
        dtype: (`torch.dtype`, *optional*):
            torch dtype
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, negative_prompt_embeds


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def _extract_latent_reward(item):
    """Wan2.2 transformers may return the per-block hidden state either as a
    dict keyed by the block index (e.g. ``{"29": tensor}``) or directly as a
    tensor. This helper unifies both formats so the reward computation below
    works regardless of which Wan2_2 model variant is used.
    """
    if isinstance(item, dict):
        return item.get("29", next(iter(item.values())))
    return item


# Modified from EasyAnimateInpaintPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    import inspect
    
    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--validation_prompt_path",
        type=str,
        default=None,
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=1,
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_sample_height",
        type=int,
        default=512,
        help="The height of sampling videos in validation.",
    )
    parser.add_argument(
        "--validation_sample_width",
        type=int,
        default=512,
        help="The width of sampling videos in validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing (for DiT) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--vae_gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing (for VAE) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="normal",
        help="The path to the training prompt file.",
    )
    parser.add_argument(
        '--train_sample_height', 
        type=int,
        default=384,
        help='The height of sampling videos in training'
    )
    parser.add_argument(
        '--train_sample_width', 
        type=int,
        default=672,
        help='The width of sampling videos in training'
    )
    parser.add_argument(
        "--video_length", 
        type=int,
        default=49,
        help="The number of frames to generate in training and validation."
    )
    parser.add_argument(
        '--eta', 
        type=float,
        default=0.0,
        help='eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, '
        'with 0.0 being fully deterministic and 1.0 being equivalent to the DDPM sampler.'
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float,
        default=6.0,
        help="The classifier-free diffusion guidance."
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int,
        default=10,
        help="The number of denoising steps in training and validation."
    )
    parser.add_argument(
        "--num_decoded_latents",
        type=int,
        default=3,
        help="The number of latents to be decoded."
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=None,
        help="The number of sampled frames for the reward function."
    )
    parser.add_argument(
        "--reward_fn", 
        type=str,
        default="aesthetic_loss_fn",
        help='The reward function.'
    )
    parser.add_argument(
        "--reward_fn_kwargs",
        type=str,
        default=None,
        help='The keyword arguments of the reward function.'
    )
    parser.add_argument(
        "--use_mps_reward",
        action="store_true",
        default=False,
        help="Enable the MPS (Multi-dimensional Preference Score) reward model and combine it "
             "with the latent reward via weighted sum. When disabled, training is unchanged.",
    )
    parser.add_argument(
        "--mps_reward_weight",
        type=float,
        default=0.5,
        help="Weight of the MPS reward in the combined loss/reward (used together with --latent_reward_weight).",
    )
    parser.add_argument(
        "--latent_reward_weight",
        type=float,
        default=0.5,
        help="Weight of the latent reward in the combined loss/reward (used together with --mps_reward_weight).",
    )
    parser.add_argument(
        "--mps_num_sampled_frames",
        type=int,
        default=4,
        help="Number of frames sampled from the decoded video for the MPS reward (None = use all frames).",
    )
    parser.add_argument(
        "--mps_reward_device",
        type=str,
        default="cpu",
        help="Device for the (frozen) MPS reward model: 'cuda' or 'cpu'. Use 'cpu' to save VRAM "
             "(gradient still flows back through the frozen model via autograd).",
    )
    parser.add_argument(
        "--num_reward_steps",
        type=int,
        default=10,
        help="Number of (leading) denoising steps used to compute the latent reward. The "
             "expensive gradient-tracking denoising loops break early after this many steps.",
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        default=False,
        help="Whether to use the reward backprop training mode.",
    )
    parser.add_argument(
        "--backprop_step_list",
        nargs="+",
        type=int,
        default=None,
        help="The preset step list for reward backprop. If provided, overrides `backprop_strategy`."
    )
    parser.add_argument(
        "--backprop_strategy",
        choices=["last", "tail", "uniform", "random"],
        default="last",
        help="The strategy for reward backprop."
    )
    parser.add_argument(
        "--stop_latent_model_input_gradient",
        action="store_true",
        default=False,
        help="Whether to stop the gradient of the latents during reward backprop.",
    )
    parser.add_argument(
        "--backprop_random_start_step",
        type=int,
        default=0,
        help="The random start step for reward backprop. Only used when `backprop_strategy` is random."
    )
    parser.add_argument(
        "--backprop_random_end_step",
        type=int,
        default=50,
        help="The random end step for reward backprop. Only used when `backprop_strategy` is random."
    )
    parser.add_argument(
        "--backprop_num_steps",
        type=int,
        default=5,
        help="The number of steps for backprop. Only used when `backprop_strategy` is tail/uniform/random."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="control",
        help=(
            'The format of training data. Support `"control"`'
            ' (default), `"control_ref"`, `"control_camera_ref"`.'
        ),
    )
    parser.add_argument(
        "--control_ref_image",
        type=str,
        default="first_frame",
        help=(
            'The format of training data. Support `"first_frame"`'
            ' (default), `"random"`.'
        ),
    )
    parser.add_argument(
        "--add_full_ref_image_in_self_attention",
        action="store_true",
        help=(
            'Whether enable add full ref image in self attention.'
        ),
    )
    parser.add_argument(
        "--add_inpaint_info",
        action="store_true",
        help="Whether to add inpaint info to the dataset.",
    )
    parser.add_argument(
        "--boundary_type",
        type=str,
        default="full",
        choices=["full", "high", "low"],
        help=(
            'The boundary type for Wan2.2. "full" uses a single 5B transformer; '
            '"high" / "low" train the high / low-noise sub-model of the 14B two-expert model.'
        ),
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=1,
        help="Repeat the video data.",
    )
    parser.add_argument(
        "--vae_mini_batch",
        type=int,
        default=1,
        help="Mini batch size for VAE encoding.",
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    # Boundary (in [0, 1]) that splits the high-noise and low-noise experts of Wan2.2.
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Sanity check for validation
    do_validation = (args.validation_prompt_path is not None or args.validation_prompts is not None)
    if do_validation:
        if not (os.path.exists(args.validation_prompt_path) or args.validation_prompt_path.endswith(".txt")):
            raise ValueError("The `--validation_prompt_path` must be a txt file containing prompts.")
        if args.validation_batch_size < accelerator.num_processes or args.validation_batch_size % accelerator.num_processes != 0:
            raise ValueError("The `--validation_batch_size` must be divisible by the number of processes.")
    
    # Sanity check for validation
    if args.backprop:
        if args.backprop_step_list is not None:
            logger.warning(
                f"The backprop_strategy {args.backprop_strategy} will be ignored "
                f"when using backprop_step_list {args.backprop_step_list}."
            )
            assert any(step <= args.num_inference_steps - 1 for step in args.backprop_step_list)
        else:
            if args.backprop_strategy in set(["tail", "uniform", "random"]):
                assert args.backprop_num_steps <= args.num_inference_steps - 1
            if args.backprop_strategy == "random":
                assert args.backprop_random_start_step <= args.backprop_random_end_step
                assert args.backprop_random_end_step <= args.num_inference_steps - 1

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder.eval()
        # Get Vae
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")


    # Get Transformer
    # Wan2.2-5B uses `boundary_type="full"` (a single transformer). The 14B two-expert model uses
    # a high-noise and a low-noise transformer that are combined at `boundary`; in that case we load
    # both and only the one selected by `boundary_type` is trainable.
    if args.boundary_type != "full":
        sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
        sub_path_2 = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
        low_transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, sub_path),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        high_transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, sub_path_2),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        low_transformer3d.requires_grad_(False)
        high_transformer3d.requires_grad_(False)
        # `transformer3d` always points to the trainable sub-model.
        transformer3d = low_transformer3d if args.boundary_type == "low" else high_transformer3d
    else:
        sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
        transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, sub_path),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
    transformer3d.train()

    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    )
    clip_image_encoder.eval()

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)

    transformer3d_ori = Wan2_2Transformer3DModel_ori.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, sub_path),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    )
    transformer3d_ori.to(accelerator.device, dtype=weight_dtype)
    transformer3d_ori.requires_grad_(False)
    # Load transformer and vae from path if it needs.
    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        # m, u = transformer3d.load_state_dict(state_dict, strict=False)
        m, u = transformer3d_ori.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                
        accelerator.register_save_state_pre_hook(save_model_hook)
        # Save the model weights directly before save_state instead of using a hook.
        # accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        transformer3d_ori.enable_gradient_checkpointing()
    
    if args.vae_gradient_checkpointing:
        # Since 3D casual VAE need a cache to decode all latents autoregressively, .Thus, gradient checkpointing can only be 
        # enabled when decoding the first batch (i.e. the first three) of latents, in which case the cache is not being used.
        
        # num_decoded_latents > 3 is support in EasyAnimate now.
        # if args.num_decoded_latents > 3:
        #     raise ValueError("The vae_gradient_checkpointing is not supported for num_decoded_latents > 3.")
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )

    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in transformer3d.named_parameters():
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                in_already.append(name)
                trainable_params_optim[0]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate}")
                break
    
    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))

    # Init optimizer
    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # MPS reward model (optional, combined with the latent reward).
    # MPS is frozen (requires_grad_(False)) and runs on sampled frames, so it only
    # contributes a scalar reward/loss; the gradient signal still comes from the latent reward.
    mps_reward_fn = None
    if args.use_mps_reward:
        if accelerator.is_main_process:
            # Build on the main process first to avoid concurrent weight downloads.
            _ = reward_fn.MPSReward(device="cpu", dtype=weight_dtype)
        accelerator.wait_for_everyone()
        mps_device = accelerator.device if args.mps_reward_device == "cuda" else args.mps_reward_device
        mps_reward_fn = reward_fn.MPSReward(device=mps_device, dtype=weight_dtype)
        mps_reward_fn.model.eval()
        accelerator.print(f"MPS reward model loaded on {mps_device}.")

    latent_reward_fn = LatentReward(
        metric = "mse", # "cosine"
    )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # Get the dataset
    train_dataset = ImageVideoControlDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, 
        enable_inpaint=args.add_inpaint_info,
        enable_camera_info=args.train_mode == "control_camera_ref"
    )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    
    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]
                
                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }

                return length_to_frame_num

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
                        remaining_sum = 1.0 - first_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [first_element] + [other_elements_value] * (length - 1)
                        return special_list
                        
                if sample_size >= 1536:
                    number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
                elif sample_size >= 1024:
                    number_list = [1, 1.25, 1.5, 2] + image_ratio
                elif sample_size >= 768:
                    number_list = [1, 1.25, 1.5] + image_ratio
                elif sample_size >= 512:
                    number_list = [1] + image_ratio
                else:
                    number_list = [1]

                if all_choices:
                    return number_list

                number_list_prob = np.array(_create_special_list(len(number_list)))
                if rng is None:
                    return np.random.choice(number_list, p = number_list_prob)
                else:
                    return rng.choice(number_list, p = number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            # Used in Control Mode
            new_examples["control_pixel_values"] = []
            # Used in Control Ref Mode
            if args.train_mode != "control":
                new_examples["ref_pixel_values"] = []
                new_examples["clip_pixel_values"] = []
                new_examples["clip_idx"] = []
            # Used in Inpaint mode
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
            # Used in Control Camera Ref Mode
            if args.train_mode == "control_camera_ref":
                new_examples["control_camera_values"] = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

                aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))

                        def get_random_downsample_probability(choice_list, token_sample_size):
                            length = len(choice_list)
                            if length == 1:
                                return [1.0]  # If there's only one element, it gets all the probability
                            
                            # Find the index of the closest value to token_sample_size
                            closest_index = min(range(length), key=lambda i: abs(choice_list[i] - token_sample_size))
                            
                            # Assign 50% to the closest index
                            first_element = 0.50
                            remaining_sum = 1.0 - first_element
                            
                            # Distribute the remaining 50% evenly among the other elements
                            other_elements_value = remaining_sum / (length - 1) if length > 1 else 0.0
                            
                            # Construct the probability distribution
                            probability_list = [other_elements_value] * length
                            probability_list[closest_index] = first_element
                            
                            return probability_list

                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        probabilities = get_random_downsample_probability(choice_list, args.token_sample_size)
                        local_video_sample_size = np.random.choice(choice_list, p=probabilities)

                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            if args.fix_sample_size is not None:
                fix_sample_size = [int(x / 16) * 16 for x in args.fix_sample_size]
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / 16) * 16 for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                closest_size = [int(x / 16) * 16 for x in closest_size]

            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            batch_video_length = int(min(batch_video_length, min_example_length))

            # Magvae needs the number of frames to be 4n + 1.
            batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

            if batch_video_length <= 0:
                batch_video_length = 1

            for example in examples:
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                control_pixel_values = torch.from_numpy(example["control_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                control_pixel_values = control_pixel_values / 255.

                if args.fix_sample_size is not None:
                    # Get adapt hw for resize
                    fix_sample_size = list(map(lambda x: int(x), fix_sample_size))
                    transform = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])

                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                    ])
                elif args.random_ratio_crop:
                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
    
                    transform_no_normalize = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                    ])
                else:
                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
    
                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                    ])
    
                new_examples["pixel_values"].append(transform(pixel_values)[:batch_video_length])
                new_examples["control_pixel_values"].append(transform(control_pixel_values))
            
                if args.train_mode == "control_camera_ref":
                    control_camera_values = example.get("control_camera_values", None)
                    if control_camera_values is None:
                        control_camera_values_size = (
                            new_examples["control_pixel_values"][-1].size()[0], 
                            6, 
                            new_examples["control_pixel_values"][-1].size()[2], 
                            new_examples["control_pixel_values"][-1].size()[3]
                        )
                        local_control_camera_values = torch.zeros(control_camera_values_size)
                        new_examples["control_camera_values"].append(local_control_camera_values)
                    else:
                        local_control_camera_values = process_pose_params(example["control_camera_values"], height=resize_size[0], width=resize_size[1]).permute(0, 3, 1, 2).contiguous()
                        new_examples["control_camera_values"].append(transform_no_normalize(local_control_camera_values))
                
                new_examples["text"].append(example["text"])

                if args.train_mode != "control":
                    if args.control_ref_image == "first_frame":
                        clip_index = 0
                    else:
                        def _create_special_list(length):
                            if length == 1:
                                return [1.0]
                            if length >= 2:
                                first_element = 0.40
                                remaining_sum = 1.0 - first_element
                                other_elements_value = remaining_sum / (length - 1)
                                special_list = [first_element] + [other_elements_value] * (length - 1)
                                return special_list
                        number_list_prob = np.array(_create_special_list(len(new_examples["pixel_values"][-1])))
                        clip_index = np.random.choice(list(range(len(new_examples["pixel_values"][-1]))), p = number_list_prob)
                    new_examples["clip_idx"].append(clip_index)

                    ref_pixel_values = new_examples["pixel_values"][-1][clip_index].unsqueeze(0)
                    new_examples["ref_pixel_values"].append(ref_pixel_values)

                    clip_pixel_values = new_examples["pixel_values"][-1][clip_index].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    new_examples["clip_pixel_values"].append(clip_pixel_values)

                    if args.add_inpaint_info:
                        mask = get_random_mask(new_examples["pixel_values"][-1].size())
                        mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask)
                        # Wan 2.1 use 0 for masked pixels
                        # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                        new_examples["mask_pixel_values"].append(mask_pixel_values)
                        new_examples["mask"].append(mask)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
            new_examples["control_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_pixel_values"]])
            if args.train_mode != "control":
                new_examples["ref_pixel_values"] = torch.stack([example for example in new_examples["ref_pixel_values"]])
                new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])
                new_examples["clip_idx"] = torch.tensor(new_examples["clip_idx"])
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = torch.stack([example for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example for example in new_examples["mask"]])
            if args.train_mode == "control_camera_ref":
                new_examples["control_camera_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_camera_values"]])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(transformer3d, optimizer, train_dataloader, lr_scheduler)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    if args.boundary_type != "full":
        # Keep the non-trainable expert on device too (used during denoising).
        if args.boundary_type == "low":
            high_transformer3d.to(accelerator.device, dtype=weight_dtype)
        else:
            low_transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)
    clip_image_encoder.to(accelerator.device)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            from safetensors.torch import load_file, safe_open
            state_dict = load_file(os.path.join(os.path.join(args.output_dir, path), "model.safetensors"))
            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_reward = 0.0

        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            
            pixel_values = batch["pixel_values"].to(weight_dtype)
            control_pixel_values = batch["control_pixel_values"].to(weight_dtype)

            ref_pixel_values = batch["ref_pixel_values"].to(weight_dtype)
            clip_pixel_values = batch["clip_pixel_values"]
            clip_idx = batch["clip_idx"]

            if args.add_inpaint_info:
                mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                mask = batch["mask"].to(weight_dtype)

            train_prompt = batch['text']
            logger.info(f"train_prompt: {train_prompt}")

            # default height and width
            height = int(args.train_sample_height // 16 * 16)
            width = int(args.train_sample_width // 16 * 16)

            do_classifier_free_guidance = args.guidance_scale > 1.0
            
            # Reduce the vram by offload text encoders
            if args.low_vram:
                torch.cuda.empty_cache()
                text_encoder.to(accelerator.device)

            # Encode input prompt
            (
                prompt_embeds,
                negative_prompt_embeds
            ) = encode_prompt(
                tokenizer,
                text_encoder,
                train_prompt,
                negative_prompt=[""] * len(train_prompt),
                device=accelerator.device,
                dtype=weight_dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            uncond_prompt_embeds = prompt_embeds

            if do_classifier_free_guidance:
                prompt_embeds = negative_prompt_embeds + prompt_embeds
            # `cfg_context` is the CFG list [uncond(neg), cond(text)] used by the teacher
            # denoising loop so its two branches receive genuinely different contexts.
            cfg_context = prompt_embeds

            # Reduce the vram by offload text encoders
            if args.low_vram:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

            # Prepare timesteps
            if hasattr(noise_scheduler, "use_dynamic_shifting") and noise_scheduler.use_dynamic_shifting:
                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device, mu=1)
            else:
                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
            timesteps = noise_scheduler.timesteps

            # Wan2.2 combines a high-noise and a low-noise expert at `boundary`. Select the student
            # transformer for the current timestep. For `boundary_type="full"` only one transformer
            # exists, so this simply returns it.
            num_train_ts = noise_scheduler.config.num_train_timesteps
            if args.boundary_type != "full":
                def get_student_transformer(t):
                    is_high = t >= boundary * num_train_ts
                    if args.boundary_type == "high":
                        return transformer3d if is_high else low_transformer3d
                    else:  # boundary_type == "low"
                        return transformer3d if not is_high else high_transformer3d
            else:
                def get_student_transformer(t):
                    return transformer3d

            # Prepare latent variables
            vae_scale_factor = vae.spatial_compression_ratio

            # Denoising steps
            with accelerator.accumulate(transformer3d):
                do_classifier_free_guidance = True
                
                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = 1
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)
                    
                    gt_latents = _batch_encode_vae(pixel_values)

                    # The noise latent must share the shape of the (bucket-adapted)
                    # encoded frames, otherwise it mismatches gt / control / ref latents.
                    latents = torch.randn(gt_latents.shape, device=accelerator.device, dtype=weight_dtype)

                    # control latents
                    control_latents = _batch_encode_vae(control_pixel_values)
                    # Make control latents to zero
                    for bs_index in range(control_latents.size()[0]):
                        if rng is None:
                            zero_init_control_latents_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                        else:
                            zero_init_control_latents_conv_in = rng.choice([0, 1], p = [0.90, 0.10])

                        if zero_init_control_latents_conv_in:
                            control_latents[bs_index] = control_latents[bs_index] * 0

                    # camera latents
                    control_camera_latents = None
                
                    # ref latents
                    full_ref = None
                    ref_latents = _batch_encode_vae(ref_pixel_values)
                    if args.add_full_ref_image_in_self_attention:
                        full_ref = ref_latents[:, :, 0].clone()

                    ref_latents_conv_in = torch.zeros_like(latents).to(ref_latents.device, ref_latents.dtype)
                    ref_latents_conv_in[:, :, :1] = ref_latents
                    for bs_index in range(ref_latents.size()[0]):
                        if rng is None:
                            zero_init_ref_latents_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                        else:
                            zero_init_ref_latents_conv_in = rng.choice([0, 1], p = [0.90, 0.10])

                        if clip_idx[bs_index] != 0 or (zero_init_ref_latents_conv_in and latents.size()[1] != 1):
                            ref_latents_conv_in[bs_index, :, :1] = ref_latents_conv_in[bs_index, :, :1] * 0

                        if args.add_full_ref_image_in_self_attention:
                            if rng is None:
                                zero_init_full_ref_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                            else:
                                zero_init_full_ref_conv_in = rng.choice([0, 1], p = [0.90, 0.10])
                            if clip_idx[bs_index] == 0 or zero_init_full_ref_conv_in:
                                full_ref[bs_index] = full_ref[bs_index] * 0
                    
                    if args.add_inpaint_info:
                        t2v_flag = [(_mask == 1).all() for _mask in mask]
                        new_t2v_flag = []
                        for _mask in t2v_flag:
                            if _mask and np.random.rand() < 0.90:
                                new_t2v_flag.append(0)
                            else:
                                new_t2v_flag.append(1)
                        t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)

                        mask = rearrange(mask, "b f c h w -> b c f h w")
                        mask = torch.concat(
                            [
                                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                mask[:, :, 1:]
                            ], dim=2
                        )
                        mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                        mask = mask.transpose(1, 2)
                        mask_conditions = F.interpolate(mask[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(accelerator.device, weight_dtype)
                        mask = resize_mask(1 - mask, latents)

                        # Encode inpaint latents.
                        mask_latents = _batch_encode_vae(mask_pixel_values)

                        inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                        inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents
                    else:
                        inpaint_latents = None

                    if control_latents is None:
                        if inpaint_latents is None:
                            control_latents = ref_latents_conv_in
                        else:
                            control_latents = inpaint_latents
                    else:
                        if inpaint_latents is None:
                            control_latents = torch.cat([control_latents, ref_latents_conv_in], dim = 1)
                        else:
                            control_latents = torch.cat([control_latents, inpaint_latents], dim = 1)
                
                control_latents_cfg = (
                    torch.cat([control_latents] * 2) if do_classifier_free_guidance else control_latents
                )

                if hasattr(noise_scheduler, "init_noise_sigma"):
                    latents = latents * noise_scheduler.init_noise_sigma

                cond_latent = latents.clone().to(device=accelerator.device, dtype=weight_dtype)

                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                # Prepare extra step kwargs.
                extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.eta)

                bsz, channel, num_frames, height, width = latents.size()
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )
                
                num_reward_steps = args.num_reward_steps
                use_latent = args.latent_reward_weight > 0

                backprop_step_list = None
                if args.backprop:
                    if args.backprop_step_list is not None:
                        backprop_step_list = args.backprop_step_list
                    else:
                        if args.backprop_strategy == "last":
                            backprop_step_list = [args.num_inference_steps - 1]
                        elif args.backprop_strategy == "tail":
                            backprop_step_list = list(range(args.num_inference_steps))[-args.backprop_num_steps:]
                        elif args.backprop_strategy == "uniform":
                            interval = max(1, args.num_inference_steps // args.backprop_num_steps)
                            random_start = random.randint(0, interval)
                            backprop_step_list = [random_start + i * interval for i in range(args.backprop_num_steps)]
                        elif args.backprop_strategy == "random":
                            backprop_step_list = random.sample(
                                range(args.backprop_random_start_step, args.backprop_random_end_step + 1),
                                args.backprop_num_steps,
                            )

               
                latent_rewards = []
                for i, t in enumerate(tqdm(timesteps)):
                    if i >= num_reward_steps:
                        break
                    latent_model_input = latents
                    if args.stop_latent_model_input_gradient:
                        latent_model_input = latent_model_input.detach()
                    if hasattr(noise_scheduler, "scale_model_input"):
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                    t_expand = torch.tensor([t] * latent_model_input.shape[0], device=accelerator.device).to(
                        dtype=latent_model_input.dtype
                    )

                    # predict noise model_output
                    with accelerator.autocast():
                        noise_pred, latent_rewards = get_student_transformer(t)(
                            x=latent_model_input,
                            context=uncond_prompt_embeds,
                            t=t_expand,
                            seq_len=seq_len,
                            y=control_latents,
                            clip_fea=None,
                            full_ref=full_ref,
                            latent_rewards=latent_rewards,
                            reward_flag=use_latent and (i < num_reward_steps),
                            denoise_step=i + 1,
                        )
                    
                    if args.backprop:
                        if i in backprop_step_list:
                            noise_pred = noise_pred
                        else:
                            noise_pred = noise_pred.detach()

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

               
                teacher_timesteps = timesteps if use_latent else []
                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
                cond_latent_rewards = []
                do_classifier_free_guidance = True
               
                full_ref_cfg = torch.cat([full_ref] * 2) if full_ref is not None else None
                for i, t in enumerate(tqdm(teacher_timesteps)):
                    if i >= num_reward_steps:
                        break

                    latent_model_input = torch.cat([cond_latent] * 2) if do_classifier_free_guidance else cond_latent
                    if hasattr(noise_scheduler, "scale_model_input"):
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                    t_expand = torch.tensor([t] * latent_model_input.shape[0], device=accelerator.device).to(
                        dtype=latent_model_input.dtype
                    )
                
                    with torch.no_grad(), accelerator.autocast():
                        cond_noise_pred, cond_latent_rewards = transformer3d_ori(
                            x=latent_model_input,
                            context=cfg_context,
                            t=t_expand,
                            seq_len=seq_len,
                            y=control_latents_cfg,
                            clip_fea=None,
                            full_ref=full_ref_cfg,
                            latent_rewards=cond_latent_rewards,
                            reward_flag=use_latent and (i < num_reward_steps),
                            denoise_step=i + 1,
                        )
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_cond, noise_pred_text = cond_noise_pred[0], cond_noise_pred[1]
                        noise_pred = noise_pred_cond + args.guidance_scale * (noise_pred_text - noise_pred_cond)
                        
                    # compute the previous noisy sample x_t -> x_t-1
                    cond_latent = noise_scheduler.step(noise_pred, t, cond_latent, **extra_step_kwargs, return_dict=False)[0]
                
                mps_loss, mps_reward = None, None
                if mps_reward_fn is not None:
                    if hasattr(vae, "enable_cache_in_vae"):
                        vae.enable_cache_in_vae()
                    sampled_latents = latents[:, :, : args.num_decoded_latents]
                    sampled_frames = vae.decode(sampled_latents.to(vae.device, vae.dtype))[0]
                    sampled_frames = sampled_frames.clamp(-1, 1)
                    sampled_frames = (sampled_frames / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
                    if hasattr(vae, "disable_cache_in_vae"):
                        vae.disable_cache_in_vae()
                    # Sample a few frames for the MPS reward to limit compute/memory.
                    if args.mps_num_sampled_frames is not None and sampled_frames.size(2) > args.mps_num_sampled_frames:
                        num_f = sampled_frames.size(2) - 1
                        fidx = torch.linspace(0, num_f, steps=args.mps_num_sampled_frames).long()
                        sampled_frames = sampled_frames[:, :, fidx, :, :]
                    # Gradient flows: MPS model is frozen but differentiable, and it moves
                    # the frames to its own device (cpu by default) which autograd supports.
                    mps_loss, mps_reward = mps_reward_fn(sampled_frames, train_prompt)
                    del sampled_frames
                    accelerator.print({"mps_loss": mps_loss.detach(), "mps_reward": mps_reward.detach()})

                del latents, cond_latent

                latent_loss, latent_reward = None, None
                if use_latent and latent_rewards:
                    latent_rewards = [_extract_latent_reward(i) for i in latent_rewards][:num_reward_steps]
                    cond_latent_rewards = [_extract_latent_reward(i)[1:2] for i in cond_latent_rewards][:num_reward_steps]
                    latent_loss, latent_reward = latent_reward_fn.forward(latent_rewards, cond_latent_rewards)

                if mps_loss is not None and latent_loss is not None:
                    mps_loss = mps_loss.to(latent_loss.device, latent_loss.dtype)
                    mps_reward = mps_reward.to(latent_loss.device, latent_loss.dtype)
                    loss = args.mps_reward_weight * mps_loss + args.latent_reward_weight * latent_loss
                    reward = args.mps_reward_weight * mps_reward + args.latent_reward_weight * latent_reward
                elif mps_loss is not None:
                    loss, reward = mps_loss, mps_reward
                elif latent_loss is not None:
                    loss, reward = latent_loss, latent_reward
                else:
                    loss = torch.zeros((), device=accelerator.device, dtype=weight_dtype)
                    reward = torch.zeros((), device=accelerator.device, dtype=weight_dtype)

                # Gather the losses and rewards across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_reward = accelerator.gather(reward.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_reward += avg_reward.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    # If use_deepspeed, `total_norm` cannot be logged by accelerator.
                    if not args.use_deepspeed:
                        accelerator.log({"total_norm": total_norm}, step=global_step)
                    else:
                        if hasattr(optimizer, "optimizer") and hasattr(optimizer.optimizer, "_global_grad_norm"):
                            accelerator.log({"total_norm":  optimizer.optimizer._global_grad_norm}, step=global_step)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "train_reward": train_reward}, step=global_step)
                train_loss = 0.0
                train_reward = 0.0

                if global_step % args.checkpointing_steps == 0 and global_step != 0:
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                    if args.use_deepspeed or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        
                        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(accelerator_save_path)
                        logger.info(f"Saved state to {accelerator_save_path}")
                
                # Validation (distributed)
                if do_validation and (global_step % args.validation_steps) == 0 and 0:
                    if args.validation_prompts is None and args.validation_prompt_path.endswith(".txt"):
                        validation_prompts = []
                        with open(args.validation_prompt_path, "r") as f:
                            for line in f:
                                validation_prompts.append(line.strip())
                        # Do not select randomly to ensure that `args.validation_prompts` is the same for each process.
                        args.validation_prompts = validation_prompts[:args.validation_batch_size]
                    validation_prompts_idx = [(i, p) for i, p in enumerate(args.validation_prompts)]

                    if hasattr(vae, "enable_cache_in_vae"):
                        vae.enable_cache_in_vae()
                    accelerator.wait_for_everyone()
                    with accelerator.split_between_processes(validation_prompts_idx) as splitted_prompts_idx:
                        validation_loss, validation_reward = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            clip_image_encoder,
                            transformer3d,
                            network,
                            loss_fn,
                            config,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            splitted_prompts_idx
                        )
                        if validation_loss is not None and validation_reward is not None:
                            avg_validation_loss = accelerator.gather(validation_loss).mean()
                            avg_validation_reward = accelerator.gather(validation_reward).mean()
                            accelerator.print(avg_validation_loss, avg_validation_reward)
                            if accelerator.is_main_process:
                                accelerator.log(
                                    {"validation_loss": avg_validation_loss, "validation_reward": avg_validation_reward},
                                    step=global_step
                                )
                    
                    accelerator.wait_for_everyone()
            
            logs = {"step_loss": loss.detach().item(), "step_reward": reward.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
             
            if global_step >= args.max_train_steps:
                break

if __name__ == "__main__":
    main()
