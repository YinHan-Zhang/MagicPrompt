export MODEL_NAME="MagicPrompt/models/Wan2.2-Fun-5B-Control"
export DATASET_NAME="MagicPrompt/datasets/TikTok_dataset"
export DATASET_META_NAME="MagicPrompt/datasets/TikTok_dataset/metadata_control.json"


# NOTE: this launches train_reward.py (the variant that has the MPS reward wired in),
# NOT train_reward_latent.py. train_reward_latent.py is the latent-only baseline.
CUDA_VISIBLE_DEVICES=7 accelerate launch --mixed_precision="bf16" scripts/control2v/train_reward.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=1 \
  --checkpointing_steps=100 \
  --learning_rate=2e-05 \
  --output_dir="checkpoint_reward" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --num_inference_steps=10 \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="full" \
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_inpaint_info \
  --add_full_ref_image_in_self_attention \
  --low_vram \
  --vae_gradient_checkpointing \
  --trainable_modules "soft_k" "soft_v" "soft_scale" "soft_bias" "patch_embedding" "text_soft_k" "text_soft_v" \
  --use_mps_reward \
  --mps_reward_weight=0.5 \
  --latent_reward_weight=0.5 \
  --mps_num_sampled_frames=4 \
  --mps_reward_device="cpu" \
  --num_reward_steps=10 \
  --num_decoded_latents=3 \
  --backprop \
  --backprop_strategy="tail" \
  --backprop_num_steps=4 \
  --checkpoints_total_limit=5
