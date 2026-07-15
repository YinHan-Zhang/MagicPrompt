export MODEL_NAME="MagicPrompt/models/Wan2.2-Fun-5B-Control"
export DATASET_NAME="MagicPrompt/datasets/TikTok_dataset"
export DATASET_META_NAME="MagicPrompt/datasets/TikTok_dataset/metadata_control.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" scripts/control2v/train_control.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2e-05 \
  --output_dir="output_dir_wan2.2_fun_control" \
  --lr_warmup_steps=100 \
  --seed=42 \
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
  --trainable_modules "soft_k" "soft_v" "soft_scale" "soft_bias" "patch_embedding"