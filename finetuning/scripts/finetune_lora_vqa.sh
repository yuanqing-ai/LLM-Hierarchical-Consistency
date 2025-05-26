#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="/projectnb/ivc-ml/yuan/model_zoo/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=16
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Find a free port
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

# --data_path /projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Hier_Training/new_full_taxonomy_qa_output.json \

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

NCCL_P2P_DISABLE=1 deepspeed --master_port $FREE_PORT src/training/train.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.2 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path data/training/train_plant_img.json \
    --image_folder /path/to/your/image/folder \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/testing_lora_full_img_mcq_5e-5_0.2_new \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 20 \
    --dataloader_num_workers 4 