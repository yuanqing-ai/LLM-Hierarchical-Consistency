# #!/bin/bash

# MODEL_NAME="/projectnb/ivc-ml/yuan/model_zoo/Qwen2.5-VL-7B-Instruct"
# # MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

# export PYTHONPATH=src:$PYTHONPATH

# python src/merge_lora_weights.py \
#     --model-path /projectnb/ivc-ml/yuan/Qwen2-VL-Finetune/output/testing_lora_full_text_mcq_5e-5_0.2/checkpoint-200 \
#     --model-base $MODEL_NAME  \
#     --save-model-path /projectnb/ivc-ml/yuan/Qwen2-VL-Finetune/output/merged_text_full_mcq_5e-5_0.2\
#     --safe-serialization

#!/bin/bash

MODEL_NAME="/projectnb/ivc-ml/yuan/model_zoo/Qwen2.5-VL-7B-Instruct" # Change this to your base model path
BASE_CHECKPOINT_DIR="/projectnb/ivc-ml/yuan/Qwen2-VL-Finetune/output/testing_lora_full_text_mcq_1e-5_0.2" #Change this to tuned output path
OUTPUT_DIR="/projectnb/ivc-ml/yuan/Qwen2-VL-Finetune/output/merged_text_mcq_1e-5_0.2" #Change this to your output directory
CHECKPOINTS=(100 200) #Change this to your desired checkpoints

export PYTHONPATH=src:$PYTHONPATH

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "Merging checkpoint $ckpt..."
  python src/merge_lora_weights.py \
    --model-path "$BASE_CHECKPOINT_DIR/checkpoint-$ckpt" \
    --model-base "$MODEL_NAME" \
    --save-model-path "$OUTPUT_DIR/$ckpt" \
    --safe-serialization
done
