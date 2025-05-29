#! /bin/bash

## VLM Probing for CUB200
python probing/vlm_probing.py \
    --dataset CUB200 \
    --data probing/probing_data/VLM/CUB200_all.jsonl \
    --taxonomy data/taxonomies/hierarchy_CUB200.json \
    --feature_path /path/to/CUB200_Qwen_features.pt \
    --model_name Qwen \
    --probe linear \
    --output_path your_output_file

## VLM Probing for Inat21-Plant
python probing/vlm_probing.py \
    --dataset Inat21-Plant \
    --data probing/probing_data/VLM/Inat21_Plant_all.json \
    --taxonomy data/taxonomies/hierarchy_Inat_plantae.json \
    --feature_path /path/to/Inat21_Plant_Qwen_features.pt \
    --model_name Qwen \
    --probe linear \
    --output_path /path/to/output/Inat21_Plant_results

