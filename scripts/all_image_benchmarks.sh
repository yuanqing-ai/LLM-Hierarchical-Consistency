#! /bin/bash

## CLIP

python evaluation/clip/openclip.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file
## SigLIP
python evaluation/clip/siglip.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file

## LLaVA-OV
python evaluation/vllm/llava/eval_natural_animal.py --prompt_order 0  \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/llava/eval_natural_plant.py --prompt_order 0  \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/llava/eval_artifact.py --prompt_order 0  \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/llava/eval_animal.py --prompt_order 0  \
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/llava/eval_CUB.py --prompt_order 0  \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/llava/eval_food.py --prompt_order 0 \
--test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
--output_file your_output_file

## InternVL2.5-8B
python evaluation/vllm/internvl/eval_img_natrual_ani.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/internvl/eval_img_natrual_plant.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/internvl/eval_img_artifact.py --prompt_order 0 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/internvl/eval_img_animal.py --prompt_order 2\
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/internvl/eval_img_cub.py --prompt_order 0 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/internvl/eval_img_food.py --prompt_order 0 \
--test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
--output_file your_output_file

## InternVL3-8B

python evaluation/vllm/internvl/eval_img_natrual_ani.py --prompt_order 1\
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

python evaluation/vllm/internvl/eval_img_natrual_plant.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3    

python evaluation/vllm/internvl/eval_img_artifact.py --prompt_order 0 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

python evaluation/vllm/internvl/eval_img_animal.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

python evaluation/vllm/internvl/eval_img_cub.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

python evaluation/vllm/internvl/eval_img_food.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

## Qwen2.5-VL-7B-Instruct

python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_artifact.py --prompt_order 0 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_animal.py --prompt_order 0 \
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file  

python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file     

python evaluation/vllm/qwen/eval_Food101.py --prompt_order 0 \
--test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file


## Qwen2.5-VL-32B-Instruct

python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_artifact.py --prompt_order 0 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_animal.py --prompt_order  0\
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file  

python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file      

python evaluation/vllm/qwen/eval_Food101.py --prompt_order 0 \
--test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file

## Qwen2.5-VL-72B-Instruct

python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_artifact.py --prompt_order 0 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file

python evaluation/vllm/qwen/eval_animal.py --prompt_order 0 \
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file  

python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file      

python evaluation/vllm/qwen/eval_Food101.py --prompt_order 0 \
--test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file

## GPT-4O

python evaluation/vllm/gpt/gpt_img_all.py  \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file \
--dataset_prompt natural_animal

python evaluation/vllm/gpt/gpt_img_all.py \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file \
--dataset_prompt natural_plant

python evaluation/vllm/gpt/gpt_img_all.py \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file \
--dataset_prompt artifact

python evaluation/vllm/gpt/gpt_img_all.py \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file \
--dataset_prompt animal

python evaluation/vllm/gpt/gpt_img_all.py \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file \
--dataset_prompt cub

python evaluation/vllm/gpt/gpt_img_all.py \
--test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
--output_file your_output_file \
--dataset_prompt food
















