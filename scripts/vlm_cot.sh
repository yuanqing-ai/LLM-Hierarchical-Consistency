#! /bin/bash

## LLaVA-OV
python evaluation/vllm/llava/eval_natural_plant.py --prompt_order 0  \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/llava/eval_animal.py --prompt_order 0  \
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/llava/eval_CUB.py --prompt_order 0  \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file

## InternVL2.5-8B
python evaluation/vllm/internvl/eval_img_natrual_plant.py --prompt_order 0 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/internvl/eval_img_animal.py --prompt_order 2\
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/vllm/internvl/eval_img_cub.py --prompt_order 0 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file


## Qwen2.5-VL-7B-Instruct
python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
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








