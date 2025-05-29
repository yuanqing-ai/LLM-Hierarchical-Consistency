#! /bin/bash

## LLaVA-OV
python evaluation/llm/llava/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/llava/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/llava/eval_text_artifact.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/llava/eval_text_animal.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_animal_with_similarity_choice.jsonl \
--output_file your_output_file  

python evaluation/llm/llava/eval_text_cub.py --prompt_order  \
--test_set data/annotations/similar_choices/CUB200_with_similarity_choice.jsonl \
--output_file your_output_file     

## InternVL2.5-8B
python evaluation/llm/internvl/eval_text_natural.py --prompt_order \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/internvl/eval_text_natural.py --prompt_order \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/internvl/eval_text_artifact.py --prompt_order \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/internvl/eval_text_animal.py --prompt_order \
--test_set data/annotations/similar_choices/imagenet_animal_with_similarity_choice.jsonl \
--output_file your_output_file

python evaluation/llm/internvl/eval_text_cub.py --prompt_order \
--test_set data/annotations/similar_choices/CUB200_with_similarity_choice.jsonl \
--output_file your_output_file

## InternVL3-8B

python evaluation/llm/internvl/eval_text_natural.py --prompt_order \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3


python evaluation/llm/internvl/eval_text_natural.py --prompt_order \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

python evaluation/llm/internvl/eval_text_artifact.py --prompt_order \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

python evaluation/llm/internvl/eval_text_animal.py --prompt_order \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3

python evaluation/llm/internvl/eval_text_cub.py --prompt_order \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file \
--internvl_3


## Qwen2.5-VL-7B-Instruct

python evaluation/llm/qwen/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_artifact.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_animal.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_animal_with_similarity_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file  

python evaluation/llm/qwen/eval_text_cub.py --prompt_order  \
--test_set data/annotations/similar_choices/CUB200_with_similarity_choice.jsonl \
--model_path your_qwen_7B_path \
--output_file your_output_file     


## Qwen2.5-VL-32B-Instruct

python evaluation/llm/qwen/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_artifact.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_animal.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_animal_with_similarity_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file  

python evaluation/llm/qwen/eval_text_cub.py --prompt_order  \
--test_set data/annotations/similar_choices/CUB200_with_similarity_choice.jsonl \
--model_path your_qwen_32B_path \
--output_file your_output_file      

## Qwen2.5-VL-72B-Instruct

python evaluation/llm/qwen/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_natural.py --prompt_order  \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_artifact.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file

python evaluation/llm/qwen/eval_text_animal.py --prompt_order  \
--test_set data/annotations/similar_choices/imagenet_animal_with_similarity_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file  

python evaluation/llm/qwen/eval_text_cub.py --prompt_order  \
--test_set data/annotations/similar_choices/CUB200_with_similarity_choice.jsonl \
--model_path your_qwen_72B_path \
--output_file your_output_file      


## GPT-4O

python evaluation/llm/gpt/gpt_natural_text.py \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/gpt/gpt_natural_text.py \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/gpt/gpt_imgnet_text.py \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file

python evaluation/llm/gpt/gpt_imgnet_text.py \
--test_set data/annotations/similar_choices/imagenet_animal_with_similarity_choice.jsonl \
--output_file your_output_file

python evaluation/llm/gpt/gpt_cub_text.py \
--test_set data/annotations/similar_choices/CUB200_with_similarity_choice.jsonl \
--output_file your_output_file


