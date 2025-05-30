#!/bin/bash

# Qwen2.5-IT
python evaluation/og_llm/qwen/eval_text_og_animal.py --prompt_order 2\
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2.5-IT_path
python evaluation/og_llm/qwen/eval_text_og_artifact.py --prompt_order 2\
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2.5-IT_path
python evaluation/og_llm/qwen/eval_text_og_natural.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2.5-IT_path
python evaluation/og_llm/qwen/eval_text_og_natural.py --prompt_order  1\
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2.5-IT_path
python evaluation/og_llm/qwen/eval_text_og_cub.py --prompt_order 1 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2.5-IT_path

# Qwen2-IT
python evaluation/og_llm/qwen/eval_text_og_animal.py --prompt_order  2\
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2-IT_path
python evaluation/og_llm/qwen/eval_text_og_artifact.py --prompt_order 2 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2-IT_path
python evaluation/og_llm/qwen/eval_text_og_natural.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2-IT_path
python evaluation/og_llm/qwen/eval_text_og_natural.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2-IT_path
python evaluation/og_llm/qwen/eval_text_og_cub.py --prompt_order  1 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_qwen_2-IT_path


# Qwen2.5
python evaluation/og_llm/qwen/eval_text_og_animal.py --prompt_order 2 \
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file --base
python evaluation/og_llm/qwen/eval_text_og_artifact.py --prompt_order 2 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file --base
python evaluation/og_llm/qwen/eval_text_og_natural.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file --base
python evaluation/og_llm/qwen/eval_text_og_natural.py --prompt_order 1 \
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file --base   
python evaluation/og_llm/qwen/eval_text_og_cub.py --prompt_order 1 \
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file --base

# InternLM2.5
python evaluation/og_llm/interlm2.5/eval_text_natural.py --prompt_order 2 \
--test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_internlm_2.5_path
python evaluation/og_llm/interlm2.5/eval_text_natural.py --prompt_order  2\
--test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_internlm_2.5_path
python evaluation/og_llm/interlm2.5/eval_text_artifact.py --prompt_order 1 \
--test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_internlm_2.5_path
python evaluation/og_llm/interlm2.5/eval_text_animal.py --prompt_order  1\
--test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_internlm_2.5_path
python evaluation/og_llm/interlm2.5/eval_text_cub.py --prompt_order  1\
--test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
--output_file your_output_file --model_path your_internlm_2.5_path