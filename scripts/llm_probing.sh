# run_cub200.sh
#!/bin/bash
for i in $(seq -w 0 28); do
  echo "Running CUB200 probing with layer $i"
  python probing/llm_probing.py \
    --dataset CUB200 \
    --train_hierarchy_path "probing/probing_data/LLM/CUB200/train_taxonomy.json" \
    --test_hierarchy_path "probing/probing_data/LLM/CUB200/test_taxonomy.json" \
    --train_feature_dir "Input your feature path" \
    --test_feature_dir "Input your feature path" \
    --output_dir "specify the output path" \
    --layer $i
done

# run_inat21_plant.sh  
#!/bin/bash
for i in $(seq -w 0 28); do
  echo "Running Inat21-Plant probing with layer $i"
  python probing/llm_probing.py \
    --dataset Inat21-Plant \
    --train_hierarchy_path "probing/probing_data/LLM/Inat21_Plant/train_taxonomy.json" \
    --test_hierarchy_path "probing/probing_data/LLM/Inat21_Plant/test_taxonomy.json" \
    --train_feature_dir "Input your feature path" \
    --test_feature_dir "Input your feature path" \
    --output_dir "specify the output path" \
    --layer $i
done