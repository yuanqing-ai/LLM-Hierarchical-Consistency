# run_cub200.sh
#!/bin/bash
for i in $(seq -w 0 28); do
  echo "Running CUB200 probing with layer $i"
  python /projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/probing.py \
    --dataset CUB200 \
    --train_hierarchy_path "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/CUB/train_taxonomy.json" \
    --test_hierarchy_path "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/CUB/test_taxonomy.json" \
    --train_feature_dir "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/CUB_code/new_with_gt/train" \
    --test_feature_dir "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/CUB_code/new_with_gt/test" \
    --output_dir "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/cub_results" \
    --layer $i
done

# run_inat21_plant.sh  
#!/bin/bash
for i in $(seq -w 0 28); do
  echo "Running Inat21-Plant probing with layer $i"
  python /projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/probing.py \
    --dataset Plantae \
    --train_hierarchy_path "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/Inaturalist/train_taxonomy.json" \
    --test_hierarchy_path "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/Inaturalist/test_taxonomy.json" \
    --train_feature_dir "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Inat_code/new_with_gt/train" \
    --test_feature_dir "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Inat_code/new_with_gt/test" \
    --output_dir "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/plant_results" \
    --layer $i
done