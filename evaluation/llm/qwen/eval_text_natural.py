import json
import copy
import torch
import random
from tqdm import tqdm
import base64
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_qwen(json_file, output_file, prompt_order,model_path):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    )

    # default processor
    processor = AutoProcessor.from_pretrained(model_path)
    with open(json_file, "r") as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]
    
    results = []
    checkpoint_interval = 20  # Save results every 10 entries
    
    for i, entry in enumerate(tqdm(test_data, desc="Processing")):
        # if i >= 100:
        #     continue
        image_path = entry["image"]
        label = entry["label"]
        
        # Find all level keys and choices_level keys in entry
        level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()], key=lambda x: int(x[5:]))
        choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()], key=lambda x: int(x[13:]))
        
        if not level_keys or not choices_keys:
            print(f"Warning: Missing level or choices data for {image_path}")
            continue
        
        # Process image
        # try:
        #     # Getting the Base64 string
        #     base64_image = encode_image(image_path)
        # except Exception as e:
        #     print(f"Error processing image {image_path}: {str(e)}")
        #     continue
        
        result_entry = {
            "image": image_path,
            "label": label,
        }
        
        # 处理每一层的预测
        for t, (level_key, choices_key) in enumerate(zip(level_keys, choices_keys)):
            level_number = level_key[5:]  # 提取层级数字
            ground_truth = entry[level_key]
            choices = entry[choices_key]
            # wait = input("")
            
            # 判断是否是最后一层（叶子节点）
            is_leaf_node = (t == len(level_keys) - 1)
            
            if is_leaf_node:
                # 如果是叶子节点，直接使用label作为预测结果
                choice_map = {chr(65 + j): opt for j, opt in enumerate(choices)}
                predicted_letter = next((k for k, v in choice_map.items() if v.lower() == label.lower()), "Unknown")
                predicted_label = label
            else:
                # 非叶子节点，使用模型推理
                #Based on taxonomy, where does 'label' fall in terms of order, family, and genus?
                if prompt_order == 0:
                    if t==0:
                        #prompt_template = f"Given the {label}, what is its taxonomic classification at the order level?"
                        prompt_template=f"Based on taxonomy, where does {label} fall in terms of kingdom"
                    elif t==1:
                    # prompt_template=f"Given the {label}, what is its taxonomic classification at the family level?"
                        prompt_template=f"Based on taxonomy, where does {label} fall in terms of phylum"
                    elif t==2:
                    # prompt_template=f"Given the {label}, what is its taxonomic classification at the family level?"
                        prompt_template=f"Based on taxonomy, where does {label} fall in terms of class"
                    elif t==3:
                    # prompt_template=f"Given the {label}, what is its taxonomic classification at the family level?"
                        prompt_template=f"Based on taxonomy, where does {label} fall in terms of order"
                    elif t==4:
                    # prompt_template=f"Given the {label}, what is its taxonomic classification at the family level?"
                        prompt_template=f"Based on taxonomy, where does {label} fall in terms of family"
                    elif t==5:
                    # prompt_template=f"Given the {label}, what is its taxonomic classification at the family level?"
                        prompt_template=f"Based on taxonomy, where does {label} fall in terms of genus"
                    else:
                        prompt_template=f"Based on taxonomy, where does {label} fall in terms of species"
                elif prompt_order == 1:
                    if t==0:
                        prompt_template = f"Given the {label}, what is its taxonomic classification at the kingdom level?"
                    elif t==1:
                        prompt_template = f"Given the {label}, what is its taxonomic classification at the phylum level?"
                    elif t==2:
                        prompt_template = f"Given the {label}, what is its taxonomic classification at the class level?"
                    elif t==3:
                        prompt_template = f"Given the {label}, what is its taxonomic classification at the order level?"
                        # prompt_template=f"Based on taxonomy, where does {label} fall in terms of order"
                    elif t==4:
                        prompt_template = f"Given the {label}, what is its taxonomic classification at the family level?"
                    elif t==5:
                        prompt_template = f"Given the {label}, what is its taxonomic classification at the genus level?"
                        # prompt_template=f"Based on taxonomy, where does {label} fall in terms of family"
                    else:
                        # prompt_template=f"Based on taxonomy, where does {label} fall in terms of genus"
                        prompt_template = f"Given the {label}, what is its taxonomic classification at the species level?"
                elif prompt_order == 2:
                    prompt_template=f"What could {label} be classified as?"
                elif prompt_order == 3:
                    prompt_template=f"How can {label} be taxonomically categorized?"
                else:
                    prompt_template=f"What is the systematic position of {label} in the biological classification hierarchy?"
                choice_map = {chr(65 + j): opt for j, opt in enumerate(choices)}
                predicted_letter, predicted_label, response = infer_level(prompt_template, choice_map, processor, model)
            
            # 存储这一层的结果
            result_entry[f"ground_truth_level{level_number}"] = ground_truth
            result_entry[f"prediction_level{level_number}"] = response
            result_entry[f"predicted_level{level_number}_letter"] = predicted_letter
            result_entry[f"predicted_level{level_number}"] = predicted_label
            result_entry[f"choices_level{level_number}"] = choice_map
        
        results.append(result_entry)
        
        # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0 or i == len(test_data) - 1:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Checkpoint saved ({i+1}/{len(test_data)} entries)")
    
    # Final save (in case the total wasn't divisible by checkpoint_interval)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")


def infer_level(prompt_template, choice_map, processor, model):
    """
    Helper function to infer label for a specific level.
    
    Args:
        tokenizer: LLaVA tokenizer
        model: LLaVA model
        image_tensors: Preprocessed image tensors
        prompt_template: Text prompt for the specific level
        choice_map: Mapping from option letters to option text
        device: Device to run inference on
    
    Returns:
        predicted_letter: The selected option letter
        predicted_label: The corresponding label text
    """
    # Format the question with options
    question = prompt_template + "\n" + "\n".join([f"{key}. {val}" for key, val in choice_map.items()]) + "\nAnswer with the option's letter from the given choices directly."
    # question = "Identify the botanical class of this plant"
    # print(f"Question: {question}")
    # print(f"Prompt: {question}")
    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
        ],
    }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)


    try:
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        # print(f"Response: {response}")
        # Extract predicted letter from response
        predicted_letter = next((key for key in choice_map.keys() if key in response), "Unknown")
        predicted_label = choice_map.get(predicted_letter, "Unknown")
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        predicted_letter = "Error"
        predicted_label = "Error"

    return predicted_letter, predicted_label, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the QWEN text evaluation")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output file."
    )
    parser.add_argument(
        "--prompt_order",
        type=int,
        default=0,
        help="Specify the prompt order."
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default=None,
        help="Specify the test split."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Specify the model."
    )

    args = parser.parse_args()
    print("model:", args.model)

    seed = 42
    random.seed(seed)
    
    json_file = args.test_set
    
    test_qwen(json_file, args.output_file, args.prompt_order,args.model)