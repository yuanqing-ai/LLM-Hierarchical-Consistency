import json
import copy
import torch
import random
import argparse
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def test_llm_hierarchy(json_file, output_file, prompt_order, model_path="lmms-lab/llava-onevision-qwen2-7b-ov", device="cuda"):
    device_map = "auto"
    llm_model_args = {}  
    tokenizer, model, _, max_length = load_pretrained_model(model_path, None, "llava_qwen", device_map=device_map, attn_implementation="sdpa", **llm_model_args)
    model.eval()

    test_data = []
    with open(json_file, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    results = []
    checkpoint_interval = 20
    for t, entry in enumerate(tqdm(test_data, desc="Processing")):
        image_path = entry["image"]
        label = entry["label"]
        label=label.lower()
        

        level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()], key=lambda x: int(x[5:]))
        choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()], key=lambda x: int(x[13:]))
        
        if not level_keys or not choices_keys:
            print(f"Warning: Missing level or choices data for {image_path}")
            continue
        
        result_entry = {
            "image": image_path,
            "label": label,
        }
        
        for i, (level_key, choices_key) in enumerate(zip(level_keys, choices_keys)):
            level_number = level_key[5:]  
            ground_truth = entry[level_key]
            choices = entry[choices_key]
            
            is_leaf_node = (i == len(level_keys) - 1)
            
            if is_leaf_node:
                choice_map = {chr(65 + j): opt for j, opt in enumerate(choices)}
                predicted_letter = next((k for k, v in choice_map.items() if v == label), "Unknown")
                predicted_label = label
            else:
                if prompt_order == 0:
                    prompt_template = f"Given the {label}, what is its taxonomic classification?"
                elif prompt_order == 1:
                    prompt_template=f"Based on taxonomy, where does {label} fall?"
                elif prompt_order == 2:
                    prompt_template=f"What could {label} be classified as?"
                elif prompt_order == 3:
                    prompt_template=f"How can {label} be taxonomically categorized?"
                else:
                    prompt_template=f"What is the categorical classification of {label} in the artifact taxonomy?"
                choice_map = {chr(65 + j): opt for j, opt in enumerate(choices)}
                predicted_letter, predicted_label = infer_hierarchy_level(tokenizer, model, prompt_template, choice_map, device)
    
            result_entry[f"ground_truth_level{level_number}"] = ground_truth
            result_entry[f"predicted_level{level_number}_letter"] = predicted_letter
            result_entry[f"predicted_level{level_number}"] = predicted_label
            result_entry[f"choices_level{level_number}"] = choice_map
        
        results.append(result_entry)
        if (t + 1) % checkpoint_interval == 0 or t == len(test_data) - 1:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Checkpoint saved ({i+1}/{len(test_data)} entries)")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")


def infer_hierarchy_level(tokenizer, model, prompt_template, choice_map, device):
    question = prompt_template + "\n" + "\n".join([f"{key}. {val}" for key, val in choice_map.items()]) + "\nAnswer with the option's letter from the given choices directly."
    print(question)
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    try:
        output_ids = model.generate(
                input_ids,
                images=None,
                image_sizes=None,
                do_sample=False,
                temperature=0,
                max_new_tokens=5,
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_letter = next((key for key in choice_map.keys() if key in response), "Unknown")
        predicted_label = choice_map.get(predicted_letter, "Unknown")
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        predicted_letter = "Unknown"
        predicted_label = "Unknown"

    return predicted_letter, predicted_label



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLaVA-OV test script with custom arguments.")
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
        "--model_path",
        type=str,
        default="lmms-lab/llava-onevision-qwen2-7b-ov",
        help="Specify the model paths."
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default=None,
        help="Specify the test set path."
    )

    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    json_file = args.test_set
    
    test_llm_hierarchy(json_file, args.output_file, args.prompt_order,args.model_path)