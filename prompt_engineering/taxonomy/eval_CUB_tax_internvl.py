import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import json
import copy
import random
from tqdm import tqdm
import base64
import os
import argparse
import time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_internvl(json_file, output_file, prompt_order):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

    path = '/projectnb/ivc-ml/yuan/model_zoo/InternVL2_5-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    context_file = "data/taxonomies/hierarchy_CUB200.json"
    with open(context_file, "r") as f:
        context = json.load(f)
    
    context = str(context)


    with open(json_file, "r") as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]
    
    results = []
    checkpoint_interval = 20  # Save results every 10 entries
    
    for i, entry in enumerate(tqdm(test_data, desc="Processing")):
        # if i == 21:
        #     break
        image_path = entry["image"]
        label = entry["label"]
        
        # Find all level keys and choices_level keys in entry
        level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()])
        choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()])
        
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
        pixel_values = load_image(image_path, max_num=4).to(torch.bfloat16).cuda()
        # 处理每一层的预测
        for t, (level_key, choices_key) in enumerate(zip(level_keys, choices_keys)):
            level_number = level_key[5:]  # Extract the level number
            # if level_key == "level1":
            #     continue
            ground_truth = entry[level_key]
            choices = entry[choices_key]
            
            # Skip levels with None values
            if ground_truth is None:
                continue
            
            # Shuffle options and create letter mapping
            random.shuffle(choices)
            # choice_map = {chr(65 + i): opt for i, opt in enumerate(choices)}
            if prompt_order == 0:
                if t==0:
                    #prompt_template = f"Given the {label}, what is its taxonomic classification at the order level?"
                    prompt_template=f"Based on taxonomy, what is the order of the bird in this image?"
                elif t==1:
                    prompt_template=f"Based on taxonomy, what is the family of the bird in this image?"
                elif t==2:
                    prompt_template=f"Based on taxonomy, what is the genus of the bird in this image?"
                else:
                    prompt_template=f"Based on taxonomy, what is the species of the bird in this image?"
            elif prompt_order == 1:
                if t==0:
                    prompt_template = f"Based on the image, what is the taxonomic classification at the order level?"
                elif t==1:
                    prompt_template = f"Based on the image, what is the taxonomic classification at the family level?"
                elif t==2:
                    prompt_template = f"Based on the image, what is the taxonomic classification at the genus level?"
                else:
                    prompt_template = f"Based on the image, what is the taxonomic classification at the species level?"
            elif prompt_order == 2:
                prompt_template = "What is the taxonomic classification of the bird in this image?"
            elif prompt_order == 3:
                prompt_template = "How can the bird in this image be categorized taxonomically?"
            else:
                prompt_template = "What is the systematic position of the bird shown in the image?"
            choice_map = {chr(65 + j): opt for j, opt in enumerate(choices)}
            predicted_letter, predicted_label, response = infer_level(prompt_template, choice_map, tokenizer, model, pixel_values, context)
            
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


def infer_level(prompt_template, choice_map, tokenizer, model, pixel_values, context):
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
    question = "Here is a taxonomy: " + context + '<image>\n' + question

    try:
        generation_config = dict(max_new_tokens=5, do_sample=False)
        response, history = model.chat(tokenizer, question, history=[],max_new_tokens=5, do_sample=False)

        # print(f'User: {question}\nAssistant: {response}')
        
        # Extract predicted letter from response
        predicted_letter = next((key for key in choice_map.keys() if key in response), "Unknown")
        predicted_label = choice_map.get(predicted_letter, "Unknown")
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        predicted_letter = "Error"
        predicted_label = "Error"

    return predicted_letter, predicted_label, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the QWEN test script with custom arguments.")
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
        help="Specify the test set."
    )
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    json_file = args.test_set
    
    test_internvl(json_file, args.output_file, args.prompt_order)