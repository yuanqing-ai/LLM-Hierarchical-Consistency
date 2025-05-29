import sys
import json
import copy
import torch
import random
import sys
from PIL import Image
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import argparse

def test_llava(json_file, output_file,prompt_order, model_path="lmms-lab/llava-onevision-qwen2-7b-ov", device="cuda"):
    device_map = "auto"
    llava_model_args = { "multimodal": True, "attn_implementation": "sdpa","overwrite_config": {"image_aspect_ratio": "pad"}}
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, "llava_qwen", device_map=device_map, **llava_model_args)
    model.eval()

    with open(json_file, "r") as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]
    results = []
    checkpoint_interval=20
    for i, entry in enumerate(tqdm(test_data, desc="Processing")):
        image_path = entry["image"]
        label = entry["label"]
  
        level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()])
        choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()])
        
        if not level_keys or not choices_keys:
            print(f"Warning: Missing level or choices data for {image_path}")
            continue
            
        # Process image
        try:
            images = [Image.open(image_path).convert("RGB")]
            image_tensors = process_images(images, image_processor, model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue

        result_entry = {
            "image": image_path,
            "label": label,
        }
        image_sizes = [image.size for image in images]
        
        # Process each hierarchy level
        for t, (level_key, choices_key) in enumerate(zip(level_keys, choices_keys)):
            level_number = level_key[5:]  # Extract the level number
            ground_truth = entry[level_key]
            choices = entry[choices_key]
            
            # Skip levels with None values
            if ground_truth is None:
                continue
                
            random.shuffle(choices)
            choice_map = {chr(65 + i): opt for i, opt in enumerate(choices)}
            if prompt_order == 0:
                prompt_template = "What is the taxonomic category of the food item in this image?"
            elif prompt_order == 1:
                prompt_template = "How can the food item in this image be categorized in taxonomy?"
            elif prompt_order == 2:
                prompt_template = "Based on classification, what type of food is this?"
            elif prompt_order == 3:
                prompt_template = "What is the hierarchical class of the food shown here?"
            else:
                prompt_template = "Where does this food item belong in the taxonomic hierarchy?"
            # Store results for this level
            predicted_letter, predicted_label = infer_level(tokenizer, model, image_tensors,image_sizes,prompt_template, choice_map, device)
            result_entry[f"ground_truth_level{level_number}"] = ground_truth
            result_entry[f"predicted_level{level_number}_letter"] = predicted_letter
            result_entry[f"predicted_level{level_number}"] = predicted_label
            result_entry[f"choices_level{level_number}"] = choice_map
        
        results.append(result_entry)
         # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0 or i == len(test_data) - 1:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Checkpoint saved ({i+1}/{len(test_data)} entries)")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")


def infer_level(tokenizer, model, image_tensors,image_sizes, prompt_template, choice_map, device):
    # Format the question with options
    question = "\n" + prompt_template + "\n" + "\n".join([f"{key}. {val}" for key, val in choice_map.items()])+ "\nAnswer with the option's letter from the given choices directly."
    print(question)
    prompt = DEFAULT_IMAGE_TOKEN + question
    conv_template="qwen_animal"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Generate response
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    try:
        cont = model.generate(input_ids, images=image_tensors,image_sizes=image_sizes, do_sample=False, temperature=0, max_new_tokens=10)
        response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        print(response)
        predicted_letter = next((key for key in choice_map.keys() if key in response), "Unknown")
        predicted_label = choice_map.get(predicted_letter, "Unknown")
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        predicted_letter = "Unknown"
        predicted_label = "Unknown"

    return predicted_letter, predicted_label


if __name__ == "__main__":
    seed = 42
    random.seed(seed)

    parser = argparse.ArgumentParser(description="Evaluate LLaVA prompts on the CUB." )
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
        default='lmms-lab/llava-onevision-qwen2-7b-ov',
        help="Specify the model path."
    )

    parser.add_argument(
        "--test_set",
        type=str,
        default=None,
        help="Specify the test path."
    )

    args = parser.parse_args()
    json_file = args.test_set

    test_llava(json_file,output_file,  args.prompt_order,args.model_path)