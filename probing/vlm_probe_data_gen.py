import json
import copy
import torch
import random
from tqdm import tqdm
import base64
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
from PIL import Image


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_qwen(json_file, output_dir):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model_path = "/projectnb/ivc-ml/yuan/model_zoo/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # default processor
    processor = AutoProcessor.from_pretrained(model_path)
    with open(json_file, "r") as f:
        test_data = json.load(f)
        # lines = f.readlines()
        # test_data = [json.loads(line) for line in lines]
    
    results = []
    checkpoint_interval = 20  # Save results every 10 entries
    image_in_last = []
    image_out_last = []
    image_in_avg = []
    image_out_avg = []
    
    for i, entry in enumerate(tqdm(test_data, desc="Processing")):
        # if i  == 5:
        #     break
        image_path = entry["image"]
        label = entry["label"]
        
        # Process image
        try:
            # Getting the Base64 string
            base64_image = encode_image(image_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue
        
        v_i, v_o = infer_level(image_path, processor, model)
        v_i_last = v_i[-1,:].unsqueeze(0)
        v_o_last = v_o[-1,:].unsqueeze(0)
        v_i_avg = torch.mean(v_i, dim=0,keepdim=True)
        v_o_avg = torch.mean(v_o, dim=0,keepdim=True)
        image_in_last.append(v_i_last.cpu())
        image_out_last.append(v_o_last.cpu())
        image_in_avg.append(v_i_avg.cpu())
        image_out_avg.append(v_o_avg.cpu())
        # wait = input("")
    in_last = torch.cat(image_in_last,dim=0)
    out_last = torch.cat(image_out_last,dim=0)
    in_avg = torch.cat(image_in_avg,dim=0)
    out_avg = torch.cat(image_out_avg,dim=0)

    torch.save(in_last, os.path.join(output_dir, "in_last.pt"))
    torch.save(out_last, os.path.join(output_dir, "out_last.pt"))
    torch.save(in_avg, os.path.join(output_dir, "in_avg.pt"))
    torch.save(out_avg, os.path.join(output_dir, "out_avg.pt"))


def infer_level(image_path, processor, model):
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

    messages = [
    {
        "role": "system",
        "content": "You are an assistant."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    width, height = image_inputs[0].size

    max_dim = max(width, height)
    if max_dim > 1680:
        scale_factor = 1680 / max_dim
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        new = []
        for image in image_inputs:
            resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=28,
                        min_pixels=56*56,
                        max_pixels=28 * 28 * 1280,
                    )
            new.append(image.resize((resized_width, resized_height), Image.LANCZOS))
        image_inputs = new
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)


    try:
        with torch.inference_mode():
            generated_ids = model(**inputs)
            v_i = generated_ids.v_i
            v_o = generated_ids.v_o

    except Exception as e:
        print(f"Error in generating response: {str(e)}")

    return v_i, v_o


if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    
    json_file = "probing/probing_data/VLM/Inat21_Plant_all.json"
    output_dir = "probing/probing_data/VLM/Qwen_Inat21_Plant_vision"
    
    test_qwen(json_file, output_dir)