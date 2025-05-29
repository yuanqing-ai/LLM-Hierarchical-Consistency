import json
import torch
import random
import base64
import os
import gc
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
from PIL import Image

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_qwen(json_file, output_dir, batch_size=1, checkpoint_interval=10, start_idx=0):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = "/projectnb/ivc-ml/yuan/model_zoo/Qwen2.5-VL-7B-Instruct"
    
    # Load model with memory efficiency configurations
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load data
    with open(json_file, "r") as f:
        test_data = json.load(f)
    
    # Lists to store results
    lasts = []
    avgs = []
    
    # Resume from checkpoint if specified
    if start_idx > 0:
        print(f"Resuming from index {start_idx}")
        if os.path.exists(os.path.join(output_dir, "last_partial.pt")):
            lasts = torch.load(os.path.join(output_dir, "last_partial.pt"))
        if os.path.exists(os.path.join(output_dir, "avg_partial.pt")):
            avgs = torch.load(os.path.join(output_dir, "avg_partial.pt"))
    
    # Process samples with progress bar
    for i, entry in enumerate(tqdm(test_data[start_idx:], desc="Processing")):
        idx = i + start_idx
        image_path = entry["image"]
        
        # Skip if already processed
        if idx < start_idx:
            continue
        
        try:
            # Clear CUDA cache before processing each image
            if idx % 500 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Process image and get embeddings
            last, avg = infer_level(image_path, processor, model)
            
            # Add results to lists
            lasts.append(last.cpu())  # Move tensor to CPU to save GPU memory
            avgs.append(avg.cpu())    # Move tensor to CPU to save GPU memory
            
            # Save checkpoint periodically
            if (idx + 1) % checkpoint_interval == 0:
                print(f"Saving checkpoint at index {idx}")
                torch.save(lasts, os.path.join(output_dir, "last_partial.pt"))
                torch.save(avgs, os.path.join(output_dir, "avg_partial.pt"))
                
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM at index {idx}. Saving progress and exiting...")
            torch.save(lasts, os.path.join(output_dir, "last_partial.pt"))
            torch.save(avgs, os.path.join(output_dir, "avg_partial.pt"))
            with open(os.path.join(output_dir, "last_processed_idx.txt"), "w") as f:
                f.write(str(idx))
            print(f"Restart the script with start_idx={idx+1}")
            return
        except Exception as e:
            print(f"Error processing image {image_path} at index {idx}: {str(e)}")
            continue
    
    # Concatenate and save final results
    try:
        if lasts:
            lasts_tensor = torch.cat(lasts, dim=0)
            torch.save(lasts_tensor, os.path.join(output_dir, "last.pt"))
        
        if avgs:
            avgs_tensor = torch.cat(avgs, dim=0)
            torch.save(avgs_tensor, os.path.join(output_dir, "avg.pt"))
    except Exception as e:
        print(f"Error saving final results: {str(e)}")
        # Save individual tensors as fallback
        torch.save(lasts, os.path.join(output_dir, "last_list.pt"))
        torch.save(avgs, os.path.join(output_dir, "avg_list.pt"))

def infer_level(image_path, processor, model):
    """
    Helper function to infer label for a specific image.
    
    Args:
        image_path: Path to the image
        processor: The Qwen processor
        model: Qwen model
    
    Returns:
        last: Last hidden state for the image token
        avg: Average hidden state for the image tokens
    """
    try:
        # Load and preprocess image with reduced memory footprint
        image = Image.open(image_path).convert("RGB")
        
        # Create messages
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

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        width, height = image_inputs[0].size

        # Resize large images to save memory
        max_dim = max(width, height)
        if max_dim > 1024:  # Lowered from 1680 to save memory
            scale_factor = 1024 / max_dim
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            new = []
            for img in image_inputs:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=28,
                    min_pixels=56*56,
                    max_pixels=28 * 28 * 640,  # Reduced from 1280 to save memory
                )
                new.append(img.resize((resized_width, resized_height), Image.LANCZOS))
            image_inputs = new

        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate with memory optimizations
        with torch.inference_mode():
            # Use smaller max_new_tokens to save memory
            generated_ids = model.generate(
                **inputs,
                output_hidden_states=True,
                max_new_tokens=1,
                return_dict_in_generate=True,
                use_cache=True  # Enable KV caching for memory efficiency
            )
            
            # Extract hidden states
            image_ids = (generated_ids.sequences[0] == 151655).nonzero(as_tuple=True)[0].tolist()
            all_hid = generated_ids.hidden_states[0][-1][0][image_ids]
            last = all_hid[-1].unsqueeze(0)
            avg = torch.mean(all_hid, dim=0, keepdim=True)

        return last, avg

    except torch.cuda.OutOfMemoryError:
        raise  # Re-raise OOM error to be handled by the caller
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        # Return zero tensors as fallback
        device = next(model.parameters()).device
        return torch.zeros((1, model.config.hidden_size), device=device), torch.zeros((1, model.config.hidden_size), device=device)

if __name__ == "__main__":
    # Set random seed
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    json_file = "probing/probing_data/VLM/Inat21_Plant_all.json"
    output_dir = "probing/probing_data/VLM/Qwen_Inat21_Plant_llm"
    
    # Read last processed index if available
    start_idx = 0
    last_idx_file = os.path.join(output_dir, "last_processed_idx.txt")
    if os.path.exists(last_idx_file):
        with open(last_idx_file, "r") as f:
            start_idx = int(f.read().strip()) + 1
    
    # Run with memory optimizations
    test_qwen(
        json_file=json_file, 
        output_dir=output_dir,
        batch_size=1,  # Process one image at a time
        checkpoint_interval=500,  # Save checkpoints more frequently
        start_idx=start_idx  # Resume from last processed index
    )