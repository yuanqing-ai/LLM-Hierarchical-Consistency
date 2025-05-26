import json
import base64
import random
import concurrent.futures
import argparse
import os
from tqdm import tqdm
from openai import OpenAI
from functools import partial
os.environ["OPENAI_API_KEY"] = "" # add your own key

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Process a single entry
def process_entry(entry, client):
    image_path = entry["image"]
    label = entry["label"]
    
    # Find all level keys and choices_level keys in entry
    level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()])
    choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()])
    
    if not level_keys or not choices_keys:
        print(f"Warning: Missing level or choices data for {image_path}")
        return None
    
    # Process image
    try:
        # Getting the Base64 string
        base64_image = encode_image(image_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None
    
    result_entry = {
        "image": image_path,
        "label": label,
    }
    
    # Process each level's prediction
    for t, (level_key, choices_key) in enumerate(zip(level_keys, choices_keys)):
        level_number = level_key[5:]  # Extract level number
        ground_truth = entry[level_key]
        choices = entry[choices_key]
        
        # Determine if this is the last level (leaf node)
        is_leaf_node = (t == len(level_keys) - 1)
        
        if is_leaf_node:
            # If it's a leaf node, directly use label as prediction
            choice_map = {chr(65 + j): opt for j, opt in enumerate(choices)}
            predicted_letter = next((k for k, v in choice_map.items() if v.lower() == label.lower()), "Unknown")
            predicted_label = label
            response = label
        else:
            # Generate taxonomic level prompt based on current level
            level_names = ["order", "family", "genus", "species"]
            if t < len(level_names):
                prompt_template = f"Given the {label}, what is its taxonomic classification at the {level_names[t]} level?"
            else:
                prompt_template = f"Given the {label}, what is its taxonomic classification at level {t+1}?"
                print(f"Warning: Level name not defined for level {t+1}. Using generic prompt.")
            # print(f"Prompt: {prompt_template}")
                
            choice_map = {chr(65 + j): opt for j, opt in enumerate(choices)}
            predicted_letter, predicted_label, response = infer_level(prompt_template, choice_map, client)
        
        # Store results for this level
        result_entry[f"ground_truth_level{level_number}"] = ground_truth
        result_entry[f"prediction_level{level_number}"] = response
        result_entry[f"predicted_level{level_number}_letter"] = predicted_letter
        result_entry[f"predicted_level{level_number}"] = predicted_label
        result_entry[f"choices_level{level_number}"] = choice_map
    
    return result_entry

def infer_level(prompt_template, choice_map, client):
    """
    Helper function to infer label for a specific level.
    """
    # Format the question with options
    question = prompt_template + "\n" + "\n".join([f"{key}. {val}" for key, val in choice_map.items()]) + "\nAnswer with the option's letter from the given choices directly."
    # print(f"Question: {question}")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question}
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            max_tokens=5,
            temperature=0.2,
            seed=42,
        )
        response_text = response.choices[0].message.content
        
        # Extract predicted letter from response
        predicted_letter = next((key for key in choice_map.keys() if key in response_text), "Unknown")
        predicted_label = choice_map.get(predicted_letter, "Unknown")
    except Exception as e:
        print(f"Error in generating response: {str(e)}")
        predicted_letter = "Error"
        predicted_label = "Error"
        response_text = "Error"

    return predicted_letter, predicted_label, response_text

def test_gpt(json_file, output_file, max_workers=10, batch_size=50):
    """
    Process test data with parallel execution and batching.
    
    Args:
        json_file: Path to the input JSON file
        output_file: Path to save results
        max_workers: Maximum number of parallel workers
        batch_size: Number of entries to process in a batch before saving
    """
    # Load test data
    with open(json_file, "r") as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]
    
    results = []
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process in batches to reduce memory usage and save progress frequently
    for batch_start in range(0, len(test_data), batch_size):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch = test_data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(test_data) + batch_size - 1)//batch_size}")
        
        # Create a client for each worker to avoid connection issues
        process_func = partial(process_entry, client=client)
        
        # Process entries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(tqdm(
                executor.map(process_func, batch),
                total=len(batch),
                desc=f"Batch {batch_start//batch_size + 1}"
            ))
        
        # Filter out None results (failed entries)
        batch_results = [r for r in batch_results if r is not None]
        results.extend(batch_results)
        
        # Save progress after each batch
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Progress saved: {len(results)}/{len(test_data)} entries processed")
    
    print(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
    seed = 42
    random.seed(seed)

    

    parser = argparse.ArgumentParser(description='Process cub classification data')
    parser.add_argument('--test_set', type=str, 
                    default=None,
                    help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str,
                    default=None,
                    help='Path to output JSON file')
    args = parser.parse_args()

    json_file = args.test_set
    output_file = args.output_file
    
    # Run with parallel processing and batching
    test_gpt(
        json_file=json_file,
        output_file=output_file,
        max_workers=10,  # Adjust based on your system's capabilities
        batch_size=50    # Adjust based on memory constraints
    )