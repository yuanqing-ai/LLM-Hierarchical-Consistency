import json
import torch
import random
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


def test_siglip(json_file, output_path, model_path, device, style_index):
    # Load model
    model = AutoModel.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    with open(json_file, "r") as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]

    results = []
    for entry in tqdm(test_data, desc=f"testing"):
        image_path = entry["image"]
        label = entry["label"]

        # Find taxonomy keys
        level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()], key=lambda x: int(x[5:]))
        choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()], key=lambda x: int(x[13:]))


        if not level_keys or not choices_keys:
            print(f"Warning: Missing taxonomy data for {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            continue

        result_entry = {"image": image_path, "label": label}

        for level_key, choices_key in zip(level_keys, choices_keys):
            level_number = level_key[5:]
            ground_truth = entry[level_key]
            choices = entry[choices_key]

            if ground_truth is None:
                continue

            choice_map = {chr(65 + i): opt for i, opt in enumerate(choices)}
            descriptions = get_taxonomy_descriptions_single_style(choices, level_number, style_index)

            predicted_letter, predicted_label = infer_level(model, processor, image, descriptions, choice_map, device)

            result_entry[f"ground_truth_level{level_number}"] = ground_truth
            result_entry[f"predicted_level{level_number}_letter"] = predicted_letter
            result_entry[f"predicted_level{level_number}"] = predicted_label
            result_entry[f"choices_level{level_number}"] = choice_map

        results.append(result_entry)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[Prompt {style_index}] Results saved to {output_path}")


def get_taxonomy_descriptions_single_style(choices, level_number):
    template = "a photo of a {}."
    return [template.format(choice) for choice in choices]


def infer_level(model, processor, image, descriptions, choice_map, device):
    """
    Run SigLIP inference using provided descriptions.
    Uses pred_index to directly map to label.
    """
    inputs = processor(text=descriptions, images=image, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_image
        pred_index = logits.argmax(dim=1).item()

        predicted_label = list(choice_map.values())[pred_index]
        predicted_letter = list(choice_map.keys())[pred_index]

    except Exception as e:
        print(f"Error in inference: {str(e)}")
        predicted_letter = "Unknown"
        predicted_label = "Unknown"

    return predicted_letter, predicted_label


if __name__ == "__main__":
    seed=42
    random.seed(seed)
    parser = argparse.ArgumentParser(description="SigLIP hierarchical classification")

    parser.add_argument("--test_set", type=str, default="/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Evaluate_CLS/data/ImageNet/imagenet_animal_with_similarity_choice.jsonl",
                        help="Path to the input JSONL file")

    parser.add_argument("--output_file", type=str,default="/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/CLIP_Style/SigLIP/ImageNet-Animal/Animal_hierarchy_prompt_ensemble_results.json",help="Path to save output JSON")

    parser.add_argument("--model_path", type=str,default="google/siglip-so400m-patch14-384",help="SigLIP model identifier or path")

    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda or cpu)")
    args = parser.parse_args()
    test_siglip(args.test_set, args.output_file, args.model_path, args.device)
