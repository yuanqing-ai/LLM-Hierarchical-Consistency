import json
import torch
import random
import argparse
from PIL import Image
from tqdm import tqdm
import open_clip

def load_openclip_model(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device='cuda'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device).eval()
    return model, preprocess, tokenizer

def get_taxonomy_descriptions_single_style(choices, level_number):
    template = "a photo of a {}."
    return [template.format(choice) for choice in choices]


def infer_level_openclip(model, tokenizer, preprocess, image_path, descriptions, choice_map, device):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text_tokens = tokenizer(descriptions).to(device)

        with torch.no_grad(), torch.autocast(device_type=device):
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred_index = logits.argmax(dim=-1).item()
        predicted_label = list(choice_map.values())[pred_index]
        predicted_letter = list(choice_map.keys())[pred_index]

    except Exception as e:
        print(f"Error in OpenCLIP inference: {str(e)}")
        predicted_letter = "Unknown"
        predicted_label = "Unknown"

    return predicted_letter, predicted_label


def test_openclip(json_file, output_path, model, preprocess, tokenizer, device):
    with open(json_file, "r") as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]

    results = []
    for entry in tqdm(test_data, desc=f"testing"):
        image_path = entry["image"]
        label = entry["label"]

        level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()], key=lambda x: int(x[5:]))
        choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()], key=lambda x: int(x[13:]))

        if not level_keys or not choices_keys:
            print(f"Warning: Missing taxonomy data for {image_path}")
            continue

        result_entry = {"image": image_path, "label": label}

        for level_key, choices_key in zip(level_keys, choices_keys):
            level_number = level_key[5:]
            ground_truth = entry[level_key]
            choices = entry[choices_key]

            if ground_truth is None:
                continue

            choice_map = {chr(65 + i): opt for i, opt in enumerate(choices)}
            descriptions = get_taxonomy_descriptions_single_style(choices, level_number)

            predicted_letter, predicted_label = infer_level_openclip(
                model, tokenizer, preprocess, image_path, descriptions, choice_map, device
            )

            result_entry[f"ground_truth_level{level_number}"] = ground_truth
            result_entry[f"predicted_level{level_number}_letter"] = predicted_letter
            result_entry[f"predicted_level{level_number}"] = predicted_label
            result_entry[f"choices_level{level_number}"] = choice_map

        results.append(result_entry)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    seed=42
    random.seed(seed)

    parser = argparse.ArgumentParser(description="OpenCLIP hierarchical classification")
    parser.add_argument("--test_set", type=str,default="/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Inat_code/Animalia_with_similar_choice.jsonl")
    parser.add_argument("--output_file", type=str, default="/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/CLIP_Style/OpenCLIP/animal/animal_hierarchy_prompt_openclip_new.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    model, preprocess, tokenizer = load_openclip_model(device=args.device)

    test_openclip(args.test_set, args.output_file, model, preprocess, tokenizer, args.device)
