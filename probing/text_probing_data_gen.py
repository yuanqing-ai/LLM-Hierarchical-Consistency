"""
Save averaged token embeddings from every hidden layer of Qwen2.5‑VL‑7B‑Instruct
for a taxonomy‑prompting task.

For each hierarchy level (kingdom, phylum, class, order, family, genus) we
construct prompts of the form
    "<species> belongs to the <level> <label>."
run them through the model, average the hidden states over the input tokens,
and collect the embeddings.

The script creates one directory per transformer layer (e.g. layer_00, layer_01,
...), and inside each directory saves six .pt files, one for each hierarchy
level (kingdom.pt, phylum.pt, ...).  Each .pt file contains a tensor of shape
(n_prompts, hidden_dim).
"""

import json
import os
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info  # provided by Qwen‑VL repo


def average_hidden_states(model, processor, prompt):
    """Return a list of averaged hidden states (one tensor per layer)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    # Build model input (text + possible vision fields)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1,           # we only need hidden states for the prompt
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # generated.hidden_states is a tuple: (layers, seq_len, batch, hidden)
    # Take mean over the sequence dimension and squeeze batch → (hidden,)
    return [layer.mean(dim=1).squeeze(0) for layer in generated.hidden_states[0]]


def main(taxonomy_path,output_root):
    model_path = "/projectnb/ivc-ml/yuan/model_zoo/Qwen2.5-VL-7B-Instruct"

    # ── Load model & processor ────────────────────────────────────────────────
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # ── Load taxonomy ─────────────────────────────────────────────────────────
    with open(taxonomy_path, "r") as fp:
        taxonomy = json.load(fp)

    hierarchy_levels = ["kingdom", "phylum", "class", "order", "family", "genus"]
    n_layers = model.config.num_hidden_layers

    # storage[layer][level] → list[Tensor]
    storage = {layer: {lvl: [] for lvl in hierarchy_levels} for layer in range(n_layers+1)}

    # ── Iterate over species & prompts ───────────────────────────────────────
    for species, lineage in tqdm(taxonomy.items(), desc="Processing species"):
        for lvl, label in zip(hierarchy_levels, lineage[:-1]):  # exclude species itself
            # prompt = f"{species} belongs to the {lvl} {label}."
            # prompt = f"Given the {species}, what is its taxonomic classification at the {lvl} level?"
            prompt = f"Given the {species}, what is its taxonomic classification at the {lvl} level? It belongs to {label}."
            # print(f"Prompt: {prompt}")
            layer_embs = average_hidden_states(model, processor, prompt)
            for li, emb in enumerate(layer_embs):
                storage[li][lvl].append(emb.cpu())

    # ── Save embeddings ──────────────────────────────────────────────────────
    os.makedirs(output_root, exist_ok=True)
    for li, lvl_dict in storage.items():
        layer_dir = os.path.join(output_root, f"layer_{li:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        for lvl, vectors in lvl_dict.items():
            if vectors:  # should always be true, but just in case
                tensor = torch.stack(vectors)  # (N, hidden_dim)
                torch.save(tensor, os.path.join(layer_dir, f"{lvl}.pt"))

    print(f"Saved embeddings for {n_layers} layers under '{output_root}/'")


if __name__ == "__main__":
    taxonomy_path = "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/Inaturalist/test_taxonomy.json"
    output_root = "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Inat_code/new_with_gt/test"  # change if needed
    main(taxonomy_path,output_root)
