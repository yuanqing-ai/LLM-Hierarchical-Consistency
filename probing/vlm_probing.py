import numpy as np
import torch
import json
import math
import random
import argparse
from tqdm import trange
from matplotlib import pyplot as plt
import os
from collections import defaultdict


class LinearProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbing, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MLPProbing, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim, feature_dim)
        self.fc2 = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)



def extract_classes_from_data(data):
    """Extract unique classes from data"""
    classes = set()
    for item in data:
        classes.add(item["label"])
    return sorted(list(classes))


def process_hierarchy_data(data, hierarchy_data, level_idx):
    """Process data to get labels for a specific hierarchy level"""
    # Create a mapping from original label to hierarchy level label
    label_to_hierarchy = {}
    hierarchy_classes = []
    
    for item in data:
        original_label = item["label"]
        if original_label in hierarchy_data:
            hierarchy_label = hierarchy_data[original_label][level_idx]
            
            if hierarchy_label not in hierarchy_classes:
                hierarchy_classes.append(hierarchy_label)
            
            label_to_hierarchy[original_label] = hierarchy_label
    
    # Update data with hierarchy labels
    valid_data = []
    for item in data:
        if item["label"] in label_to_hierarchy:
            new_item = item.copy()
            new_item["hierarchy_label"] = label_to_hierarchy[item["label"]]
            new_item["hierarchy_index"] = hierarchy_classes.index(label_to_hierarchy[item["label"]])
            valid_data.append(new_item)
    
    return valid_data, hierarchy_classes


def run_probing(args, data, features, level_name, classes):
    """Run probing experiment for a specific hierarchy level"""
    # Prepare labels
    labels = []
    valid_indices = []
    
    for i, item in enumerate(data):
        labels.append(item["hierarchy_index"])
        valid_indices.append(i)
    
    labels = torch.tensor(labels)
    
    train_idxs = [i for i, item in enumerate(data) if item["split"] == "train"]
    test_idxs = [i for i, item in enumerate(data) if item["split"] == args.split]
    
    if not train_idxs or not test_idxs:
        print(f"Not enough data for {level_name}. Skipping.")
        return None, None, None

    # Get the appropriate indices from the original features tensor
    feature_indices = [valid_indices[i] for i in train_idxs + test_idxs]
    labels_indices = [i for i in range(len(train_idxs + test_idxs))]
    
    # Map to original features
    used_features = features[torch.tensor(feature_indices)]
    
    train_features = used_features[:len(train_idxs),:]
    test_features = used_features[len(train_idxs):, :]
    train_labels = labels[torch.tensor(train_idxs)]
    test_labels = labels[torch.tensor(test_idxs)]

    print(
        f"Level: {level_name}, Classes: {len(classes)}, "
        f"Train: {train_features.shape}, Test: {test_features.shape}"
    )

    if args.probe == "linear":
        model = LinearProbing(len(train_features[0]), len(classes)).cuda()
    elif args.probe == "mlp":
        model = MLPProbing(len(train_features[0]), len(classes)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    bsz = 512

    accs = []
    res_best = None  
    best_test_acc = 0  

    for epoch in trange(args.n_epochs):
        for i in range(0, len(train_features), bsz):
            optimizer.zero_grad()
            output = model(train_features[i : i + bsz].cuda())
            loss = criterion(output, train_labels[i : i + bsz].cuda())
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            eval_bsz = 512
            train_preds = []
            train_correct = [] 
            
            for i in range(0, len(train_features), eval_bsz):
                output = model(train_features[i : i + eval_bsz].cuda())
                pred = output.argmax(dim=1).cpu()
                train_preds.append(pred)
                # Store 1 for correct predictions, 0 for incorrect
                batch_correct = (pred == train_labels[i : i + eval_bsz]).int()
                train_correct.append(batch_correct)
                
            train_preds = torch.cat(train_preds)
            train_correct = torch.cat(train_correct)  # Binary tensor of correct (1) / incorrect (0)
            train_acc = train_correct.float().mean().item()
            
            test_preds = []
            test_correct = []  
            
            for i in range(0, len(test_features), eval_bsz):
                output = model(test_features[i : i + eval_bsz].cuda())
                pred = output.argmax(dim=1).cpu()
                test_preds.append(pred)
                # Store 1 for correct predictions, 0 for incorrect
                batch_correct = (pred == test_labels[i : i + eval_bsz]).int()
                test_correct.append(batch_correct)
                
            test_preds = torch.cat(test_preds)
            test_correct = torch.cat(test_correct)  # Binary tensor of correct (1) / incorrect (0)
            test_acc = test_correct.float().mean().item()
            
            accs.append((train_acc, test_acc))
            
            # Update best results if this epoch has better test accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                res_best = {
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_preds': train_preds.clone(),  # Store actual predictions
                    'test_preds': test_preds.clone(),    # Store actual predictions
                    'train_correct': train_correct.clone(),  # Store binary correct/incorrect
                    'test_correct': test_correct.clone()     # Store binary correct/incorrect
                }

    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot([train_acc for train_acc, _ in accs], label="train")
    plt.plot([test_acc for _, test_acc in accs], label="test")
    plt.title(f"Accuracy for {level_name} classification")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    output_prefix = os.path.join(
        args.output_path,
        f"{args.dataset}_{args.model_name}_{args.probe}_{args.split}_{args.feature_type}_{level_name}"
    )
    plt.savefig(f"{output_prefix}.png")
    plt.close()

    # Save best results
    # torch.save(res_best, f"{output_prefix}_best_results.pt")
    print(f"{level_name} best test accuracy: {best_test_acc:.4f} (epoch {res_best['epoch']})")
    
    return accs, model, res_best


def normalize_label(label):
    """Normalize labels to handle naming inconsistencies"""
    return label.lower().replace("-", " ").replace("_", " ")


def parse_args():
    parser = argparse.ArgumentParser(description='Feature probing experiment at different taxonomy levels')
    parser.add_argument('--dataset', type=str, default='CUB200', help='Dataset name',choices=['CUB200','Inat21-Plant'])
    parser.add_argument('--data', type=str, default=None, help='Data File')
    parser.add_argument('--taxonomy', type=str, default=None, help='taxonomy')
    parser.add_argument('--feature_path', type=str, default=None, help='Path to the feature file')
    parser.add_argument('--model_name', type=str, default='Qwen', help='Model name',choices=['Qwen','llava'])
    parser.add_argument('--probe', type=str, default='linear', help='Probe type (linear or mlp)')
    parser.add_argument('--split', type=str, default='test', help='Evaluation split')
    parser.add_argument('--feature_type', type=str, default='avg', help='Feature type (last or avg)')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--output_path', type=str, default=None, required=True, help='Output directory path')
    parser.add_argument('--hierarchy_level', type=str, default='all',  choices=['order', 'family', 'genus', 'species', 'all'],help='Taxonomic level to classify (order, family, genus, species, or all)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    try:
        with open(args.data, "r") as f:
            data = json.load(f)  # Try JSON format first
    except:
        with open(args.data, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]  # Try JSONL format
    
    original_classes = extract_classes_from_data(data)
    print(f"Extracted {len(original_classes)} classes from data")
    with open(args.taxonomy, "r") as f:
        hierarchy_data = json.load(f)
    normalized_hierarchy = {}
    for key, value in hierarchy_data.items():
        normalized_key = normalize_label(key)
        normalized_hierarchy[normalized_key] = value
    
    # Normalize labels in data for better matching
    for item in data:
        item["original_label"] = item["label"]
        item["normalized_label"] = normalize_label(item["label"])
    feature_path = args.feature_path
    if os.path.exists(feature_path):
        feature = torch.load(feature_path, weights_only=True).float()
    else:
        print(f"Warning: Feature file {feature_path} not found")
        return
 
    features = feature
    print(f"Features shape: {features.shape}")
    
    # Define hierarchy levels
    if args.dataset == 'Inat21-Plant':  # Fixed syntax error: added colon
        hierarchy_levels = {
            'kingdom': 0,
            'phylum': 1,
            'class': 2,
            'order': 1,
            'family': 2,
            'genus': 3,
            'species': 6
        }
    else:
        hierarchy_levels = {
            'order': 0,
            'family': 1,
            'genus': 2,
            'species': 3
        }
    # Run probing for each level or just the specified level
    results = {}
    
    if args.hierarchy_level == 'all':
        levels_to_run = hierarchy_levels.items()
    else:
        levels_to_run = [(args.hierarchy_level, hierarchy_levels[args.hierarchy_level])]

    all_level_results = {}
    layer_correctness_matrix = None
    n_samples = 0
    
    for level_name, level_idx in levels_to_run:
        print(f"\nRunning probing for {level_name} level...")
        # Match data with hierarchy using normalized labels
        matched_hierarchy = {}
        unmatched_items = []

        for item in data:
            if item["normalized_label"] in normalized_hierarchy:
                matched_hierarchy[item["original_label"]] = normalized_hierarchy[item["normalized_label"]]
            else:
                unmatched_items.append(item)
        print(f"Total of {len(unmatched_items)} unmatched items:")
        for item in unmatched_items:
            print(f"Not matched: {item['original_label']} (normalized label: {item['normalized_label']})")
       
        processed_data, level_classes = process_hierarchy_data(data, matched_hierarchy, level_idx)
        
        if not processed_data:
            print(f"No data could be processed for {level_name} level. Skipping.")
            continue
        
        accs, model, res_best = run_probing(args, processed_data, features, level_name, level_classes)

        if accs is None or res_best is None:  # Handle None returns from run_probing
            continue

        # Store level results
        all_level_results[level_name] = res_best
        
        # Initialize the layer correctness matrix if this is the first level
        if layer_correctness_matrix is None and 'test_correct' in res_best:
            n_samples = len(res_best['test_correct'])
            layer_correctness_matrix = np.zeros((len(levels_to_run), n_samples), dtype=np.int32)
        
        # Update the matrix with this level's results
        if 'test_correct' in res_best:
            current_level_index = [i for i, (name, _) in enumerate(levels_to_run) if name == level_name][0]
            # Convert PyTorch tensor to numpy if needed
            if torch.is_tensor(res_best['test_correct']):
                layer_correctness_matrix[current_level_index] = res_best['test_correct'].cpu().numpy()
            else:
                layer_correctness_matrix[current_level_index] = res_best['test_correct']

        if accs:
            results[level_name] = {
                'accs': accs,
                'classes': level_classes,
                'best_acc': max([test_acc for _, test_acc in accs])
            }
    
    # Fixed indentation error: moved summary_path inside proper scope
    summary_path = os.path.join(
        args.output_path,
        f"{args.dataset}_{args.model_name}_{args.probe}_{args.split}_{args.feature_type}_summary.json"
    )

    # Calculate the "all layers correct" accuracy
    if layer_correctness_matrix is not None:
        # For each sample, check if all layers predicted correctly
        all_correct = np.all(layer_correctness_matrix == 1, axis=0)
        all_layers_accuracy = np.sum(all_correct) / n_samples
        
        print(f"\nAccuracy when all {len(levels_to_run)} layers are correct: {all_layers_accuracy:.4f} ({np.sum(all_correct)}/{n_samples})")
        
        # Save the matrix and the all-correct results
        output_prefix = os.path.join(
            args.output_path,
            f"{args.dataset}_{args.model_name}_{args.probe}_{args.split}_{args.feature_type}_all_layers"
        )
        
        # Save the layer correctness matrix
        np.save(f"{output_prefix}_correctness_matrix.npy", layer_correctness_matrix)
        
        # Save the all-correct results
        with open(f"{output_prefix}_all_correct.txt", 'w') as f:
            f.write(f"All layers correct accuracy: {all_layers_accuracy:.4f} ({np.sum(all_correct)}/{n_samples})\n")
            
        # Also save the individual accuracies for reference
        with open(f"{output_prefix}_individual_accuracies.txt", 'w') as f:
            for i, (level_name, _) in enumerate(levels_to_run):
                if level_name in all_level_results:
                    level_acc = all_level_results[level_name]['test_acc']
                    f.write(f"{level_name} accuracy: {level_acc:.4f}\n")

    with open(summary_path, 'w') as f:
        summary = {
            'dataset': args.dataset,
            'model': args.model_name,
            'probe': args.probe,
            'feature_type': args.feature_type,
            'results': {
                level: {
                    'num_classes': len(info['classes']),
                    'best_accuracy': info['best_acc'],
                    'classes': info['classes']
                }
                for level, info in results.items()
            }
        }
        json.dump(summary, f, indent=2)
    
    print("\nSummary of results:")
    for level, info in results.items():
        print(f"{level.capitalize()} level ({len(info['classes'])} classes): {info['best_acc']:.4f}")
    
    # Plot comparison of different levels
    if len(results) > 1:
        plt.figure(figsize=(12, 8))
        for level, info in results.items():
            test_accs = [acc for _, acc in info['accs']]
            plt.plot(test_accs, label=f"{level} ({len(info['classes'])} classes)")
        
        plt.title(f"Comparison of Test Accuracy across Taxonomy Levels")
        plt.xlabel("Epochs")
        plt.ylabel("Test Accuracy")
        plt.legend()
        comparison_plot_path = os.path.join(
            args.output_path,
            f"{args.dataset}_{args.model_name}_{args.probe}_{args.split}_{args.feature_type}_comparison.png"
        )
        plt.savefig(comparison_plot_path)
        plt.close()


if __name__ == "__main__":
    main()