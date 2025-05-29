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


def parse_args():
    parser = argparse.ArgumentParser(description='Feature probing experiment at different taxonomy levels')
    parser.add_argument('--dataset', type=str, default='Plantae', help='Dataset name')
    parser.add_argument('--train_hierarchy_path', type=str, default = "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/Inaturalist/train_taxonomy.json", help='Path to the training hierarchy JSON file')
    parser.add_argument('--test_hierarchy_path', type=str, default = "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/Inaturalist/test_taxonomy.json",  help='Path to the testing hierarchy JSON file')
    parser.add_argument('--train_feature_dir', type=str, default = "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Inat_code/new_with_gt/train", help='Directory containing training feature files for each level')
    parser.add_argument('--test_feature_dir', type=str, default = "/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Inat_code/new_with_gt/test",  help='Directory containing testing feature files for each level')
    parser.add_argument('--model_name', type=str, default='Qwen', help='Model name', choices=['Qwen', 'llava'])
    parser.add_argument('--probe', type=str, default='linear', help='Probe type (linear or mlp)')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--layer', type=str, default="01", help='Number of training epochs')
    parser.add_argument('--feature_type', type=str, default='avg', help='Feature type (last or avg)')
    parser.add_argument('--output_dir', type=str, default='/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/LLM_Probing/plant_results', help='Output directory for results')
    return parser.parse_args()


def normalize_label(label):
    """Normalize labels to handle naming inconsistencies"""
    return label.lower().replace("-", " ").replace("_", " ")


def load_features_for_level(feature_dir, level_name,layer, feature_type='avg'):
    """Load features for a specific taxonomy level"""
    feature_path = os.path.join(feature_dir,"layer_"+(layer), f"{level_name}.pt")
    if os.path.exists(feature_path):
        features = torch.load(feature_path, weights_only=True).float()
        print(f"Loaded {level_name} features with shape: {features.shape}")
        return features
    else:
        raise FileNotFoundError(f"Feature file not found: {feature_path}")


def prepare_data(train_hierarchy, test_hierarchy):
    """Prepare data for all taxonomic levels"""
    # Define hierarchy levels (excluding species)
    hierarchy_levels = {
        'kingdom': 0,
        'phylum': 1,
        'class': 2,
        'order': 3,
        'family': 4,
        'genus': 5
    }
    
    level_data = {}
    
    for level_name, level_idx in hierarchy_levels.items():
        train_data = []
        test_data = []
        all_classes = set()
        
        # Process training data
        for i, (species, hierarchy) in enumerate(train_hierarchy.items()):
            if level_idx < len(hierarchy):
                level_class = hierarchy[level_idx]
                all_classes.add(level_class)
                train_data.append({
                    'species': species,
                    'level_class': level_class,
                    'feature_idx': i
                })
        
        # Process testing data
        for i, (species, hierarchy) in enumerate(test_hierarchy.items()):
            if level_idx < len(hierarchy):
                level_class = hierarchy[level_idx]
                all_classes.add(level_class)
                test_data.append({
                    'species': species,
                    'level_class': level_class,
                    'feature_idx': i
                })
        
        # Convert classes to a sorted list for consistent indexing
        all_classes = sorted(list(all_classes))
        
        # Update data with class indices
        for item in train_data + test_data:
            item['class_idx'] = all_classes.index(item['level_class'])
        
        level_data[level_name] = {
            'train': train_data,
            'test': test_data,
            'classes': all_classes
        }
    
    return level_data


def run_probing_experiment(train_data, test_data, train_features, test_features, args, level_name, classes):
    """Run probing experiment for a specific hierarchy level"""
    print(f"Running probing for {level_name} level with {len(classes)} classes")
    print(f"Training set: {len(train_data)} species, Testing set: {len(test_data)} species")
    
    # Prepare features and labels
    train_indices = torch.tensor([item['feature_idx'] for item in train_data])
    train_features_subset = train_features[train_indices]
    train_labels = torch.tensor([item['class_idx'] for item in train_data])
    
    test_indices = torch.tensor([item['feature_idx'] for item in test_data])
    test_features_subset = test_features[test_indices]
    test_labels = torch.tensor([item['class_idx'] for item in test_data])
    
    # Create and train the model
    feature_dim = train_features_subset.shape[1]
    
    if args.probe == "linear":
        model = LinearProbing(feature_dim, len(classes)).cuda()
    elif args.probe == "mlp":
        model = MLPProbing(feature_dim, len(classes)).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    bsz = 512
    accs = []
    best_test_acc = 0
    res_best = None
    
    for epoch in trange(args.n_epochs):
        # Training
        model.train()
        for i in range(0, len(train_features_subset), bsz):
            optimizer.zero_grad()
            output = model(train_features_subset[i : i + bsz].cuda())
            loss = criterion(output, train_labels[i : i + bsz].cuda())
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_preds = []
            train_correct = []
            
            for i in range(0, len(train_features_subset), bsz):
                output = model(train_features_subset[i : i + bsz].cuda())
                pred = output.argmax(dim=1).cpu()
                train_preds.append(pred)
                batch_correct = (pred == train_labels[i : i + bsz]).int()
                train_correct.append(batch_correct)
            
            train_preds = torch.cat(train_preds)
            train_correct = torch.cat(train_correct)
            train_acc = train_correct.float().mean().item()
            
            # Test accuracy
            test_preds = []
            test_correct = []
            
            for i in range(0, len(test_features_subset), bsz):
                output = model(test_features_subset[i : i + bsz].cuda())
                pred = output.argmax(dim=1).cpu()
                test_preds.append(pred)
                batch_correct = (pred == test_labels[i : i + bsz]).int()
                test_correct.append(batch_correct)
            
            test_preds = torch.cat(test_preds)
            test_correct = torch.cat(test_correct)
            test_acc = test_correct.float().mean().item()
            
            accs.append((train_acc, test_acc))
            
            # Update best results
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                res_best = {
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_preds': train_preds.clone(),
                    'test_preds': test_preds.clone(),
                    'train_correct': train_correct.clone(),
                    'test_correct': test_correct.clone()
                }
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot([train_acc for train_acc, _ in accs], label="train")
    plt.plot([test_acc for _, test_acc in accs], label="test")
    plt.title(f"Accuracy for {level_name} classification")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_prefix = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.model_name}_{args.probe}_{args.feature_type}_{level_name}"
    )
    plt.savefig(f"{output_prefix}.png")
    plt.close()
    
    print(f"{level_name} best test accuracy: {best_test_acc:.4f} (epoch {res_best['epoch']})")
    
    return accs, model, res_best


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load hierarchy data
    print("Loading hierarchies...")
    with open(args.train_hierarchy_path) as f:
        train_hierarchy = json.load(f)
    
    with open(args.test_hierarchy_path) as f:
        test_hierarchy = json.load(f)
    
    print(f"Loaded {len(train_hierarchy)} species in training hierarchy")
    print(f"Loaded {len(test_hierarchy)} species in testing hierarchy")
    
    # Prepare data for all levels
    print("Preparing data for all taxonomic levels...")
    level_data = prepare_data(train_hierarchy, test_hierarchy)
    
    # Define hierarchy levels (excluding species)
    hierarchy_levels = {
        'kingdom': 0,
        'phylum': 1,
        'class': 2,
        'order': 3,
        'family': 4,
        'genus': 5
    }
    
    # Run probing for each level
    results = {}
    all_level_results = {}
    
    # Track correctness for each sample and level
    test_sample_count = len(level_data['genus']['test'])  # Use genus level to get test count
    level_correctness_matrix = np.zeros((len(hierarchy_levels), test_sample_count))
    
    for i, (level_name, level_idx) in enumerate(hierarchy_levels.items()):
        print(f"\n=== Running probing for {level_name} level ===")
        
        # Load features for this level
        train_features = load_features_for_level(args.train_feature_dir, level_name, args.layer, args.feature_type)
        test_features = load_features_for_level(args.test_feature_dir, level_name, args.layer, args.feature_type)
        
        level_info = level_data[level_name]
        train_data = level_info['train']
        test_data = level_info['test']
        classes = level_info['classes']
        
        # Run probing
        accs, model, res_best = run_probing_experiment(
            train_data,
            test_data,
            train_features,
            test_features,
            args, 
            level_name, 
            classes
        )
        
        # Store results
        all_level_results[level_name] = res_best
        
        # Update correctness matrix
        if 'test_correct' in res_best:
            level_correctness_matrix[i, :len(res_best['test_correct'])] = res_best['test_correct'].cpu().numpy()
        
        results[level_name] = {
            'accs': accs,
            'classes': classes,
            'best_acc': max([test_acc for _, test_acc in accs])
        }
    
    # Calculate all-layers-correct accuracy
    all_correct = np.all(level_correctness_matrix == 1, axis=0)
    all_layers_accuracy = np.sum(all_correct) / level_correctness_matrix.shape[1]
    
    print(f"\nAccuracy when all {len(hierarchy_levels)} layers are correct: {all_layers_accuracy:.4f} ({np.sum(all_correct)}/{level_correctness_matrix.shape[1]})")
    
    # Save the matrix and results
    output_prefix = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.model_name}_{args.probe}_{args.feature_type}_all_layers"
    )
    
    np.save(f"{output_prefix}_correctness_matrix.npy", level_correctness_matrix)
    
    with open(f"{output_prefix}_all_correct.txt", 'w') as f:
        f.write(f"All layers correct accuracy: {all_layers_accuracy:.4f} ({np.sum(all_correct)}/{level_correctness_matrix.shape[1]})\n")
    
    # Save individual accuracies
    with open(f"{output_prefix}_individual_accuracies.txt", 'w') as f:
        for level_name in hierarchy_levels.keys():
            if level_name in all_level_results:
                level_acc = all_level_results[level_name]['test_acc']
                f.write(f"{level_name} accuracy: {level_acc:.4f}\n")
    
    # Save summary
    summary_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.model_name}_{args.probe}_{args.feature_type}_summary.json"
    )
    
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
    
    # Plot comparison of different levels
    plt.figure(figsize=(12, 8))
    for level, info in results.items():
        test_accs = [acc for _, acc in info['accs']]
        plt.plot(test_accs, label=f"{level} ({len(info['classes'])} classes)")
    
    plt.title(f"Comparison of Test Accuracy across Taxonomy Levels")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.model_name}_{args.probe}_{args.feature_type}_comparison.png"
    ))
    plt.close()


if __name__ == "__main__":
    main()