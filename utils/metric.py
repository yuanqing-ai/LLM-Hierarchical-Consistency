import json
import ast
import argparse

def normalize_case(hierarchy_list):
    """ Convert all elements in the hierarchical list to lowercase for case-insensitive comparison """
    return [item.lower() if item is not None else None for item in hierarchy_list]


def compute_strict_por(y_pred_list, y_true_list, ignore_first_layer=True):
    """
    Compute a stricter version of the Point-Overlap Ratio (POR).
    This finds the longest common prefix (LCP) between predicted and ground truth hierarchies.
    """
    assert len(y_pred_list) == len(y_true_list), "The number of predictions and ground truths must match."

    por_total = 0
    N = len(y_pred_list)

    for i in range(N):
        # Skip the first layer if requested
        start_idx = 1 if ignore_first_layer and len(y_pred_list[i]) > 1 and len(y_true_list[i]) > 1 else 0
        
        pred_hierarchy = normalize_case(y_pred_list[i][start_idx:])
        true_hierarchy = normalize_case(y_true_list[i][start_idx:])

        lcp_length = 0
        for pred, true in zip(pred_hierarchy, true_hierarchy):
            if pred == true:
                lcp_length += 1
            else:
                break  

        q_ci = len(true_hierarchy)  
        por = lcp_length / q_ci if q_ci > 0 else 0
        por_total += por

    return (por_total / N) * 100


def compute_tor(y_pred_list, y_true_list, ignore_first_layer=True):
    """
    Compute the average Top-Overlap Ratio (TOR) for a batch of samples based on consecutive pairs (case insensitive).
    """
    assert len(y_pred_list) == len(y_true_list), "The number of predictions and ground truths must match."
    por_total = 0
    N = len(y_pred_list)
    
    for i in range(N):
        # Skip the first layer if requested
        start_idx = 1 if ignore_first_layer and len(y_pred_list[i]) > 1 and len(y_true_list[i]) > 1 else 0
        
        pred_hierarchy = normalize_case(y_pred_list[i][start_idx:])
        true_hierarchy = normalize_case(y_true_list[i][start_idx:])
        
        # Count consecutive pair matches
        consecutive_matches = 0
        for j in range(len(true_hierarchy) - 1):
            if j + 1 < len(pred_hierarchy):  # Ensure we don't go out of bounds with predictions
                if (pred_hierarchy[j] == true_hierarchy[j] and 
                    pred_hierarchy[j+1] == true_hierarchy[j+1]):
                    consecutive_matches += 1
        
        # Calculate POR using only consecutive pairs
        total_possible = len(true_hierarchy) - 1  # Total possible consecutive pairs
        por = consecutive_matches / total_possible if total_possible > 0 else 0
        por_total += por
    
    return (por_total / N) * 100


def compute_por(y_pred_list, y_true_list, ignore_first_layer=True):
    """
    Compute the average Point-Overlap Ratio (POR) for a batch of samples (case insensitive).
    """
    assert len(y_pred_list) == len(y_true_list), "The number of predictions and ground truths must match."

    por_total = 0
    N = len(y_pred_list)

    for i in range(N):
        # Skip the first layer if requested
        start_idx = 1 if ignore_first_layer and len(y_pred_list[i]) > 1 and len(y_true_list[i]) > 1 else 0
        
        pred_hierarchy = normalize_case(y_pred_list[i][start_idx:])
        true_hierarchy = normalize_case(y_true_list[i][start_idx:])

        match_count = sum(1 for pred, true in zip(pred_hierarchy, true_hierarchy) if pred == true)
        por = match_count / len(true_hierarchy) if len(true_hierarchy) > 0 else 0
        por_total += por

    return (por_total / N) * 100 

def compute_layer_by_layer_accuracy(y_pred_list, y_true_list, ignore_first_layer=True):
    """
    Compute accuracy for each layer of the hierarchy separately (case insensitive).
    
    Args:
        y_pred_list: List of predicted hierarchies
        y_true_list: List of ground truth hierarchies
        ignore_first_layer: Whether to skip the first layer in the evaluation
    
    Returns:
        Dictionary with layer numbers as keys and their corresponding accuracy as values
    """
    assert len(y_pred_list) == len(y_true_list), "The number of predictions and ground truths must match."
    
    # Find the maximum depth of hierarchies
    max_depth = max(
        max([len(hier) for hier in y_pred_list]),
        max([len(hier) for hier in y_true_list])
    )
    
    # Initialize counters for each layer
    layer_correct = {i: 0 for i in range(1, max_depth + 1)}
    layer_total = {i: 0 for i in range(1, max_depth + 1)}
    
    # Determine the starting layer index based on ignore_first_layer
    start_idx = 1 if ignore_first_layer else 0
    
    for i in range(len(y_true_list)):
        pred_hierarchy = normalize_case(y_pred_list[i])
        true_hierarchy = normalize_case(y_true_list[i])
        
        # Compare each layer
        for j in range(start_idx, max_depth):
            layer_idx = j + 1  # Layer index (1-based for reporting)
            
            # Only count if both hierarchies have this layer
            if j < len(true_hierarchy):
                layer_total[layer_idx] += 1
                
                # Check if prediction has this layer and if it matches
                if j < len(pred_hierarchy) and pred_hierarchy[j] == true_hierarchy[j]:
                    layer_correct[layer_idx] += 1
    
    # Calculate accuracy for each layer
    layer_accuracy = {}
    for layer, total in layer_total.items():
        if total > 0:
            layer_accuracy[layer] = (layer_correct[layer] / total) * 100
        else:
            layer_accuracy[layer] = 0.0
    
    return layer_accuracy


def compute_hierarchy_consistency_accuracy(y_pred, y_true, ignore_first_layer=True):
    """
    Computes Hierarchy Consistent Accuracy (HCA) in a case-insensitive manner.
    """
    total_samples = len(y_true)
    correct_samples = 0
    
    for pred, true in zip(y_pred, y_true):
        # Skip the first layer if requested
        start_idx = 1 if ignore_first_layer and len(pred) > 1 and len(true) > 1 else 0
        
        pred_hierarchy = normalize_case(pred[start_idx:])
        true_hierarchy = normalize_case(true[start_idx:])
        
        if pred_hierarchy == true_hierarchy:
            correct_samples += 1

    return (correct_samples / total_samples) * 100

def por_f1(y_pred_list, y_true_list, ignore_first_layer=True):
    """
    Compute the Hierarchical F1 Score based on the Point-Overlap Ratio (POR).
    """
    assert len(y_pred_list) == len(y_true_list), "The number of predictions and ground truths must match."

    f1_total = 0
    N = len(y_pred_list)  # Number of samples

    for i in range(N):
        # Skip the first layer if requested
        start_idx = 1 if ignore_first_layer and len(y_pred_list[i]) > 1 and len(y_true_list[i]) > 1 else 0
        
        pred_hierarchy = normalize_case(y_pred_list[i][start_idx:])
        true_hierarchy = normalize_case(y_true_list[i][start_idx:])

        match_count = sum(1 for pred, true in zip(pred_hierarchy, true_hierarchy) if pred == true)

        precision = match_count / len(pred_hierarchy) if len(pred_hierarchy) > 0 else 0
        recall = match_count / len(true_hierarchy) if len(true_hierarchy) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f1_total += f1

    return (f1_total / N) * 100  # Convert to percentage


def hierarchical_f1(y_pred_hier, y_true_hier, ignore_first_layer=True):
    """
    Compute Example-level Hierarchical F1 Score (case insensitive).
    """
    assert len(y_pred_hier) == len(y_true_hier), "The number of predictions and ground truths must match."

    f1_total = 0
    N = len(y_pred_hier)  # Number of samples

    for i in range(N):
        # Skip the first layer if requested
        start_idx = 1 if ignore_first_layer and len(y_pred_hier[i]) > 1 and len(y_true_hier[i]) > 1 else 0
        
        pred_set = set(normalize_case(y_pred_hier[i][start_idx:]))
        true_set = set(normalize_case(y_true_hier[i][start_idx:]))

        intersection = pred_set & true_set
        precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0
        recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_total += f1

    return (f1_total / N) * 100


def compute_leaf_node_accuracy(y_pred_list, y_true_list, ignore_first_layer=True):
    """
    Compute the Leaf Node Accuracy (LNA) in a case-insensitive manner and return incorrect predictions.
    
    Args:
        y_pred_list: List of prediction paths
        y_true_list: List of ground truth paths
        ignore_first_layer: Not used in leaf node calculation as mentioned in the docstring
        
    Returns:
        tuple: (accuracy percentage, list of mismatched items)
    """
    assert len(y_pred_list) == len(y_true_list), "The number of predictions and ground truths must match."
    
    mismatched_items = []
    correct_leaf_count = 0
    
    for i, (y_pred, y_true) in enumerate(zip(y_pred_list, y_true_list)):
        pred_leaf = (y_pred[-1] or "").lower()
        true_leaf = (y_true[-1] or "").lower()
        
        if pred_leaf == true_leaf:
            correct_leaf_count += 1
        else:
            mismatched_items.append({
                'index': i,
                'prediction': y_pred,
                'ground_truth': y_true,
                'pred_leaf': y_pred[-1],
                'true_leaf': y_true[-1]
            })
    
    accuracy = (correct_leaf_count / len(y_true_list)) * 100
    return accuracy


def extract_hierarchies_from_results(results,cub=False):
    """
    Extract complete ground truth and predicted hierarchies from each sample.
    We extract the values from ground_truth_level1, ground_truth_level2, ... and
    predicted_level1, predicted_level2, ... to form complete paths.
    
    Args:
        results: List of dictionaries with result format
        
    Returns:
        Tuple of (ground_truth_hierarchies, predicted_hierarchies)
    """
    ground_truth_hierarchies = []
    predicted_hierarchies = []
    
    print(f"Total items in results: {len(results)}")
    
    for i, item in enumerate(results):
        # Extract the ground truth hierarchy (path from root to leaf)
        gt_hierarchy = []
        if cub:
            level = 1
        else:
            level = 2
        
        # Keep adding levels as long as they exist
        while f"ground_truth_level{level}" in item:
            value = item[f"ground_truth_level{level}"]
            if value is not None:  # Only include non-None values
                gt_hierarchy.append(value)
            level += 1
        
        # Extract the predicted hierarchy (path from root to leaf)
        pred_hierarchy = []
        if cub:
            level = 1
        else:
            level = 2
        
        # Keep adding levels as long as they exist
        while f"predicted_level{level}" in item:
            # Skip letter keys
            if f"predicted_level{level}_letter" in item:
                pass
            
            value = item[f"predicted_level{level}"]
            if value is not None:  # Only include non-None values
                pred_hierarchy.append(value)
            level += 1
        
        # Add to results if both hierarchies have values
        if gt_hierarchy and pred_hierarchy:
            ground_truth_hierarchies.append(gt_hierarchy)
            predicted_hierarchies.append(pred_hierarchy)
            
            # Print the first few samples for verification
            if i < 3:
                print(f"\nSample {i+1}:")
                print(f"  Ground Truth: {gt_hierarchy}")
                print(f"  Prediction:   {pred_hierarchy}")
    
    print(f"\nExtracted {len(ground_truth_hierarchies)} valid hierarchy pairs")
    return ground_truth_hierarchies, predicted_hierarchies


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract hierarchies from results')
    parser.add_argument('--cub', action='store_true', help='Use CUB dataset')
    parser.add_argument('--file_path', type=str, help='Path to the results file')
    args = parser.parse_args()



    file_path = args.file_path
    with open(file_path, "r") as f:
        data = json.load(f)
    # Extract hierarchies
    y_true, y_pred = extract_hierarchies_from_results(data, cub=args.cub)
    
    # Print sample hierarchies for verification
    if y_true and y_pred:
        print("\nSample hierarchies:")
        for i in range(min(3, len(y_true))):
            print(f"Sample {i+1}:")
            print(f"  Ground Truth: {y_true[i]}")
            print(f"  Prediction:   {y_pred[i]}")
            print(f"  Ground Truth (ignoring first layer): {y_true[i][1:]}")
            print(f"  Prediction (ignoring first layer):   {y_pred[i][1:]}")
        print()
  
    print("\n=== Modified Evaluation (Ignoring First Layer) ===")
    # Calculate metrics ignoring the first layer
    layer_by_layer_accuracy=compute_layer_by_layer_accuracy(y_pred, y_true, ignore_first_layer=False)
    tor_score_no_first = compute_tor(y_pred, y_true, ignore_first_layer=False)
    por_score_no_first = compute_por(y_pred, y_true, ignore_first_layer=False)
    strict_por_score_no_first = compute_strict_por(y_pred, y_true, ignore_first_layer=False)
    hca_score_no_first = compute_hierarchy_consistency_accuracy(y_pred, y_true, ignore_first_layer=False)
    hierarchical_f1_score_no_first = por_f1(y_pred, y_true, ignore_first_layer=False)
    leaf_accuracy_no_first = compute_leaf_node_accuracy(y_pred, y_true, ignore_first_layer=False)

    # Print modified results
    print(f"Top-Overlap Ratio (TOR): {tor_score_no_first:.4f}%")
    print(f"Point-Overlap Ratio (POR): {por_score_no_first:.4f}%")
    print(f"Strict Point-Overlap Ratio (POR): {strict_por_score_no_first:.4f}%")
    print(f"Hierarchy Consistency Accuracy (HCA): {hca_score_no_first:.4f}%")
    # print(f"Hierarchical F1 Score: {hierarchical_f1_score_no_first:.4f}%")
    print(f"Leaf Node Accuracy (LNA): {leaf_accuracy_no_first:.4f}%")   
    # print(f"Layer by Layer accuracy: {tor_score_no_first:.4f}%")


