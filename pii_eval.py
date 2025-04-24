import json
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
from collections import defaultdict


def evaluate_redactions(pred_spans: List[Dict], gold_spans: List[Dict], ignore_labels: bool = False) -> Tuple[int, int, int, Dict[str, Dict[str, int]]]:
    """
    Evaluate the performance of predicted PII spans against gold standard spans.
    
    Args:
        pred_spans: List of dictionaries containing predicted spans with 'start', 'end', and 'label' keys
        gold_spans: List of dictionaries containing gold standard spans with 'start', 'end', and 'label' keys
        ignore_labels: If True, only match spans based on start and end positions, ignoring labels
    
    Returns:
        Tuple containing (true_positives, false_positives, false_negatives) counts and per-label metrics
    """
    if ignore_labels:
        pred_set = {(s["start"], s["end"]) for s in pred_spans}
        gold_set = {(s["start"], s["end"]) for s in gold_spans}
    else:
        pred_set = {(s["start"], s["end"], s["label"]) for s in pred_spans}
        gold_set = {(s["start"], s["end"], s["label"]) for s in gold_spans}

    # Calculate overall metrics
    tp = len(pred_set & gold_set)  # Intersection of predicted and gold spans
    fp = len(pred_set - gold_set)  # Predicted spans that are not in gold spans
    fn = len(gold_set - pred_set)  # Gold spans that are not in predicted spans

    # Calculate per-label metrics
    label_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # Count true positives and false positives for each label
    for pred in pred_spans:
        label = pred["label"]
        pred_key = (pred["start"], pred["end"]) if ignore_labels else (pred["start"], pred["end"], pred["label"])
        gold_key = (pred["start"], pred["end"]) if ignore_labels else (pred["start"], pred["end"], pred["label"])
        
        if gold_key in gold_set:
            label_metrics[label]["tp"] += 1
        else:
            label_metrics[label]["fp"] += 1
    
    # Count false negatives for each label
    for gold in gold_spans:
        label = gold["label"]
        gold_key = (gold["start"], gold["end"]) if ignore_labels else (gold["start"], gold["end"], gold["label"])
        
        if gold_key not in pred_set:
            label_metrics[label]["fn"] += 1

    return tp, fp, fn, dict(label_metrics)


def evaluate_redaction_file(json_file_path: str, per_label: bool = False, ignore_labels: bool = False) -> Dict:
    """
    Evaluate redactions across all entries in a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing redaction results
        per_label: Whether to include per-label metrics in the results
        ignore_labels: If True, only match spans based on start and end positions, ignoring labels
    
    Returns:
        Dictionary containing precision, recall, and f1 scores calculated over all entries
        If per_label is True, also includes per-label metrics
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        redaction_results = json.load(f)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_label_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for entry in redaction_results:
        tp, fp, fn, label_metrics = evaluate_redactions(
            entry['pii_entities'],
            entry['original_privacy_mask'],
            ignore_labels=ignore_labels
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Aggregate per-label metrics
        for label, metrics in label_metrics.items():
            total_label_metrics[label]["tp"] += metrics["tp"]
            total_label_metrics[label]["fp"] += metrics["fp"]
            total_label_metrics[label]["fn"] += metrics["fn"]
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    results = {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    # Add per-label metrics if requested
    if per_label:
        results['per_label'] = {}
        for label, metrics in total_label_metrics.items():
            label_precision = metrics["tp"] / (metrics["tp"] + metrics["fp"] + 1e-9)
            label_recall = metrics["tp"] / (metrics["tp"] + metrics["fn"] + 1e-9)
            label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall + 1e-9)
            
            results['per_label'][label] = {
                'precision': label_precision,
                'recall': label_recall,
                'f1': label_f1,
                'counts': {
                    'true_positives': metrics["tp"],
                    'false_positives': metrics["fp"],
                    'false_negatives': metrics["fn"]
                }
            }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate PII redaction results')
    parser.add_argument('--file', type=str, default='pii_redaction_results_val.json',
                      help='Path to the JSON file containing redaction results')
    parser.add_argument('--per-label', action='store_true',
                      help='Include per-label metrics in the results')
    parser.add_argument('--ignore-labels', action='store_true',
                      help='Only match spans based on start and end positions, ignoring labels')
    
    args = parser.parse_args()
    
    results = evaluate_redaction_file(args.file, args.per_label, args.ignore_labels)
    
    print("\nOverall Metrics:")
    print(f"Precision: {results['overall']['precision']:.4f}")
    print(f"Recall: {results['overall']['recall']:.4f}")
    print(f"F1: {results['overall']['f1']:.4f}")
    
    if args.per_label and 'per_label' in results:
        print("\nPer-Label Metrics:")
        for label, metrics in results['per_label'].items():
            print(f"\n{label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Counts:")
            print(f"    True Positives: {metrics['counts']['true_positives']}")
            print(f"    False Positives: {metrics['counts']['false_positives']}")
            print(f"    False Negatives: {metrics['counts']['false_negatives']}")