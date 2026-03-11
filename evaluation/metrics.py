# evaluate the metrics and store metrics for this user
# metrics include precision, recall, and f-score at the specified p-value
def evaluate_metrics(ranked_predictions, actual_items, p):
    # Ensure actual_items is a set for set operations
    actual_items = set(actual_items)

    # Get the top p% of predictions
    top_p_count = max(1, int(len(actual_items) * p))

    top_p_predictions = set(ranked_predictions[:top_p_count])

    # Calculate true positives, false positives, and false negatives
    true_positives = len(top_p_predictions.intersection(actual_items))
    false_positives = len(top_p_predictions.difference(actual_items))
    false_negatives = len(actual_items.difference(top_p_predictions))

    # Safe precision and recall calculations (avoid division by zero)
    precision_denom = true_positives + false_positives
    recall_denom = true_positives + false_negatives

    precision = (true_positives / precision_denom) if precision_denom > 0 else 0.0
    recall = (true_positives / recall_denom) if recall_denom > 0 else 0.0

    # F0.1: guard denominator as well
    beta_sq = 0.1**2
    f_denom = (beta_sq * precision) + recall
    f_0_1 = ((1 + beta_sq) * (precision * recall) / f_denom) if f_denom > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f_0.1': f_0_1
    }