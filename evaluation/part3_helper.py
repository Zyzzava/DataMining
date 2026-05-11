import numpy as np

def init_trace_tracker():
    """Initializes the structure to track rule activations."""
    return {
        'total_rules_suggested': 0,
        'total_cf_suggested': 0,
        'users_with_rules': 0,
        'users_total': 0
    }

def update_trace(tracker, final_recommendations, rule_predictions):
    """
    Tracks how many items in the final list came from rules vs CF.
    """
    rule_set = set(rule_predictions)
    # Count how many of our final top-N came from the rule generator
    rules_in_final = [t for t in final_recommendations if t in rule_set]
    
    tracker['total_rules_suggested'] += len(rules_in_final)
    tracker['total_cf_suggested'] += (len(final_recommendations) - len(rules_in_final))
    tracker['users_total'] += 1
    if len(rules_in_final) > 0:
        tracker['users_with_rules'] += 1

def log_trace_results(tracker, cluster_id):
    """Prints a summary of the distribution for the cluster."""
    total = tracker['total_rules_suggested'] + tracker['total_cf_suggested']
    rule_pct = (tracker['total_rules_suggested'] / total * 100) if total > 0 else 0
    cf_pct = (tracker['total_cf_suggested'] / total * 100) if total > 0 else 0
    
    print(f"\n--- Trace for Cluster {cluster_id} ---")
    print(f"Items from Rules: {tracker['total_rules_suggested']} ({rule_pct:.1f}%)")
    print(f"Items from CF:    {tracker['total_cf_suggested']} ({cf_pct:.1f}%)")
    print(f"Coverage: {tracker['users_with_rules']}/{tracker['users_total']} users received rule-based tips.")