"""
Experiment 3: CloudTrail Anomaly Detection
Generates synthetic CloudTrail-style event records (normal admin/user
activity vs anomalous behavior: unusual hours, rare source IP/regions,
atypical event names) and detects anomalies using Isolation Forest.

All numbers are REAL outputs of running this code.
"""

import json
import random
import numpy as np
from sklearn.ensemble import IsolationForest

random.seed(42)
np.random.seed(42)

COMMON_EVENTS = ["GetObject", "DescribeInstances", "PutLogEvents", "ListBuckets",
                  "GetItem", "AssumeRole", "InvokeFunction"]
RARE_EVENTS = ["DeleteTrail", "StopLogging", "PutBucketAcl", "CreateAccessKey",
                "DeleteRole", "DisableKey"]
COMMON_REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]
RARE_REGIONS = ["ap-southeast-3", "af-south-1", "sa-east-1"]


def gen_event(anomalous=False):
    if not anomalous:
        return {
            "hour": np.random.normal(14, 3) % 24,           # business hours cluster
            "event_rarity": 1 if random.choice(COMMON_EVENTS) in RARE_EVENTS else 0,
            "region_rarity": 0 if random.random() > 0.05 else 1,
            "error_flag": 0 if random.random() > 0.03 else 1,
            "calls_per_minute": np.random.normal(5, 2),
        }
    else:
        return {
            "hour": np.random.choice([2, 3, 4, 23, 1]),      # off-hours
            "event_rarity": 1 if random.random() < 0.7 else 0,
            "region_rarity": 1 if random.random() < 0.5 else 0,
            "error_flag": 1 if random.random() < 0.4 else 0,
            "calls_per_minute": np.random.normal(25, 8),      # burst
        }


def build_dataset(n_normal=48500, n_anomalous=1500):
    rows, labels = [], []
    for _ in range(n_normal):
        rows.append(gen_event(False)); labels.append(0)
    for _ in range(n_anomalous):
        rows.append(gen_event(True)); labels.append(1)
    return rows, labels


def main():
    rows, labels = build_dataset()
    X = np.array([[r["hour"], r["event_rarity"], r["region_rarity"],
                   r["error_flag"], r["calls_per_minute"]] for r in rows])
    y = np.array(labels)

    contamination = y.mean()  # true anomaly rate, ~3% - matches paper's framing
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    model.fit(X)
    raw_preds = model.predict(X)          # -1 = anomaly, 1 = normal
    preds = (raw_preds == -1).astype(int)

    flagged = preds.sum()
    flagged_rate = flagged / len(X)
    true_anomalies_flagged = int(((preds == 1) & (y == 1)).sum())
    false_positives_flagged = int(((preds == 1) & (y == 0)).sum())
    genuine_pct = true_anomalies_flagged / flagged if flagged else 0
    noise_pct = false_positives_flagged / flagged if flagged else 0
    recall = true_anomalies_flagged / y.sum()

    results = {
        "dataset_size": len(rows),
        "true_anomaly_count": int(y.sum()),
        "flagged_count": int(flagged),
        "flagged_rate": round(flagged_rate, 4),
        "genuine_anomalies_pct_of_flagged": round(genuine_pct, 4),
        "noise_outliers_pct_of_flagged": round(noise_pct, 4),
        "recall_on_true_anomalies": round(recall, 4),
    }
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
