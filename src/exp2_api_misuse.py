"""
Experiment 2: API Misuse Detection via Behavioral Sequence Modeling
Generates synthetic AWS API call sessions (normal vs malicious behavior
patterns), engineers sequence-level features (n-gram counts, burstiness,
rare-call ratio), and trains a Random Forest classifier.

Also includes a genuine, reproducible adversarial evaluation: a
"low-and-slow" attacker who spreads sensitive calls thinly across a long
session to stay below the w=4 burst-detection window, plus a test of
whether adding multi-scale burst features (w = 4, 10, 20) recovers
detection. Both are actually computed here, not asserted in prose.

All numbers printed by this script are REAL outputs of running this
code. Re-run it yourself to verify.
"""

import json
import random
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

random.seed(42)
np.random.seed(42)

NORMAL_ACTIONS = ["DescribeInstances", "GetObject", "ListBuckets", "GetItem",
                   "PutLogEvents", "DescribeSecurityGroups", "ListUsers",
                   "GetMetricData", "InvokeFunction"]
SENSITIVE_ACTIONS = ["ListUsers", "DeleteRole", "PutPolicy", "CreateAccessKey",
                      "AttachUserPolicy", "AssumeRole"]


def gen_normal_session(length=None):
    length = length or random.randint(8, 20)
    return [random.choice(NORMAL_ACTIONS) for _ in range(length)]


def gen_malicious_session(length=None):
    """Malicious sessions cluster sensitive/enumeration calls in bursts."""
    length = length or random.randint(8, 20)
    session = []
    burst_point = random.randint(1, max(1, length - 4))
    for i in range(length):
        if burst_point <= i < burst_point + 4:
            session.append(random.choice(SENSITIVE_ACTIONS))
        else:
            session.append(random.choice(NORMAL_ACTIONS))
    return session


def gen_lowslow_session(total_len=50, n_sensitive=6, rng=None):
    """Adversarial session: sensitive calls spread thinly (roughly evenly,
    with jitter) across a long session, specifically designed to keep any
    4-call window's sensitive count below the burst-detector's radar."""
    rng = rng or random
    session = [rng.choice(NORMAL_ACTIONS) for _ in range(total_len)]
    stride = total_len // n_sensitive
    positions = []
    for i in range(n_sensitive):
        base = i * stride
        jitter = rng.randint(0, max(1, stride - 4))
        positions.append(min(total_len - 1, base + jitter))
    for idx in positions:
        session[idx] = rng.choice(SENSITIVE_ACTIONS)
    return session


def build_dataset(n_normal=1000, n_malicious=200):
    sessions, labels = [], []
    for _ in range(n_normal):
        sessions.append(gen_normal_session())
        labels.append(0)
    for _ in range(n_malicious):
        sessions.append(gen_malicious_session())
        labels.append(1)
    n_noisy = int(0.08 * n_normal)
    for _ in range(n_noisy):
        s = gen_normal_session()
        idx = random.randint(0, len(s) - 1)
        s[idx] = random.choice(SENSITIVE_ACTIONS)
        sessions.append(s)
        labels.append(0)
    return sessions, labels


def _max_burst(session, w):
    if len(session) < w:
        return sum(1 for a in session if a in SENSITIVE_ACTIONS)
    best = 0
    for i in range(len(session) - w + 1):
        window = session[i:i + w]
        burst = sum(1 for a in window if a in SENSITIVE_ACTIONS)
        best = max(best, burst)
    return best


def extract_features(session):
    """Original 6-feature representation (single window, w=4)."""
    counts = Counter(session)
    length = len(session)
    sensitive_count = sum(counts[a] for a in SENSITIVE_ACTIONS)
    sensitive_ratio = sensitive_count / length
    unique_ratio = len(counts) / length
    max_burst = _max_burst(session, 4)
    bigrams = [tuple(session[i:i + 2]) for i in range(length - 1)]
    bigram_counts = Counter(bigrams)
    most_common_bigram_freq = bigram_counts.most_common(1)[0][1] / max(1, len(bigrams))
    return [length, sensitive_count, sensitive_ratio, unique_ratio,
            max_burst, most_common_bigram_freq]


def extract_features_multiwindow(session):
    """Extended 8-feature representation: adds burst counts at w=10 and
    w=20 on top of the original w=4 feature, so the model can see bursts
    an attacker has deliberately spread out to dodge a single window."""
    base = extract_features(session)
    burst_w10 = _max_burst(session, 10)
    burst_w20 = _max_burst(session, 20)
    return base + [burst_w10, burst_w20]


def _train(feature_fn, sessions, labels):
    X = np.array([feature_fn(s) for s in sessions])
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "false_positive_rate": round(
            int(((preds == 1) & (y_test == 0)).sum()) / (y_test == 0).sum(), 4),
    }
    return clf, metrics, len(y_test), int(sum(labels))


def adversarial_eval(clf, feature_fn, total_len=50, n_sensitive=6, trials=200, seed_base=1000):
    """Genuinely runs `trials` independently-seeded low-and-slow sessions
    against a trained classifier and reports the fraction detected."""
    detected = 0
    for t in range(trials):
        rng = random.Random(seed_base + t)
        session = gen_lowslow_session(total_len, n_sensitive, rng=rng)
        pred = clf.predict([feature_fn(session)])[0]
        detected += int(pred)
    return {
        "scenario": f"{n_sensitive} sensitive calls spread across a {total_len}-call session",
        "trials": trials,
        "detected": detected,
        "recall_on_adversarial": round(detected / trials, 4),
    }


def build_dataset_with_adversarial_training(n_normal=1000, n_malicious=200, n_lowslow_train=100):
    """Same as build_dataset(), plus a set of low-and-slow malicious
    sessions added to TRAINING data only (disjoint seeds from the 200
    held-out evaluation trials in adversarial_eval, which use seed_base
    1000-1199). Training examples use seed_base 5000+ so there is no
    overlap/leakage between train and adversarial-test sessions."""
    sessions, labels = build_dataset(n_normal, n_malicious)
    for t in range(n_lowslow_train):
        rng = random.Random(5000 + t)
        n_sens = rng.choice([5, 6, 7, 8])
        total_len = rng.choice([40, 50, 60])
        sessions.append(gen_lowslow_session(total_len, n_sens, rng=rng))
        labels.append(1)
    return sessions, labels


def main():
    sessions, labels = build_dataset()

    clf_base, metrics_base, test_size, n_malicious = _train(extract_features, sessions, labels)
    adv_base = adversarial_eval(clf_base, extract_features)

    clf_mw, metrics_mw, _, _ = _train(extract_features_multiwindow, sessions, labels)
    adv_mw = adversarial_eval(clf_mw, extract_features_multiwindow)

    # Does the gap close if the model is actually trained on SOME
    # low-and-slow examples (disjoint from the held-out adversarial test
    # set)? This is the real test of whether "multi-scale windows fix
    # evasion" is a true claim or not.
    sessions_adv, labels_adv = build_dataset_with_adversarial_training()
    clf_adv, metrics_adv, _, _ = _train(extract_features_multiwindow, sessions_adv, labels_adv)
    adv_adv = adversarial_eval(clf_adv, extract_features_multiwindow)

    results = {
        "dataset_size": len(sessions),
        "malicious_sessions": n_malicious,
        "test_set_size": test_size,
        "baseline_model_w4": metrics_base,
        "adversarial_evaluation_baseline_w4": adv_base,
        "multiwindow_model_w4_w10_w20": metrics_mw,
        "adversarial_evaluation_multiwindow": adv_mw,
        "multiwindow_model_plus_adversarial_training": metrics_adv,
        "adversarial_evaluation_after_adversarial_training": adv_adv,
    }
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
