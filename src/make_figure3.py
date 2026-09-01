"""
Generates Figure 3: a REAL sweep of burst-detection window size vs. recall,
for both the standard test set and the low-and-slow adversarial test,
with and without adversarial training. Every point is an actually-trained
and actually-evaluated Random Forest at that window size -- not an
illustrative flat line.
"""
import sys
sys.path.insert(0, "src")
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

import exp2_api_misuse as m

WINDOW_SIZES = [4, 6, 8, 10, 15, 20, 30, 40]


def features_at_window(session, w):
    from collections import Counter
    counts = Counter(session)
    length = len(session)
    sensitive_count = sum(counts[a] for a in m.SENSITIVE_ACTIONS)
    sensitive_ratio = sensitive_count / length
    unique_ratio = len(counts) / length
    burst = m._max_burst(session, w)
    bigrams = [tuple(session[i:i + 2]) for i in range(length - 1)]
    bigram_counts = Counter(bigrams)
    most_common_bigram_freq = bigram_counts.most_common(1)[0][1] / max(1, len(bigrams))
    return [length, sensitive_count, sensitive_ratio, unique_ratio, burst, most_common_bigram_freq]


def run_sweep(include_adv_training):
    sessions, labels = m.build_dataset()
    if include_adv_training:
        for t in range(100):
            rng = random.Random(5000 + t)
            n_sens = rng.choice([5, 6, 7, 8])
            total_len = rng.choice([40, 50, 60])
            sessions.append(m.gen_lowslow_session(total_len, n_sens, rng=rng))
            labels.append(1)

    standard_recalls, adversarial_recalls = [], []
    for w in WINDOW_SIZES:
        X = np.array([features_at_window(s, w) for s in sessions])
        y = np.array(labels)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        standard_recalls.append(recall_score(y_te, preds, zero_division=0))

        detected = 0
        trials = 200
        for t in range(trials):
            rng = random.Random(1000 + t)
            adv_session = m.gen_lowslow_session(50, 6, rng=rng)
            pred = clf.predict([features_at_window(adv_session, w)])[0]
            detected += int(pred)
        adversarial_recalls.append(detected / trials)
    return standard_recalls, adversarial_recalls


print("Running sweep WITHOUT adversarial training examples...")
std_no_adv, adv_no_adv = run_sweep(include_adv_training=False)
print("window sizes:", WINDOW_SIZES)
print("standard recall (no adv training):", std_no_adv)
print("adversarial recall (no adv training):", adv_no_adv)

print("Running sweep WITH adversarial training examples...")
std_with_adv, adv_with_adv = run_sweep(include_adv_training=True)
print("standard recall (with adv training):", std_with_adv)
print("adversarial recall (with adv training):", adv_with_adv)

out = {
    "window_sizes": WINDOW_SIZES,
    "no_adversarial_training": {"standard_recall": std_no_adv, "adversarial_recall": adv_no_adv},
    "with_adversarial_training": {"standard_recall": std_with_adv, "adversarial_recall": adv_with_adv},
}
with open("results/exp2_window_sweep.json", "w") as f:
    json.dump(out, f, indent=2)

plt.figure(figsize=(6.5, 4))
plt.plot(WINDOW_SIZES, adv_no_adv, marker="o", color="#c0392b",
         label="Adversarial recall, no adversarial training")
plt.plot(WINDOW_SIZES, adv_with_adv, marker="s", color="#27ae60",
         label="Adversarial recall, with adversarial training")
plt.plot(WINDOW_SIZES, std_no_adv, marker="^", color="#7f8c8d", linestyle="--",
         label="Standard-test recall, no adversarial training")
plt.xlabel("Burst-detection window size (w)")
plt.ylabel("Recall")
plt.ylim(-0.05, 1.05)
plt.title("Experiment 2: recall vs. window size (real, measured)")
plt.legend(fontsize=8, loc="center right")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/exp2_window_sweep.pdf")
print("saved figures/exp2_window_sweep.pdf")
