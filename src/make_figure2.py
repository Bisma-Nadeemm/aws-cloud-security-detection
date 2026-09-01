"""
Generates Figure 2 from the REAL, actual output of exp3_cloudtrail_anomaly.py
-- no simulated/approximated data. Re-runs the exact same pipeline (same
seeds), captures the Isolation Forest's real predictions, and plots a
random subsample of the real points (subsampled only for plot readability,
not to change the underlying result).
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import exp3_cloudtrail_anomaly as m

rows, labels = m.build_dataset()
X = np.array([[r["hour"], r["event_rarity"], r["region_rarity"],
               r["error_flag"], r["calls_per_minute"]] for r in rows])
y = np.array(labels)

contamination = y.mean()
model = m.IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
model.fit(X)
raw_preds = model.predict(X)
preds = (raw_preds == -1).astype(int)

print("Real flagged count:", preds.sum(), "/ true anomalies:", y.sum())
print("Real recall:", ((preds == 1) & (y == 1)).sum() / y.sum())

# Subsample for a readable plot: all flagged points + a random sample of
# unflagged points (real dataset is 50,000 points, too dense to render
# individually and be readable in a print figure)
rng = np.random.RandomState(42)
flagged_idx = np.where(preds == 1)[0]
normal_idx = np.where(preds == 0)[0]
normal_sample_idx = rng.choice(normal_idx, size=1200, replace=False)
plot_idx = np.concatenate([flagged_idx, normal_sample_idx])

hours = X[plot_idx, 0]
cpm = X[plot_idx, 4]
flagged_plot = preds[plot_idx]

plt.figure(figsize=(6.2, 3.8))
plt.scatter(hours[flagged_plot == 0], cpm[flagged_plot == 0],
            color="#b0b0b0", alpha=0.5, s=14, label=f"Not flagged (n={int((flagged_plot==0).sum())}, sampled)")
plt.scatter(hours[flagged_plot == 1], cpm[flagged_plot == 1],
            color="#c0392b", edgecolor="black", linewidth=0.3, s=22,
            label=f"Flagged by Isolation Forest (n={int((flagged_plot==1).sum())}, all)")
plt.xlabel("Hour of day (0-24)")
plt.ylabel("API calls per minute")
plt.title("Experiment 3: Isolation Forest flags, real model output")
plt.legend(fontsize=8, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/exp3_isolation_forest.pdf")
print("saved figures/exp3_isolation_forest.pdf")
