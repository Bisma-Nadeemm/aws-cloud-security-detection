# AWS Cloud Security Detection: A Comparative Evaluation of Rule-Based and Machine Learning Methods

Three small, mostly-reproducible experiments in AWS-style cloud security
detection, built with scikit-learn. Every number in `results/` is a real
output of running this code -- nothing is invented or hand-typed. This
includes a genuine adversarial evaluation (Experiment 2) and a genuine
real-world external validation (Experiment 1) against real, live
AWS-managed IAM policies -- both are actually computed, not asserted in
prose.

## Authors

Bisma Nadeem, Noor-ul-Ain -- Department of Computer Science (PUCIT),
University of the Punjab.

## Experiments

| # | Task | Method | Key result (real, reproducible) |
|---|------|--------|----------------------------------|
| 1 | IAM misconfiguration detection | Rule-based static analysis + Random Forest on engineered policy features | Synthetic test: RF beats rules, 98.3% vs 75% accuracy. **Real-world test (19 genuine AWS-managed policies): RF collapses to 42.1% accuracy, worse than the 84.2% rule-based baseline** -- traced to one overly narrow feature; fixed, recovering RF to 84.2% accuracy / 87.5% recall. |
| 2 | API misuse / behavioral detection | Sequence feature engineering + Random Forest | Standard test: 99.4% accuracy, 100% recall, 0.74% FPR. **Adversarial (low-and-slow) test: baseline recall drops to 3.0%**; recovers to 100% only after training on evasive examples. |
| 3 | CloudTrail anomaly detection | Isolation Forest on engineered event features | Flagged 3% of events; 97.4% of flagged events were genuine anomalies. External grounding check against a real, incident-linked event catalog shows the "rare event" feature is a weaker real-world signal than assumed (see Limitations). |

## The two negative results (read before citing this repo)

The headline of this project isn't the accuracy numbers -- it's what
happened when we tested past the synthetic data the models were built on.

**Experiment 1: real-world transfer failure, traced and fixed.** The
Random Forest trained on synthetic IAM policies reaches 98.3% accuracy on
synthetic held-out data, beating the 75% rule-based baseline by a wide
margin. Tested against 19 real, currently-live AWS-managed IAM policies
(sourced from Cloudsplaining's open-source test corpus -- not synthetic,
not written by us), the same model's accuracy collapses to 42.1%, *worse*
than the simple rule scanner (84.2%). The cause is mechanistic: one
feature (`num_dangerous_actions`) checked for three hardcoded action
strings that happened to match the synthetic generator's vocabulary but
missed the dozens of other real AWS service wildcards. Generalizing that
one feature (no new information, just a broader pattern match) recovers
real-world accuracy to 84.2% and recall to 87.5%, with zero change to
synthetic-data performance. Full numbers: `results/exp1_realworld_results.json`.

**Experiment 2: adversarial evasion, traced and fixed.** An earlier draft
claimed a specific adversarial-evasion result (recall degrading from 100%
to 62.5%) that was never actually computed by any code. The real number is
worse and more interesting: baseline recall against a low-and-slow
attacker is 3.0%, not 62.5%. Widening the detection window alone does
nothing (still 3.0%); training on a small set of evasive examples fixes it
(100%). Full numbers: `results/exp2_results.json`.

Both fixes are implemented in this repo, not just described.

## Reproduce it yourself

```bash
pip install -r requirements.txt
python3 src/exp1_iam_analyzer.py
python3 src/exp2_api_misuse.py
python3 src/exp3_cloudtrail_anomaly.py

# Real-world validation (Experiment 1) -- requires the Cloudsplaining
# test fixture. Clone it once:
git clone --depth 1 https://github.com/salesforce/cloudsplaining.git realworld_repo
python3 src/exp1_realworld_validation.py

# Figures (all generated from real run output, not simulated data)
mkdir -p figures
python3 src/make_figure2.py   # Figure 3: real Isolation Forest scatter (Experiment 3)
python3 src/make_figure3.py   # Figure 2: real window-size sweep (Experiment 2)
```

Each script is self-contained and seeded, and prints its results as JSON.
Saved outputs are in `results/`. Figures are saved as PDF (for the LaTeX
build) in `figures/`; PNG copies for the Markdown version are also included
there.

## What's synthetic, what's real, and why that distinction is called out explicitly

- Experiments 2 and 3 train and evaluate entirely on synthetically
  generated data (standard practice here, since no public labeled dataset
  of real attacker sessions or real CloudTrail logs exists for privacy
  reasons -- see `VERIFIED_REFERENCES.md` entry 9 for a real published
  precedent for this approach).
- Experiment 1 trains on synthetic data but is **additionally validated
  against real data**: 19 real, currently-published AWS-managed IAM
  policy documents (see "Real-world data sources" in
  `VERIFIED_REFERENCES.md`).
- Experiment 3 includes a smaller **grounding check** (not a full
  real-data test) against a real, incident-linked catalog of CloudTrail
  event names, which surfaced a real weakness in the feature design
  rather than confirming it.

## Paper

- `paper_final.md` -- source of truth, plain Markdown.
- `paper_final.tex` / `paper_final.pdf` -- Overleaf-ready LaTeX build of
  the same content (compiles standalone with the base `article` class;
  swap in the official Springer Nature `sn-jnl` class from Overleaf's
  template gallery for final SN Computer Science formatting). Now uses
  `\usepackage{tikz}` for Figure 1 and `\includegraphics` for Figures 2-3
  -- if pasting into Overleaf's Springer template, make sure `tikz` is
  available (it is, by default) and that `figures/*.pdf` are uploaded
  alongside `main.tex`.
- Three figures, all generated from real run output (see
  `src/make_figure2.py`, `src/make_figure3.py` -- neither uses simulated
  or approximated data): an architecture/pipeline diagram, a real
  Isolation Forest scatter plot (Experiment 3), and a real window-size
  sweep (Experiment 2).

## Citations

See `VERIFIED_REFERENCES.md` -- 15 references, every one individually
confirmed to exist via search, including the two foundational method
papers (Breiman 2001 for Random Forest, Liu et al. 2008 for Isolation
Forest) and the two open-source projects used as actual real-world data
sources.
