"""
Experiment 1 (external validation): tests the synthetic-data-trained rule
engine and Random Forest against REAL, live AWS-managed IAM policy
documents -- not synthetic data.

Source: 30 genuine AWS-managed policies (arn:aws:iam::aws:policy/...), taken
from Salesforce's open-source Cloudsplaining project's test corpus
(test/files/example-authz-details.json), which mirrors real
`iam:GetAccountAuthorizationDetails` output and includes the actual current
JSON policy documents AWS publishes and every AWS account can attach.
These are not written by us and not shaped by our synthetic data generator.

Source repository: https://github.com/salesforce/cloudsplaining
(Apache-2.0 licensed; policy documents themselves are AWS's own public
IAM policy definitions, not Cloudsplaining's IP.)

Ground truth is NOT invented by us. It follows AWS's own documented
security guidance (AWS IAM best practices explicitly discourage
'*FullAccess' and 'AdministratorAccess'-class managed policies in
production and recommend narrowly-scoped or ReadOnly/ViewOnly policies
instead): a policy is labeled "broad/risky" if its name ends in
"FullAccess" or is "AdministratorAccess"/"PowerUserAccess", and
"narrow/safe" if its name ends in "ReadOnlyAccess" or "ViewOnlyAccess".
Policies that don't fall cleanly into either bucket by name are excluded
from the ground-truth comparison (reported separately, unlabeled) rather
than forced into a label we can't defend.

This is an external generalization test, not a benchmark this project
tuned itself against.
"""

import json
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

sys.path.insert(0, ".")
from exp1_iam_analyzer import (
    build_dataset, rule_based_flag, extract_features, extract_features_generalized,
    RandomForestClassifier,
)

FIXTURE_PATH = "realworld_repo/test/files/example-authz-details.json"


def normalize_policy(document):
    """Real IAM policy JSON allows Resource to be a string OR a list.
    Normalize Resource to a scalar so rule_based_flag/extract_features
    (which expect the scalar shape our synthetic generator always emits)
    still see a wildcard if one exists anywhere in a real multi-resource
    statement."""
    stmts = []
    for stmt in document.get("Statement", []):
        resource = stmt.get("Resource", "")
        if isinstance(resource, list):
            resource = "*" if "*" in resource else (resource[0] if resource else "")
        stmts.append({
            "Effect": stmt.get("Effect", "Allow"),
            "Action": stmt.get("Action", []),
            "Resource": resource,
        })
    return {"Version": document.get("Version", "2012-10-17"), "Statement": stmts}


def label_from_name(policy_name):
    if policy_name.endswith("FullAccess") or policy_name in (
            "AdministratorAccess", "PowerUserAccess"):
        return 1  # broad / risky, per AWS's own guidance
    if policy_name.endswith("ReadOnlyAccess") or policy_name.endswith("ViewOnlyAccess"):
        return 0  # narrow / safe
    return None  # ambiguous by name alone - excluded from labeled comparison


def load_real_policies(path=FIXTURE_PATH):
    with open(path) as f:
        data = json.load(f)
    out = []
    for p in data["Policies"]:
        arn = p.get("Arn", "")
        if not arn.startswith("arn:aws:iam::aws:policy/"):
            continue  # skip the two synthetic test fixtures in this file
        name = p["PolicyName"]
        doc = p["PolicyVersionList"][0]["Document"]
        out.append((name, normalize_policy(doc), label_from_name(name)))
    return out


def metrics(preds, y_true):
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {"accuracy": round(acc, 4), "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    real_policies = load_real_policies()
    print(f"Loaded {len(real_policies)} real, live AWS-managed IAM policies.", file=sys.stderr)

    labeled = [(n, p, l) for n, p, l in real_policies if l is not None]
    unlabeled = [n for n, p, l in real_policies if l is None]

    dataset = build_dataset()
    policies = [p for p, label in dataset]
    labels = [label for p, label in dataset]
    y_bin = np.array([0 if l == "safe" else 1 for l in labels])

    X_orig = np.array([extract_features(p) for p in policies])
    clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_orig.fit(X_orig, y_bin)

    X_gen = np.array([extract_features_generalized(p) for p in policies])
    clf_gen = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_gen.fit(X_gen, y_bin)

    y_true = np.array([l for _, _, l in labeled])
    rule_preds = np.array([int(rule_based_flag(p)) for _, p, _ in labeled])
    ml_preds_orig = clf_orig.predict(np.array([extract_features(p) for _, p, _ in labeled]))
    ml_preds_gen = clf_gen.predict(np.array([extract_features_generalized(p) for _, p, _ in labeled]))

    # sanity check: does the generalized feature hurt performance on the
    # original synthetic test the main paper already reports?
    Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(X_gen, y_bin, test_size=0.3, random_state=42, stratify=y_bin)
    clf_gen_check = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_gen_check.fit(Xg_tr, yg_tr)
    gen_synth_preds = clf_gen_check.predict(Xg_te)
    gen_synth_metrics = {
        "accuracy": round(accuracy_score(yg_te, gen_synth_preds), 4),
        "recall": round(recall_score(yg_te, gen_synth_preds, zero_division=0), 4),
    }

    per_policy = [
        {"policy": n, "true_label": "risky" if l == 1 else "safe",
         "rule_flagged": bool(rp), "rf_original_flagged": bool(mo), "rf_generalized_flagged": bool(mg)}
        for (n, _, l), rp, mo, mg in zip(labeled, rule_preds, ml_preds_orig, ml_preds_gen)
    ]

    results = {
        "source": "salesforce/cloudsplaining test fixture "
                   "(real AWS-managed policy documents, not synthetic)",
        "total_real_policies_loaded": len(real_policies),
        "labeled_for_evaluation": len(labeled),
        "excluded_ambiguous_by_name": unlabeled,
        "rule_based_on_real_policies": metrics(rule_preds, y_true),
        "random_forest_original_features_on_real_policies": metrics(ml_preds_orig, y_true),
        "random_forest_generalized_features_on_real_policies": metrics(ml_preds_gen, y_true),
        "random_forest_generalized_features_on_synthetic_test_set_sanity_check": gen_synth_metrics,
        "per_policy_detail": per_policy,
    }
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
