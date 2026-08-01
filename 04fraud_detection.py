"""
Fraud Detection in Financial Transactions
==========================================
Goal: identify suspicious / fraudulent transactions.

- Synthetic anonymized transaction dataset (structured like the classic
  "credit card fraud" schema: Amount, Time, anonymized PCA-style features
  V1..V10, Class).
- Unsupervised anomaly detection: Isolation Forest.
- Class imbalance handling: manual oversampling (SMOTE-style interpolation,
  since imbalanced-learn isn't available offline) + class_weight='balanced'
  supervised baseline for comparison.
- Alert-based dashboard: KPI cards, confusion matrix, PR curve, top alerts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score
)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Generate synthetic imbalanced transaction data
# ---------------------------------------------------------------------------
N_NORMAL = 20000
N_FRAUD = 120  # ~0.6% fraud rate, similar to real-world card fraud datasets

n_features = 10  # V1..V10, anonymized PCA-style components

# Normal transactions: centered near 0, moderate amounts
normal_V = rng.normal(loc=0, scale=1.0, size=(N_NORMAL, n_features))
normal_amount = np.abs(rng.normal(loc=60, scale=50, size=N_NORMAL))
normal_time = rng.uniform(0, 172800, size=N_NORMAL)  # 2 days, seconds

# Fraudulent transactions: shifted distribution, higher variance, often larger
# or unusually small amounts, clustered in odd hours. The shift is deliberately
# modest and the variance high so the classes overlap substantially, like in
# real fraud data -- this is what makes class-imbalance handling matter.
fraud_V = rng.normal(loc=0.9, scale=1.8, size=(N_FRAUD, n_features))
fraud_amount = np.abs(rng.normal(loc=180, scale=220, size=N_FRAUD))
fraud_time = rng.uniform(0, 172800, size=N_FRAUD)

V_cols = [f"V{i+1}" for i in range(n_features)]
df_normal = pd.DataFrame(normal_V, columns=V_cols)
df_normal["Amount"] = normal_amount
df_normal["Time"] = normal_time
df_normal["Class"] = 0

df_fraud = pd.DataFrame(fraud_V, columns=V_cols)
df_fraud["Amount"] = fraud_amount
df_fraud["Time"] = fraud_time
df_fraud["Class"] = 1

df = pd.concat([df_normal, df_fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total transactions: {len(df)}")
print(f"Fraud rate: {df['Class'].mean()*100:.3f}%  ({df['Class'].sum()} fraud / {len(df)} total)")
print(df.head())

df.to_csv("/home/claude/fraud/synthetic_transactions.csv", index=False)

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
df["Class"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"]
)
axes[0].set_xticklabels(["Normal (0)", "Fraud (1)"], rotation=0)
axes[0].set_title("Class Distribution (highly imbalanced)")
axes[0].set_ylabel("Count")
for i, v in enumerate(df["Class"].value_counts().sort_index()):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

axes[1].hist(df.loc[df["Class"] == 0, "Amount"], bins=50, alpha=0.6, label="Normal", color="#4C72B0", density=True)
axes[1].hist(df.loc[df["Class"] == 1, "Amount"], bins=50, alpha=0.6, label="Fraud", color="#C44E52", density=True)
axes[1].set_title("Transaction Amount Distribution by Class")
axes[1].set_xlabel("Amount")
axes[1].legend()
plt.tight_layout()
plt.savefig("/home/claude/fraud/01_eda.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Train / test split (stratified, so both sets keep the fraud ratio)
# ---------------------------------------------------------------------------
feature_cols = V_cols + ["Amount", "Time"]
X = df[feature_cols].copy()
# Scale Amount/Time to be comparable to the already-standardized V features
X["Amount"] = (X["Amount"] - X["Amount"].mean()) / X["Amount"].std()
X["Time"] = (X["Time"] - X["Time"].mean()) / X["Time"].std()
y = df["Class"].values

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_train, X_test = X_train_df.values, X_test_df.values
print(f"\nTrain fraud rate: {y_train.mean()*100:.3f}%  |  Test fraud rate: {y_test.mean()*100:.3f}%")

# ---------------------------------------------------------------------------
# 4. Unsupervised anomaly detection: Isolation Forest
# ---------------------------------------------------------------------------
contamination = y_train.mean()  # estimate expected fraud proportion
iso_forest = IsolationForest(
    n_estimators=200, contamination=contamination, random_state=42
)
iso_forest.fit(X_train)

# -1 = anomaly (flag as fraud), 1 = normal
raw_pred = iso_forest.predict(X_test)
iso_pred = np.where(raw_pred == -1, 1, 0)
iso_scores = -iso_forest.score_samples(X_test)  # higher = more anomalous

print("\n--- Isolation Forest (unsupervised) ---")
print(confusion_matrix(y_test, iso_pred))
print(classification_report(y_test, iso_pred, target_names=["Normal", "Fraud"], digits=3))
iso_auc = roc_auc_score(y_test, iso_scores)
iso_ap = average_precision_score(y_test, iso_scores)
print(f"ROC-AUC: {iso_auc:.3f} | Average Precision (PR-AUC): {iso_ap:.3f}")

# ---------------------------------------------------------------------------
# 5. Class-imbalance handling: manual SMOTE-style oversampling + supervised model
# ---------------------------------------------------------------------------
def smote_like_oversample(X_min, n_samples, k=5, rng=rng):
    """Simple SMOTE-style synthetic sample generation via interpolation
    between each minority point and one of its k nearest neighbors
    (nearest neighbors approximated via Euclidean distance)."""
    from sklearn.neighbors import NearestNeighbors
    X_min = np.asarray(X_min)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_min))).fit(X_min)
    _, indices = nn.kneighbors(X_min)
    synthetic = []
    for _ in range(n_samples):
        i = rng.integers(0, len(X_min))
        neighbor_choices = indices[i][1:]  # exclude self
        if len(neighbor_choices) == 0:
            j = i
        else:
            j = neighbor_choices[rng.integers(0, len(neighbor_choices))]
        gap = rng.random()
        new_point = X_min[i] + gap * (X_min[j] - X_min[i])
        synthetic.append(new_point)
    return np.array(synthetic)


X_train_min = X_train[y_train == 1]
X_train_maj = X_train[y_train == 0]
n_to_generate = len(X_train_maj) - len(X_train_min)  # balance classes fully

synthetic_fraud = smote_like_oversample(X_train_min, n_to_generate, k=5)

X_train_bal = np.vstack([X_train_maj, X_train_min, synthetic_fraud])
y_train_bal = np.array([0] * len(X_train_maj) + [1] * (len(X_train_min) + len(synthetic_fraud)))

# Shuffle
perm = rng.permutation(len(X_train_bal))
X_train_bal, y_train_bal = X_train_bal[perm], y_train_bal[perm]

print(f"\nBalanced training set: {len(X_train_bal)} rows "
      f"({(y_train_bal==1).sum()} fraud / {(y_train_bal==0).sum()} normal)")

clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
)
clf.fit(X_train_bal, y_train_bal)

rf_pred = clf.predict(X_test)
rf_scores = clf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest (supervised, SMOTE-style oversampling + class_weight) ---")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names=["Normal", "Fraud"], digits=3))
rf_auc = roc_auc_score(y_test, rf_scores)
rf_ap = average_precision_score(y_test, rf_scores)
print(f"ROC-AUC: {rf_auc:.3f} | Average Precision (PR-AUC): {rf_ap:.3f}")

# ---------------------------------------------------------------------------
# 6. Alert-based dashboard
# ---------------------------------------------------------------------------
ALERT_THRESHOLD = 0.5
alerts = rf_scores >= ALERT_THRESHOLD
n_alerts = alerts.sum()
n_true_fraud_caught = ((alerts) & (y_test == 1)).sum()
n_false_alerts = ((alerts) & (y_test == 0)).sum()
alert_precision = n_true_fraud_caught / n_alerts if n_alerts else 0
alert_recall = n_true_fraud_caught / (y_test == 1).sum()

fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)
fig.suptitle("Fraud Detection — Alert Dashboard", fontsize=16, fontweight="bold")

# KPI cards (as text panels)
kpis = [
    ("Transactions Scored", f"{len(y_test):,}"),
    ("Alerts Raised", f"{n_alerts:,}"),
    ("Fraud Caught", f"{n_true_fraud_caught} / {(y_test==1).sum()}"),
    ("Alert Precision", f"{alert_precision*100:.1f}%"),
    ("Alert Recall", f"{alert_recall*100:.1f}%"),
    ("PR-AUC (model)", f"{rf_ap:.3f}"),
]
for idx, (label, value) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, idx % 3]) if idx < 3 else fig.add_subplot(gs[1, idx % 3])
    ax.axis("off")
    ax.text(0.5, 0.65, value, ha="center", va="center", fontsize=22, fontweight="bold", color="#2A5C8A")
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=11, color="#444")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor="#ccc", linewidth=1, transform=ax.transAxes))

# Confusion matrix
ax_cm = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_test, rf_pred)
im = ax_cm.imshow(cm, cmap="Blues")
ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Normal", "Fraud"])
ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["Normal", "Fraud"])
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix (RF model)")
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

# Precision-Recall curve
ax_pr = fig.add_subplot(gs[2, 1])
prec, rec, _ = precision_recall_curve(y_test, rf_scores)
ax_pr.plot(rec, prec, color="#C44E52", label=f"RF (AP={rf_ap:.3f})")
prec_iso, rec_iso, _ = precision_recall_curve(y_test, iso_scores)
ax_pr.plot(rec_iso, prec_iso, color="#4C72B0", linestyle="--", label=f"IsolationForest (AP={iso_ap:.3f})")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve")
ax_pr.legend(fontsize=8)

# Top flagged transactions table
ax_tbl = fig.add_subplot(gs[2, 2])
ax_tbl.axis("off")
top_idx = np.argsort(-rf_scores)[:6]
table_data = []
for i in top_idx:
    table_data.append([
        f"{i}",
        f"{rf_scores[i]:.2f}",
        "FRAUD" if y_test[i] == 1 else "normal",
    ])
tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=["Txn #", "Risk Score", "Actual"],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)
ax_tbl.set_title("Top 6 Highest-Risk Alerts", fontsize=10)

plt.savefig("/home/claude/fraud/02_alert_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved visualizations: 01_eda.png, 02_alert_dashboard.png")

# ---------------------------------------------------------------------------
# 7. Business insights
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)
print(f"- Unsupervised Isolation Forest needs no labeled fraud examples at all and reaches "
      f"PR-AUC={iso_ap:.3f} / ROC-AUC={iso_auc:.3f} — a strong always-on first-pass screen for when "
      f"confirmed fraud labels are scarce or delayed (the realistic case early in a fraud program).")

better_model = "Random Forest (with SMOTE-style balancing)" if rf_ap > iso_ap else "Isolation Forest (unsupervised)"
print(f"- Once labeled fraud cases accumulate, a supervised Random Forest trained on a SMOTE-balanced, "
      f"fully rebalanced dataset reaches PR-AUC={rf_ap:.3f} / ROC-AUC={rf_auc:.3f}. On this data the "
      f"{better_model} performs better on PR-AUC — in practice both approaches should run in parallel "
      f"and be validated on rolling data, since fraud patterns drift over time.")
print(f"- At a 0.5 risk-score threshold, the supervised model raises {n_alerts} alerts, catches "
      f"{n_true_fraud_caught}/{(y_test==1).sum()} true fraud cases ({alert_recall*100:.1f}% recall) "
      f"at {alert_precision*100:.1f}% precision — i.e. investigators review roughly "
      f"{n_false_alerts} false positives for every {max(n_true_fraud_caught,1)} confirmed fraud caught.")
print("- The alert threshold is a business lever, not a fixed setting: lowering it catches more fraud "
      "at the cost of more analyst review time; raising it reduces workload but risks missing fraud. "
      "It should be tuned against the actual cost of a missed fraud vs. the cost of an analyst-hour.")
print("- Precision-Recall (not accuracy) is the right metric here: with a 0.6% fraud rate, a model "
      "that predicts 'normal' for everything would already be 99.4% accurate while catching zero fraud.")
