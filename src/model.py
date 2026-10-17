"""Baseline severity models: predict whether a crash is fatal or serious (KSI).

Temporal split: train on 2020-2023, test on 2024 (mimics using past data to flag future risk).
Features only describe the crash circumstances, never the injury counts (which define the target).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             recall_score, precision_score, f1_score, balanced_accuracy_score, brier_score_loss)
from features import build

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

CAT = ["Urbanisation", "Speed limit", "Weather", "Natural lighting", "Surface condition", "Road surface",
       "Alignment", "Type of location", "Road classification (admin)", "First impact type", "Key TU type",
       "Two-hour intervals", "Day of week of crash", "Street lighting", "School zone active",
       "Signals operation", "Other traffic control"]
NUM = ["No. of traffic units involved", "has_pedestrian", "has_motorcycle", "has_pedal_cycle",
       "has_heavy_vehicle", "hit_tree_or_pole", "has_parked_vehicle"]

import sys
# Street lighting 'Unknown / not stated' is recorded for 63% of tow-away crashes but 0.6% of fatal ones:
# its completeness depends on the outcome (recording artefact / leakage), so the main model excludes it.
LEAKY = ["Street lighting"]
SPEC = sys.argv[1] if len(sys.argv) > 1 else "main"
if SPEC == "main":
    CAT = [f for f in CAT if f not in LEAKY]
SUFFIX = "" if SPEC == "main" else "_with_street_lighting"

c, _ = build()
X, y = c[CAT + NUM].copy(), c["ksi"]
X[CAT] = X[CAT].fillna("Missing").astype(str)
train, test = c["Year of crash"] <= 2023, c["Year of crash"] == 2024
Xtr, Xte, ytr, yte = X[train], X[test], y[train], y[test]
print("SPEC", SPEC); print(f"train {len(Xtr):,} (KSI {ytr.mean():.1%}) | test {len(Xte):,} (KSI {yte.mean():.1%})")

ohe = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CAT)],
                        remainder="passthrough")
models = {
    "Baseline (always 'not KSI')": Pipeline([("m", DummyClassifier(strategy="prior"))]),
    "Logistic regression": Pipeline([("prep", ohe), ("m", LogisticRegression(max_iter=3000, C=0.5,
                                                                              class_weight="balanced"))]),
    "Gradient boosting": Pipeline([
        ("prep", ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                           encoded_missing_value=-1), CAT)],
                                   remainder="passthrough")),
        ("m", HistGradientBoostingClassifier(categorical_features=list(range(len(CAT))), learning_rate=0.05,
                                             max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
                                             class_weight="balanced", random_state=42))]),
}

results, probs = [], {}
for name, pipe in models.items():
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    probs[name] = p
    pred = (p >= 0.5).astype(int)
    results.append({"model": name, "ROC-AUC": roc_auc_score(yte, p), "PR-AUC": average_precision_score(yte, p),
                    "recall_KSI": recall_score(yte, pred, zero_division=0),
                    "precision_KSI": precision_score(yte, pred, zero_division=0),
                    "F1_KSI": f1_score(yte, pred, zero_division=0),
                    "balanced_acc": balanced_accuracy_score(yte, pred),
                    "fatal_crashes_flagged": float(pred[c.loc[test, "fatal"].values == 1].mean())})
res = pd.DataFrame(results).set_index("model").round(3)
print(res.to_string())
res.to_csv(ROOT / "reports" / f"model_results{SUFFIX}.csv")

# Diagnostics: training-set fit (overfitting check), model internals, and year-by-year stability
from sklearn.base import clone
diag = {"n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "ksi_rate_train": float(ytr.mean()), "ksi_rate_test": float(yte.mean())}
for name in ["Logistic regression", "Gradient boosting"]:
    p_tr = models[name].predict_proba(Xtr)[:, 1]
    diag[name] = {"train_roc_auc": roc_auc_score(ytr, p_tr), "test_roc_auc": roc_auc_score(yte, probs[name]),
                  "train_pr_auc": average_precision_score(ytr, p_tr), "test_pr_auc": average_precision_score(yte, probs[name])}
lr_fit = models["Logistic regression"]
diag["Logistic regression"]["n_encoded_columns"] = int(len(lr_fit.named_steps["prep"].get_feature_names_out()))
diag["Logistic regression"]["solver_iterations"] = int(lr_fit.named_steps["m"].n_iter_[0])
gb_fit = models["Gradient boosting"].named_steps["m"]
diag["Gradient boosting"]["boosting_iterations_used"] = int(gb_fit.n_iter_)
diag["Gradient boosting"]["early_stopping_active"] = bool(gb_fit.do_early_stopping_)
# Leave-one-year-out within the training years (2020-2023): train on 3 years, score the held-out year
years = c.loc[train, "Year of crash"]
loyo = {}
for name in ["Logistic regression", "Gradient boosting"]:
    loyo[name] = {}
    for yr in sorted(years.unique()):
        fit = clone(models[name]).fit(Xtr[years != yr], ytr[years != yr])
        loyo[name][int(yr)] = roc_auc_score(ytr[years == yr], fit.predict_proba(Xtr[years == yr])[:, 1])
diag["leave_one_year_out_roc_auc"] = loyo
json.dump(diag, open(ROOT / "reports" / f"model_diagnostics{SUFFIX}.json", "w"), indent=2)
print(json.dumps(diag, indent=1))

# ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for (name, p), color in zip([(k, v) for k, v in probs.items() if "Baseline" not in k], [BLUE, ORANGE]):
    fpr, tpr, _ = roc_curve(yte, p)
    axes[0].plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
    pr, rc, _ = precision_recall_curve(yte, p)
    axes[1].plot(rc, pr, lw=2, color=color, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", color=INK2, lw=1, label="Random")
axes[1].axhline(yte.mean(), ls="--", color=INK2, lw=1, label=f"Random ({yte.mean():.1%} KSI)")
axes[0].set(title="ROC curve, 2024 test set", xlabel="False positive rate", ylabel="True positive rate")
axes[1].set(title="Precision–recall curve, 2024 test set", xlabel="Recall", ylabel="Precision", ylim=(0, 1))
for ax in axes:
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / f"07_model_roc_pr{SUFFIX}.png", dpi=200); plt.close(fig)

# Logistic regression odds ratios (top +/-)
lr = models["Logistic regression"]
names = lr.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr.named_steps["m"].coef_[0], index=names)
coef.index = coef.index.str.replace("cat__", "").str.replace("remainder__", "")
odds = np.exp(coef).sort_values()
odds.to_csv(ROOT / "reports" / f"logreg_odds_ratios{SUFFIX}.csv", header=["odds_ratio"])
top = pd.concat([odds.head(10), odds.tail(12)])
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top.index, top.values - 1, left=1, color=[ORANGE if v > 1 else BLUE for v in top.values], height=0.6)
ax.axvline(1, color=INK2, lw=1); ax.set_xscale("log"); ax.grid(axis="y", visible=False)
ticks = [0.2, 0.3, 0.5, 1, 2, 3, 5, 8]
ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
ax.set_title("Logistic regression: strongest odds ratios for KSI")
ax.set_xlabel("Odds ratio (log scale; >1 raises the odds of a fatal/serious outcome)")
fig.tight_layout(); fig.savefig(FIG / f"08_logreg_odds_ratios{SUFFIX}.png", dpi=200); plt.close(fig)

# Permutation importance (gradient boosting, on the test set, metric = ROC-AUC)
gb = models["Gradient boosting"]
pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=CAT + NUM).sort_values()
imp.to_csv(ROOT / "reports" / f"gb_permutation_importance{SUFFIX}.csv", header=["auc_drop"])
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(imp.index, imp.values, color=BLUE, height=0.6); ax.grid(axis="y", visible=False)
ax.set_title("Gradient boosting: permutation importance")
ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
fig.tight_layout(); fig.savefig(FIG / f"09_gb_permutation_importance{SUFFIX}.png", dpi=200); plt.close(fig)

# Calibration / risk deciles for gradient boosting
p = probs["Gradient boosting"]
dec = pd.DataFrame({"p": p, "y": yte.values}).assign(decile=lambda d: pd.qcut(d.p, 10, labels=False) + 1)
cal = dec.groupby("decile").agg(mean_pred=("p", "mean"), actual_ksi=("y", "mean"), n=("y", "size"))
print("\nRisk deciles (GB):\n", cal.round(3).to_string())
print("\nTop permutation importances:\n", imp.sort_values(ascending=False).head(10).round(4).to_string())
print("\nTop odds ratios:\n", odds.tail(12).round(2).to_string(), "\n", odds.head(8).round(2).to_string())
json.dump({"n_train": int(len(Xtr)), "n_test": int(len(Xte)), "ksi_rate_test": float(yte.mean()),
           "top_decile_ksi": float(cal.actual_ksi.iloc[-1]), "bottom_decile_ksi": float(cal.actual_ksi.iloc[0])},
          open(ROOT / "reports" / f"model_summary{SUFFIX}.json", "w"), indent=2)
