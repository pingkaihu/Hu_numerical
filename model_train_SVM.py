import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score

# ─── Configuration ────────────────────────────────────────────────────────────

# Input: CSV produced by preprocess.py (already scaled / encoded)
INPUT_CSV  = "test_data_processed.csv"

# Column to predict (must exist in the CSV)
TARGET_COL = "performance"

# Fraction held out for testing
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# SVM settings
SVM_C      = 1.0     # regularisation strength (lower C → smoother boundary)
SVM_KERNEL = "rbf"   # 'linear', 'rbf', 'poly', 'sigmoid'
SVM_GAMMA  = "scale" # 'scale', 'auto', or float  (used by rbf / poly / sigmoid)

# ─── Step 4: Data Split (Stratified) ──────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
print(f"Loaded '{INPUT_CSV}': {df.shape[0]} rows × {df.shape[1]} cols")

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found in CSV.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Please set TARGET_COL to one of the above."
    )

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Stratified split — critical for small datasets to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\n── Step 4: Stratified Split ──")
print(f"  Train : {len(X_train)} samples  |  Test : {len(X_test)} samples")
print(f"  Target distribution (train):\n{y_train.value_counts(normalize=True).round(3)}")

# ─── Step 5: Model Training & Feature Importance ──────────────────────────────

svc = SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA,
          random_state=RANDOM_STATE)
svc.fit(X_train, y_train)

svm_acc = accuracy_score(y_test, svc.predict(X_test))
print(f"\n── Step 5: SVM ({SVM_KERNEL} kernel)  — test accuracy: {svm_acc:.4f} ──")

# Feature importance:
#   linear kernel → coef_ gives direct weights (interpretable like Lasso)
#   other kernels → permutation importance (model-agnostic, computed on test set)
if SVM_KERNEL == "linear":
    imp        = svc.coef_[0]
    imp_label  = "weight"
else:
    r          = permutation_importance(svc, X_test, y_test,
                                        n_repeats=30, random_state=RANDOM_STATE)
    imp        = r.importances_mean
    imp_label  = "perm_importance"

importance_df = pd.DataFrame({
    'feature':   X.columns,
    imp_label:   imp,
}).sort_values(imp_label, key=abs, ascending=False)

print(importance_df.to_string(index=False))
print(f"\n{classification_report(y_test, svc.predict(X_test))}")

# --- Summary ---
print("── Summary ──")
print(f"  SVM ({SVM_KERNEL})  accuracy : {svm_acc:.4f}")
print(f"\n  Features ranked by |{imp_label}|:")
for _, row in importance_df.iterrows():
    print(f"    {row['feature']:20s}  {imp_label}={row[imp_label]:+.4f}")
