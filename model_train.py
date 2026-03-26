import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ─── Configuration ────────────────────────────────────────────────────────────

# Input: CSV produced by preprocess.py (already scaled / encoded)
INPUT_CSV  = "test_data_processed.csv"

# Column to predict (must exist in the CSV)
TARGET_COL = "performance"

# Fraction held out for testing
TEST_SIZE  = 0.2
RANDOM_STATE = 42

# Lasso regularisation strength (lower C → stronger regularisation / more sparsity)
LASSO_C = 0.5

# Random Forest settings
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 5

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

# ─── Step 5: Model Training & Feature Selection ────────────────────────────────

# --- Lasso (L1) : linear feature selection ---
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=LASSO_C,
                           random_state=RANDOM_STATE, max_iter=1000)
lasso.fit(X_train, y_train)

lasso_acc = accuracy_score(y_test, lasso.predict(X_test))
print(f"\n── Step 5a: Lasso (L1)  — test accuracy: {lasso_acc:.4f} ──")

coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lasso.coef_[0] if lasso.coef_.ndim > 1 else lasso.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(coef_df.to_string(index=False))
zero_features = coef_df[coef_df['coefficient'] == 0.0]['feature'].tolist()
if zero_features:
    print(f"\n  → Lasso zeroed out (candidate noise): {zero_features}")

# --- Random Forest : non-linear importance ---
rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                            random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"\n── Step 5b: Random Forest — test accuracy: {rf_acc:.4f} ──")

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.to_string(index=False))
print(f"\n{classification_report(y_test, rf.predict(X_test))}")

# --- Summary ---
print("── Summary ──")
print(f"  Lasso  accuracy : {lasso_acc:.4f}")
print(f"  RF     accuracy : {rf_acc:.4f}")
print("\n  Features ranked by RF importance:")
for _, row in importance_df.iterrows():
    lasso_coef = coef_df.set_index('feature').loc[row['feature'], 'coefficient']
    tag = "⚠ zeroed by Lasso" if lasso_coef == 0.0 else ""
    print(f"    {row['feature']:20s}  RF={row['importance']:.4f}  Lasso={lasso_coef:+.4f}  {tag}")
