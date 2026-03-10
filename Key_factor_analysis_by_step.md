
這是一份為你整理的 **Data Science 專案標準作業程序 (SOP)**。這份指南針對你的資料規模（500-1000 筆、30 個參數）進行了最佳化，重點在於「高品質預處理」與「模型可解釋性」。

---

## 📊 數據分析與模型訓練標準流程 (Python 版)

### Step 1: 環境配置與函式庫載入

準備好數據處理、機器學習與模型解釋的核心工具。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap

```

---

### Step 2: 資料清洗與離群值處理 (Cleaning & Outliers)

針對小規模數據，避免直接刪除，改用「蓋帽法」或「轉換法」。

* **缺失值**：使用中位數（數值）或眾數（類別）填補。
* **離群值處理 (Capping)**：將極端值限制在 1% 與 99% 分位數。
* **偏態處理 (Log)**：針對「僅為正」且分佈極度不均的數值做 $log(x + 1)$。

```python
# 1. 蓋帽法處理離群值
def handle_outliers(df, cols):
    for col in cols:
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower, upper)
    return df

# 2. 定義對數轉換 (針對偏態正數)
log_transformer = FunctionTransformer(np.log1p)

```

---

### Step 3: 特徵工程與預處理 (Preprocessing)

使用 `ColumnTransformer` 針對不同型態的參數進行精確分流。

* **類別型 (String)**：使用 `OneHotEncoder`。
* **偏態數值 (Float)**：先 Log 再 `StandardScaler`。
* **一般數值 (Float)**：直接 `StandardScaler`（$z = \frac{x - \mu}{\sigma}$）。

```python
# 定義欄位群組
num_cols = ['age', 'score']        # 一般數值
skewed_cols = ['revenue', 'count'] # 偏態數值 (僅正)
cat_cols = ['city', 'gender']      # 類別欄位

# 建立預處理流水線
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('log_num', Pipeline([('log', log_transformer), ('std', StandardScaler())]), skewed_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

```

---

### Step 4: 資料分割 (Stratified Split)

由於資料量較小，必須使用 **層化抽樣** 確保訓練集與測試集的目標比例一致。

```python
X = df.drop('performance', axis=1)
y = df['performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

```

---

### Step 5: 模型訓練與特徵篩選 (Modeling)

建議並行使用兩種模型：**Lasso (L1)** 篩選關鍵線性參數，**隨機森林** 捕捉非線性重要性。

```python
# 使用 Lasso (L1) 進行自動特徵篩選
lasso_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(penalty='l1', solver='liblinear', C=0.5))
])
lasso_pipeline.fit(X_train, y_train)

# 使用隨機森林獲取魯棒性預測
rf_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

```

---

### Step 6: 關鍵參數分析 (SHAP Explainability)

這是找出「影響效能關鍵」的核心步驟。SHAP 可以量化每個參數對結果的貢獻。

```python
# 準備 SHAP 解釋
X_test_prepped = rf_pipeline.named_steps['pre'].transform(X_test)
# 獲取處理後的特徵名稱
feature_names = num_cols + skewed_cols + list(rf_pipeline.named_steps['pre'].transformers_[2][1].get_feature_names_out())

explainer = shap.TreeExplainer(rf_pipeline.named_steps['clf'])
shap_values = explainer.shap_values(X_test_prepped)

# 繪製 Summary Plot (最重要的圖表)
shap.summary_plot(shap_values[1], X_test_prepped, feature_names=feature_names)

```

---

## 💡 歸納重點總結

1. **資料量小 (500-1000)**：不要輕易刪除資料，優先使用 **Capping (蓋帽法)** 處理離群值。
2. **層化抽樣 (Stratified)**：這是保證驗證準確性的生命線，務必開啟。
3. **偏態數值**：針對僅為正的金額、次數類參數，**Log 轉換** 後的效果通常遠優於直接標準化。
4. **解釋性 > 準確性**：在分析關鍵參數的情境下，如果 Lasso 告訴你某個參數係數為 0，而 SHAP 也顯示它影響力極低，那它就是可以被剔除的雜訊。

**下一步建議：**
如果你已經有資料集，我可以幫你撰寫一段 **EDA 自動化腳本**，幫你快速產出這 30 個參數的分布圖與箱型圖，方便你決定哪些欄位需要做 Log 轉換或 Capping。需要我為你準備嗎？
