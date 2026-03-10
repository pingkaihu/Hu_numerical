這是一份專為 30 個參數設計的 **EDA 自動化腳本**。

考慮到參數多達 30 個，如果一張張畫圖會非常混亂。這份腳本會自動將參數分組，並以**網格 (Grid)** 的方式呈現，讓你在一兩張大圖中就能掃描完所有特徵的分佈狀況與離群值。

---

## 🐍 Python EDA (Exploratory Data Analysis) 自動化腳本

這段程式碼會根據資料型態自動分流，為數值型繪製「分佈圖與箱型圖」，為類別型繪製「次數統計圖」。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def run_automated_eda(df, target_col):
    # 1. 自動區分特徵型態
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_cols: num_cols.remove(target_col)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"📊 偵測完成: 數值型參數 {len(num_cols)} 個, 類別型參數 {len(cat_cols)} 個")

    # ---------------------------------------------------------
    # 2. 數值型參數分析：分佈 (Dist) 與 離群值 (Box)
    # ---------------------------------------------------------
    n_num = len(num_cols)
    if n_num > 0:
        rows = math.ceil(n_num / 3)
        # 繪製分佈圖 (Histograms)
        plt.figure(figsize=(18, 4 * rows))
        for i, col in enumerate(num_cols):
            plt.subplot(rows, 3, i + 1)
            # 依據 Target 分色，觀察參數對 Performance 的初步影響
            sns.histplot(data=df, x=col, hue=target_col, kde=True, element="step")
            plt.title(f'Distribution: {col}')
        plt.tight_layout()
        plt.show()

        # 繪製箱型圖 (Boxplots) - 專門看 Outliers
        plt.figure(figsize=(18, 4 * rows))
        for i, col in enumerate(num_cols):
            plt.subplot(rows, 3, i + 1)
            sns.boxplot(data=df, x=target_col, y=col)
            plt.title(f'Outliers: {col}')
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # 3. 類別型參數分析：次數統計 (Count Plot)
    # ---------------------------------------------------------
    n_cat = len(cat_cols)
    if n_cat > 0:
        rows_cat = math.ceil(n_cat / 3)
        plt.figure(figsize=(18, 5 * rows_cat))
        for i, col in enumerate(cat_cols):
            plt.subplot(rows_cat, 3, i + 1)
            # 計算各類別在不同 Performance 下的數量
            sns.countplot(data=df, x=col, hue=target_col)
            plt.xticks(rotation=45)
            plt.title(f'Count: {col}')
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # 4. 相關性熱圖 (Correlation Heatmap)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 10))
    # 僅計算數值型與目標的相關性
    correlation_matrix = df[num_cols + [target_col]].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix (Numerical Features)")
    plt.show()

# --- 使用範例 ---
# run_automated_eda(your_dataframe, target_col='performance')

```

---

## 🛠 腳本功能重點解析

### 1. 目標導向的分佈 (Hue=Target)

在 `sns.histplot` 中加入了 `hue=target_col`。這能讓你一眼看出：

* 如果 0 和 1 的分佈高度重合，代表該參數可能**不是**關鍵影響因素。
* 如果兩者分佈有明顯位移（例如：Performance=1 的數值普遍較高），這就是你要找的**關鍵參數**。

### 2. 箱型圖看離群值 (Boxplot)

腳本會根據 `performance` (0/1) 分別繪製箱型圖。

* **觀察重點**：如果箱子外的「點」非常多，代表離群值嚴重，這時你就需要套用我們之前討論的 **Capping (蓋帽法)** 或 **RobustScaler**。

### 3. 自動網格佈局 (Subplots)

腳本使用了 `math.ceil(n / 3)` 自動計算需要的列數，不管你有 30 個還是 50 個參數，它都會整齊地排成 3 欄式的大圖，方便你直接貼進分析報告中。

### 4. 類別參數的「稀疏性」檢查

透過 `countplot`，你可以檢查 string 參數。如果某個類別佔了 95% 以上，或者某些類別只有 1-2 筆資料，這種「高基數」或「數據稀疏」的特徵在模型訓練時通常會造成過度擬合，建議歸類為 "Others"。

---

## 💡 Data Scientist 的小建議

執行完這個腳本後，你可以快速勾選出以下清單：

1. **哪些欄位需要 Log 轉換？**（分佈圖極度右偏的欄位）
2. **哪些欄位需要處理 Outliers？**（箱型圖點很多的欄位）
3. **哪些欄位與 Target 無關？**（分佈完全重合或 Heatmap 顏色極淡的欄位）

**這段腳本在你的 IDE (如 Jupyter Notebook 或 VS Code) 跑起來後，需要我幫你解讀產出的特定圖表特徵嗎？**