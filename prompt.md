## 使用 pandas, sklearn 對一個 CSV 中的 data 進行前處理，並輸出一個新的 CSV
- 針對 float (ex:  column 'A01', 'A02', ...) 類別的 data，使用 'StandardScaler'，可編輯指定的 column
- 針對 string (ex:  column 'B01', 'B02', ...) 類別的 data，使用 'OneHotEncoder'，可編輯指定的 column 並定義對應關係 

- 將 preprocess.py 中 string 類別進行修改，不只使用 OneHotEncoder，對部分 column 使用 LabelEncoder

- 新增一個 model_train_SVM.py，功能與 model_train.py 類似，但使用 SVM (support vector machine) 演算法