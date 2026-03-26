import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_CSV  = "test_data.csv"
OUTPUT_CSV = "test_data_processed.csv"

# Float columns to scale with StandardScaler (e.g. 'A01', 'A02', ...)
FLOAT_COLS = ["x", "y", "value"]

# String columns to encode with OneHotEncoder (e.g. 'B01', 'B02', ...)
# For each column you can define the allowed categories explicitly.
# Set to None to infer categories automatically from the data.
STRING_COLS = {
    # "B01": ["cat", "dog", "bird"],   # explicit category order
    # "B02": None,                     # infer from data
}

# ─── Main ─────────────────────────────────────────────────────────────────────

def preprocess(input_csv: str, output_csv: str,
               float_cols: list, string_cols: dict) -> pd.DataFrame:

    df = pd.read_csv(input_csv)
    print(f"Loaded '{input_csv}': {df.shape[0]} rows × {df.shape[1]} cols")

    result = df.copy()

    # --- StandardScaler on float columns ---
    valid_float = [c for c in float_cols if c in df.columns]
    if valid_float:
        scaler = StandardScaler()
        result[valid_float] = scaler.fit_transform(df[valid_float].astype(float))
        print(f"StandardScaler applied to: {valid_float}")
    else:
        print("No float columns to scale.")

    # --- OneHotEncoder on string columns ---
    valid_str = {c: cats for c, cats in string_cols.items() if c in df.columns}
    if valid_str:
        for col, categories in valid_str.items():
            cats_arg = [categories] if categories is not None else "auto"
            enc = OneHotEncoder(categories=cats_arg, sparse_output=False,
                                handle_unknown="ignore")
            encoded = enc.fit_transform(df[[col]])
            feature_names = enc.get_feature_names_out([col])
            enc_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

            # Insert encoded columns next to the original, then drop it
            insert_pos = result.columns.get_loc(col)
            result = pd.concat(
                [result.iloc[:, :insert_pos], enc_df,
                 result.iloc[:, insert_pos + 1:]],
                axis=1
            )
            print(f"OneHotEncoder applied to '{col}' → {list(feature_names)}")
    else:
        print("No string columns to encode.")

    result.to_csv(output_csv, index=False)
    print(f"Saved to '{output_csv}'")
    return result


if __name__ == "__main__":
    df_out = preprocess(INPUT_CSV, OUTPUT_CSV, FLOAT_COLS, STRING_COLS)
    print(df_out.head())
