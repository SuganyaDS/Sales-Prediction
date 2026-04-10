import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Set True to run EDA plots (blocks until you close the figure window)
RUN_EDA = False
TRAIN_PATH = r"C:\Users\SUGANYA\Downloads\train_v9rqX0R.csv"
TEST_PATH = r"C:\Users\SUGANYA\Downloads\test_AbJTz2l.csv"
OUTPUT_PATH = r"C:\Users\SUGANYA\ABB hackathon\submission.csv"


def evaluate_model(model_name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2, "estimator": model}


def main():
    df = pd.read_csv(TRAIN_PATH)

    target = "Item_Outlet_Sales"

    # --- EDA (optional) ---
    if RUN_EDA:
        print(df.shape)
        print(df.head())

        null_counts = df.isna().sum()
        null_pct = (df.isna().mean() * 100).round(2)
        missing = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
        missing = missing[missing["null_count"] > 0].sort_values(
            "null_count", ascending=False
        )
        print("Columns with missing values:")
        print(missing)
        print(f"\nRows with at least one null: {df.isna().any(axis=1).sum()} / {len(df)}")

        print("Shape:", df.shape)
        print("\nDtypes:\n", df.dtypes)
        print("\nDescribe (numeric):\n", df.describe().T)

        obj_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in obj_cols:
            print(f"\n--- {col} ---")
            print("Unique:", df[col].nunique(dropna=False))
            print(df[col].value_counts(dropna=False).head(10))

        num = df.select_dtypes(include="number")
        if num.shape[1] > 1:
            print("\nCorrelation matrix (numeric):\n", num.corr(numeric_only=True))

        df.hist(figsize=(12, 8), bins=30)
        plt.tight_layout()
        plt.show()

    # No handcrafted features: model uses raw columns + imputation + one-hot encoding.

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("prep", preprocess),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb_model = Pipeline(
        steps=[
            ("prep", preprocess),
            (
                "reg",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_result = evaluate_model(
        "RandomForestRegressor", rf_model, X_train, y_train, X_val, y_val
    )
    xgb_result = evaluate_model("XGBRegressor", xgb_model, X_train, y_train, X_val, y_val)

    results = sorted([rf_result, xgb_result], key=lambda x: x["rmse"])
    best = results[0]

    print("\n--- Validation metrics (Item_Outlet_Sales) ---")
    for result in results:
        print(
            f"{result['model']}: RMSE={result['rmse']:.4f}, "
            f"MAE={result['mae']:.4f}, R²={result['r2']:.4f}"
        )
    print(f"\nSelected model for test prediction: {best['model']}")
    print(
        "Best validation metrics -> "
        f"RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}, R²={best['r2']:.4f}"
    )

    # --- Predict on test data and save submission CSV ---
    test_file = Path(TEST_PATH)
    if not test_file.exists():
        print(f"\nTest file not found: {TEST_PATH}")
        print("Update TEST_PATH to your actual test CSV path.")
        return

    test_df = pd.read_csv(TEST_PATH)
    test_pred = best["estimator"].predict(test_df)

    # Keep Item_Identifier in output if available; otherwise use row index.
    if "Item_Identifier" in test_df.columns:
        submission = pd.DataFrame(
            {
                "Item_Identifier": test_df["Item_Identifier"],
                "Item_Outlet_Sales": test_pred,
            }
        )
    else:
        submission = pd.DataFrame(
            {
                "RowID": test_df.index,
                "Item_Outlet_Sales": test_pred,
            }
        )

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
