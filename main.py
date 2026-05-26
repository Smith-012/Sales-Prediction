import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def infer_target(df: pd.DataFrame) -> str:
    for col in ["sales", "Sales", "revenue", "target"]:
        if col in df.columns:
            return col
    raise ValueError("No target found. Use sales, Sales, revenue, or target")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main(data_path: Path, out_dir: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    target = infer_target(df)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(X_train)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "rf": (
            RandomForestRegressor(random_state=42),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [None, 10, 20],
            },
        ),
        "gbr": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    }

    best_name = None
    best_search = None

    for name, (model, grid) in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name = name
            best_search = search

    preds = best_search.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_search.best_estimator_, out_dir / "model.joblib")
    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_model: {best_name}\n")
        f.write(f"cv_neg_rmse: {best_search.best_score_}\n")
        f.write(f"test_mae: {mae}\n")
        f.write(f"test_rmse: {rmse}\n")
        f.write(f"test_r2: {r2}\n")

    print("Sales portfolio pipeline complete")
    print(f"Best model: {best_name}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Portfolio-grade sales regression")
    parser.add_argument("--data", type=Path, default=Path("data/sales.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    main(args.data, args.out)


if __name__ == "__main__":
    cli()
