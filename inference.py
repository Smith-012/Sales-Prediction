"""
Sales model inference with schema validation and logging.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_input(df: pd.DataFrame) -> bool:
    """Validate input DataFrame has expected features."""
    if df.empty:
        logger.error("Input DataFrame is empty")
        return False
    logger.info(f"Input validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def load_model(model_path: Path):
    """Load trained model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def predict(model, df: pd.DataFrame) -> Dict[str, Any]:
    """Generate predictions on input DataFrame."""
    predictions = model.predict(df)
    logger.info(f"Generated predictions for {len(predictions)} samples")
    return {"predictions": predictions.tolist(), "mean": float(predictions.mean()), "std": float(predictions.std())}


def main():
    """CLI entrypoint for inference."""
    import argparse
    parser = argparse.ArgumentParser(description="Sales prediction inference")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    if not validate_input(df):
        raise ValueError("Input validation failed")
    
    model = load_model(args.model)
    result = predict(model, df)
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
