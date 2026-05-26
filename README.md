# Sales Prediction

## Portfolio Summary
This version upgrades the sales regression task into a more production-style workflow. It compares models, tunes hyperparameters, and saves the final artifact and evaluation summary.

## What Makes It Portfolio-Ready

- Multiple regressors with cross-validation
- Hyperparameter tuning optimized for RMSE
- Persisted model and metrics for reproducibility
- Clear business-facing regression metrics

## 💰 What This Project Does

This project predicts sales revenue based on advertising spend across multiple channels using machine learning. Given budget allocations for TV, radio, and newspaper advertising, the model estimates expected total sales. It demonstrates:

- **Marketing mix modeling**: Understanding how different advertising channels contribute to revenue
- **Feature interaction effects**: TV and radio spending interact to amplify sales impact
- **Regression for business metrics**: Predicting continuous revenue values in dollars
- **Hyperparameter optimization**: Tuning models to minimize prediction error (RMSE)
- **Business-friendly metrics**: Using MAE and RMSE alongside R² for stakeholder communication
- **Interactive forecasting**: Web app for sales predictions given advertising budgets

### Problem Statement

Marketing teams face the question: "How much should we spend on each channel?" This project builds a predictive model that quantifies the relationship between advertising investment and sales revenue. Understanding the impact of different channels helps optimize marketing budgets and predict ROI.

### Key Features Used

- **TV Advertising**: Budget spent on TV commercials in thousands of dollars
- **Radio Advertising**: Budget spent on radio spots in thousands of dollars
- **Newspaper Advertising**: Budget spent on print media in thousands of dollars

These budgets are combined to predict total sales in thousands of dollars. The dataset contains marketing data across multiple markets, capturing natural variation in spending and outcomes.

### Target Variable

- **Sales**: Total revenue in thousands of dollars (continuous, typically 5-25K range)

### Models Trained

1. **Random Forest Regressor**: Captures non-linear channel interactions
2. **Gradient Boosting Regressor**: Sequential learning to minimize residual errors

## Tech Stack

- Python 3
- Pandas and NumPy
- Scikit-learn
- Joblib

## Dataset
Place the CSV file here:

- `data/sales.csv`

Target column support:

- `sales`
- `Sales`
- `revenue`
- `target`

## Installation

### Development
```bash
pip install -r requirements.txt
# or with pinned versions
pip install -r requirements-lock.txt
```

### Production (via pip)
```bash
pip install -e .
```

This registers the CLI command `train-sales` globally.

## Training

```bash
# Using the CLI (after install -e .)
train-sales --data data/sales.csv --out artifacts

# Or directly
python main.py --data data/sales.csv --out artifacts
```

Outputs:
- `artifacts/model.joblib` – trained regressor
- `artifacts/metrics.txt` – evaluation metrics (MAE, RMSE, R²)

## Inference

### Batch Prediction
```bash
python inference.py --model artifacts/model.joblib --input new_sales.csv --output predictions.json
```

### Programmatic Usage
```python
from inference import load_model, predict
import pandas as pd

model = load_model("artifacts/model.joblib")
df = pd.read_csv("new_sales.csv")
result = predict(model, df)
print(result)  # {"predictions": [...], "mean": ..., "std": ...}
```

## Interactive Web App (Streamlit)

### Launch the Web App
```bash
# Install Streamlit and SHAP first
pip install streamlit shap

# Run the app
streamlit run app.py
```

The app provides:
- **Interactive predictions**: Input sales factors and get revenue forecasts
- **SHAP explanations**: Understand which factors most influence sales
- **Confidence intervals**: See estimated sales range
- **Dataset overview**: Explore regional and seasonal sales patterns

**Browser**: Opens automatically at `http://localhost:8501`

## Jupyter Notebooks

Explore the analysis and training workflow:

```bash
jupyter notebook notebooks/
```

**Available notebooks:**
- `01_eda.ipynb` - Exploratory Data Analysis (advertising impact, regional analysis, seasonal trends)
- Add more notebooks for detailed analysis

## Docker

```bash
# Build
docker build -t sales-predictor:latest .

# Train in container
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts sales-predictor:latest

# Predict
docker run --rm -v $(pwd):/app sales-predictor:latest python inference.py --input /app/new_sales.csv
```

## Testing

```bash
python -m unittest discover -v tests
```

## Production Guide

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

### Local Setup

```bash
# Clone the repo
git clone <repo-url>
cd Portfolio-Task-4-Sales-Production-Style

# Install dependencies (locked versions recommended for production)
pip install -r requirements-lock.txt
# or
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Production Monitoring

**Key Metrics:**
- **Regression Error**: Monitor MAE and RMSE (see artifacts/metrics.txt)
- **R² Score**: Track model fit on held-out validation data
- **Input Validation**: Ensure all required sales features present
- **Prediction Distribution**: Monitor mean and variance of predictions
- **Inference Latency**: Track prediction time per batch

**Alerts:**
- Set alert if MAE increases > 10% from baseline
- Alert if R² drops > 5% from baseline
- Alert if > 5% of requests fail validation
- Alert if average latency exceeds 50ms per sample

**Logging:**
All inference runs log validation and prediction stats:
```
2026-05-26 10:30:45 - INFO - Input validation passed: 200 rows, 10 columns
2026-05-26 10:30:46 - INFO - Loaded model from artifacts/model.joblib
2026-05-26 10:30:47 - INFO - Generated predictions for 200 samples
```

### CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml` runs:
- Dependency installation
- Unit tests (`python -m unittest discover`)
- On every push/PR

### Model Retraining

To retrain with new data:
```bash
python main.py --data data/sales_updated.csv --out artifacts
```

New artifacts will overwrite previous `model.joblib` and `metrics.txt`.

### Maintenance

- Keep `requirements-lock.txt` synchronized with production environment
- Test any dependency upgrades in staging first
- Monitor for seasonal trends in prediction accuracy
- Archive old models before retraining

## Project Flow

1. Load the sales dataset.
2. Prepare numeric and categorical preprocessing.
3. Compare candidate regressors (Random Forest, Gradient Boosting) with cross-validation.
4. Choose the model with the strongest RMSE performance.
5. Save the model and metrics.

## 📊 Limitations

### Current Constraints

1. **Limited Features**: Only uses 3 advertising channels (TV, Radio, Newspaper); ignores digital/social media (the fastest growing channels)
2. **No Temporal Dynamics**: Treats all time periods equally; doesn't capture seasonal patterns or trends
3. **No External Factors**: Ignores economic conditions, competition, product quality, pricing, or distribution
4. **Geographic Aggregation**: Data may be aggregated across regions, hiding regional differences
5. **Correlation vs. Causation**: Cannot prove advertising causes sales; could be reverse causality or confounders
6. **Linear Assumption Limitations**: Real marketing impact is non-linear (threshold effects, saturation)
7. **No Interaction With Competition**: Competitor advertising and market dynamics not included

### Model Trade-offs

- **Interpretability Loss**: Tree-based models are harder to explain than linear regression
- **Overfitting Risk**: Ensemble methods may memorize training patterns that don't generalize
- **Feature Engineering Gap**: Limited to provided features; domain knowledge not fully leveraged

## 🚀 Future Scope

### Short-term Enhancements (Weeks)

1. **Feature Engineering**: Create spending ratios (TV:Radio, Total:TV), squared terms for diminishing returns
2. **Interaction Terms**: Explicitly model TV-Radio synergy, total budget effects
3. **Baseline Models**: Add Linear Regression for comparison (interpretable lower bound)
4. **Residual Analysis**: Analyze prediction errors to identify systematic biases
5. **Sensitivity Analysis**: Show how sales change with 1% increase in each channel
6. **Prediction Intervals**: Provide confidence ranges instead of point estimates

### Medium-term Extensions (Months)

1. **Time-Series Modeling**: Include temporal trends, seasonality, and lag effects
2. **Regional Analysis**: Build separate models for different geographic markets
3. **Elasticity Estimation**: Calculate how 1% spending change impacts sales for each channel
4. **Budget Optimization**: Recommend optimal spending allocation given total budget constraint
5. **API Development**: Build FastAPI for real-time sales forecasting given budgets
6. **Dashboard**: Create interactive Streamlit dashboard with scenario planning

### Long-term Vision (Years)

1. **Causal Inference**: Use causal models (DAGs, instrumental variables) to prove causation
2. **Reinforcement Learning**: Train policy to maximize ROI by learning from repeated decisions
3. **Multimodal Data**: Incorporate customer sentiment from social media, competitor actions
4. **Customer-Level Modeling**: Build models for individual customers instead of aggregated markets
5. **Attribution Modeling**: Determine which advertising touchpoint deserves credit for each sale
6. **Supply-Side Integration**: Link advertising spend to actual customer acquisition costs

### Research Opportunities

- Investigate diminishing returns effect: does doubling ad spend double sales?
- Compare TV, Radio, and Newspaper effectiveness across different demographic groups
- Analyze time lag between advertising exposure and actual sales conversion
- Study interaction effects between channels (does TV amplify Radio effectiveness?)
- Predict optimal budget allocation to maximize sales subject to spending constraint

## Project Structure

```
.
├── main.py              # Training script
├── inference.py         # Inference with validation
├── app.py               # Streamlit interactive web app
├── pyproject.toml       # Package metadata
├── Dockerfile           # Container definition
├── Makefile             # Convenience commands
├── requirements.txt     # Python dependencies
├── requirements-lock.txt # Pinned versions
├── LICENSE              # MIT
├── README.md            # This file
├── .gitignore           # Git ignore patterns
├── .github/workflows/ci.yml # GitHub Actions
├── data/
│   └── sales.csv        # Training dataset
├── notebooks/           # Jupyter notebooks
│   └── 01_eda.ipynb     # Exploratory Data Analysis
├── artifacts/           # Trained model & metrics
└── tests/
    └── test_quick.py    # Smoke tests
```

## Tech Stack

- **Python 3.11+**
- **Pandas & NumPy** – data manipulation
- **Scikit-learn** – ML models and preprocessing
- **Joblib** – model persistence
- **Matplotlib & Seaborn** – visualization
