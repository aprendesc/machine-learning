# Toy Models

Three end-to-end ML pipelines validated on classic problems — tabular classification, image classification, and time-series regression — all driven by a single YAML config.

| Model | Problem | Algorithm |
|---|---|---|
| `BlendingEnsemble_v1` | Titanic survival (binary classification) | FFNN + LightGBM + XGBoost → meta-LightGBM |
| `CNN_v1` | MNIST digits (multiclass classification) | Convolutional Neural Network (PyTorch) |
| `ARIMA_v1` | AAPL stock price (regression) | ARIMA via statsmodels |

---

## Setup

```bash
# From project root
uv sync
```

---

## Interactive launcher

```bash
./machinelearning/apps/toy_models/launcher.sh
```

```
=== Toy Models ===

  [1] Smoke test (Titanic)
  [2] Train Titanic
  [3] Train MNIST
  [4] Train Yahoo Finance
  [5] Serve API (Titanic)
  [0] Exit
```

---

## CLI usage

```bash
# ETL only
uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml etl

# Train
uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml train

# Serve (see section below)
uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml serve

# Hyperparameter tuning
uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml tune
```

---

## Serving a model as a REST API

Any trained model can be exposed as a REST endpoint in one command.
The server loads the persisted `pipeline.pkl` — no re-training needed.

### 1 — Train the model (first time only)

```bash
uv run python -m machinelearning.apps.toy_models.main \
  machinelearning/apps/toy_models/configs/titanic.yaml train
```

The trained pipeline is saved to `models/toy_models/BlendingEnsemble_v1/pipeline.pkl`.

### 2 — Start the server

```bash
uv run python -m machinelearning.apps.toy_models.main \
  machinelearning/apps/toy_models/configs/titanic.yaml serve
```

Or via the launcher:

```bash
./machinelearning/apps/toy_models/launcher.sh titanic serve
```

The server starts on `0.0.0.0:8000` by default.

```
INFO:     Started server process
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3 — Available endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns registered endpoints |
| `POST` | `/predict` | Run inference |
| `POST` | `/train` | Re-train the model |
| `POST` | `/etl` | Re-run the ETL pipeline |
| `POST` | `/tune` | Run hyperparameter search |

### 4 — Call `/health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "endpoints": ["etl", "predict", "train", "tune"]
}
```

### 5 — Call `/predict`

Requests follow the format `{"args": [...], "kwargs": {...}}`.
`predict` receives a dict of column → list-of-values (same format as `DataFrame.to_dict(orient="list")`).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "args": [{
      "Pclass":    [3, 1],
      "Sex":       [1, 0],
      "Age":       [-0.5, 0.8],
      "SibSp":     [0.0, 0.0],
      "Parch":     [0.0, 0.0],
      "Fare":      [-0.5, 1.2],
      "Embarked":  [2, 0],
      "Title":     [1, 2],
      "FamilySize":[0.0, 0.0],
      "IsAlone":   [1.0, 1.0],
      "Deck":      [0, 1],
      "FareBin":   [0.0, 2.0],
      "AgeBin":    [1.0, 2.0]
    }],
    "kwargs": {}
  }'
```

```json
{
  "success": true,
  "result": [
    {"predictions": 0},
    {"predictions": 1}
  ],
  "method": "predict"
}
```

> **Note**: input features must be already encoded and scaled (post-DPP). The pipeline exposes `predict` after all preprocessing transformers — it expects the same feature space it saw during training.

### Python client

```python
from eigenframework.modules.core.api_server import ApiServer

client = ApiServer(remote_url="http://localhost:8000")

result = client.call("predict", {
    "Pclass": [3], "Sex": [1], "Age": [-0.5], "SibSp": [0.0],
    "Parch": [0.0], "Fare": [-0.5], "Embarked": [2], "Title": [1],
    "FamilySize": [0.0], "IsAlone": [1.0], "Deck": [0],
    "FareBin": [0.0], "AgeBin": [1.0],
})
print(result)  # [{"predictions": 0}]
```

---

## Switching models

Change the config path to serve a different model:

```bash
# MNIST CNN
uv run python -m machinelearning.apps.toy_models.main \
  machinelearning/apps/toy_models/configs/mnist.yaml serve

# Yahoo Finance ARIMA
uv run python -m machinelearning.apps.toy_models.main \
  machinelearning/apps/toy_models/configs/yahoo_finance.yaml serve
```

---

## Config reference

Each YAML defines a complete experiment. Top-level keys:

| Key | Description |
|---|---|
| `model_id` | Unique identifier — determines where the model is saved/loaded |
| `models_path` | Root directory for persisted pipelines |
| `checkpoint` | `0` = full run, `1–3` = resume from intermediate stage |
| `pipeline` | Stage definitions: `etl`, `dataset_loader`, `xy_split`, `spp`, `validation_split`, `dpp`, `model`, `pos`, `metrics`, `model_loader`, `logger` |
| `tuning` | Optuna hyperparameter search config (n_trials, metric, param ranges) |

