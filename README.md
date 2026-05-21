# machine-learning

Personal ML engineering framework — a unified Python library that combines functional domains and deployable applications under a single structured repository.

## Structure

```
machine_learning/
├── apps/          # Deployable applications (one folder per app)
└── modules/       # Shared reusable modules by functional domain
```

## Quick start

```bash
# Install in development mode
uv sync

# Run tests
uv run pytest tests/

# Build package
./scripts/build_project.sh

# Deploy an app
./scripts/deploy.sh [app_name]
```

## Apps

| App | Description | README |
|---|---|---|
| `toy_models` | Titanic · MNIST · Yahoo Finance — three end-to-end ML pipelines with REST serving | [→ docs](machinelearning/apps/toy_models/README.md) |

## Adding a model

The framework is **configuration-driven**: the pipeline code never changes between models. All variability lives in a YAML config file.

### 1. Create the config

Add `machinelearning/apps/toy_models/configs/<model_name>.yaml` with the fixed pipeline slots:

```yaml
checkpoint: 0                   # 0 = full run; N = resume from stage N
model_type: MyModel
model_id: MyModel_v1
models_path: models/toy_models

pipeline:
  etl:
    - class: <full.import.path.ETLStage>
      params: {}

  dataset_loader:
    class: eigenframework.modules.machine_learning.transforms.dataset_loader.DatasetLoader
    params:
      data_path: data
      dataset_name: <processed_dataset_name>
      load_mmf: false

  xy_split:
    class: eigenframework.modules.machine_learning.transforms.xy_split.XYSplit
    params:
      target_column: <target>

  spp:                          # static pre-processing (list of transformers)
    - class: eigenframework.modules.machine_learning.transforms.identity.Identity
      params: {}

  validation_split:
    class: eigenframework.modules.machine_learning.validation_split.basic_validation_split.BasicValidationSplit
    params:
      train_size: 0.8
      val_size: 0.1
      test_size: 0.1

  dpp:                          # dynamic pre-processing — fit on train, transform on val/test
    - class: eigenframework.modules.machine_learning.transforms.identity.Identity
      params: {}

  model:
    class: <full.import.path.MyModel>
    params: {}

  pos:
    - class: eigenframework.modules.machine_learning.transforms.identity.Identity
      params: {}

  metrics:
    class: eigenframework.modules.machine_learning.metrics.metrics_classification_regression.MetricsClassificationRegression
    params:
      mode: classification   # or regression

  model_loader:
    class: eigenframework.modules.machine_learning.loaders.basic_model_loader.BasicModelLoader
    params: {}

  logger:
    class: eigenframework.modules.machine_learning.utils.wandb_logging.WandbLogging
    params:
      project_name: toy_models
      run_name: MyModel_v1
```

> Use `eigenframework.modules.machine_learning.transforms.identity.Identity` for any slot that requires no processing.

### 2. Implement new stages (only if needed)

If the model requires custom ETL, transformations or model logic not already available in `eigenframework`, add the implementation under:

```
machinelearning/modules/stages/
├── data/          ← new ETL stages
├── transforms/    ← new SPP / DPP transformers
├── models/        ← new model classes
└── ...
```

Each stage is an **independent class** — no inheritance required. The pipeline resolves it by full import path from the YAML.

### 3. Run

```bash
uv run ml-toy-models --config <model_name>
```

---

## Adding an app

1. Create `machinelearning/apps/<app_name>/`
2. Add `__init__.py`, `main.py`, `configs/default.yaml`, `launcher.sh`
3. Register the entrypoint in `pyproject.toml` under `[project.scripts]`

