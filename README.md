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

## Adding an app

1. Create `machine_learning/apps/<app_name>/`
2. Add `__init__.py`, `main.py`, `configs/default.yaml`, `launcher.sh`
3. Register the entrypoint in `pyproject.toml` under `[project.scripts]`

