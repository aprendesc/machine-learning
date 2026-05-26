"""Toy Models — ML Orchestrator

Thin wrapper around eigenframework's MLPipeline.
Receives YAML config path, instantiates the pipeline, and delegates all operations.
"""

from eigenframework.modules.machine_learning.utils import ConfigLoader
from eigenframework.modules.machine_learning.pipelines.ml_pipeline import MLPipeline


class MLOrchestrator:
    """Orchestrates the ML pipeline for Toy Models.

    Composes eigenframework's MLPipeline with project-specific configuration.
    All domain logic lives in the pipeline and its configured modules.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.pipeline = None
        self.config = None

    def initialize(self) -> None:
        """Load config and initialize the ML pipeline."""
        self.config = ConfigLoader.load(self.config_path)
        self.pipeline = MLPipeline(self.config)
        self.pipeline.initialize()

    def etl(self, **kwargs):
        """Execute data acquisition."""
        return self.pipeline.etl(**kwargs)

    def train(self, **kwargs):
        """Execute full training pipeline."""
        return self.pipeline.train(**kwargs)

    def predict(self, X, **kwargs):
        """Run inference with loaded model."""
        return self.pipeline.predict(X, **kwargs)

    def tune(self, **kwargs) -> tuple:
        """Run hyperparameter tuning."""
        tuning_config = self.config.get("tuning", {})
        n_trials = tuning_config.get("n_trials", 50)
        metric = tuning_config.get("metric", "accuracy")
        hparam_config = tuning_config.get("params", {})
        return self.pipeline.hparam_tuning(
            hparam_config=hparam_config,
            n_trials=n_trials,
            metric=metric,
            **kwargs,
        )

