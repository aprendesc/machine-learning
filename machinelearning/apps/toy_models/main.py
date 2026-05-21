
import sys

import pandas as pd
from dotenv import load_dotenv

from eigenframework.modules.machine_learning.utils import ConfigLoader
from eigenframework.modules.machine_learning.pipelines.ml_pipeline import MLPipeline


class Main:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.pipeline = None
        self.config = None

    def initialize(self):
        self.config = ConfigLoader.load(self.config_path)
        self.pipeline = MLPipeline(self.config)
        self.pipeline.initialize()

    def etl(self):
        self.pipeline.etl()

    def train(self, **kwargs) -> pd.DataFrame:
        return self.pipeline.train()

    def predict(self, X, **kwargs) -> pd.DataFrame:
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        return self.pipeline.predict(X)

    def tune(self, n_trials: int = 50, **kwargs) -> tuple:
        tuning_config = self.config.get("tuning", {})
        n_trials = tuning_config.get("n_trials", n_trials)
        metric = tuning_config.get("metric", "accuracy")
        hparam_config = tuning_config.get("params", {})
        return self.pipeline.hparam_tuning(
            hparam_config=hparam_config,
            n_trials=n_trials,
            metric=metric,
            **kwargs,
        )


if __name__ == "__main__":
    load_dotenv()
    config = sys.argv[1] if len(sys.argv) > 1 else "machinelearning/apps/toy_models/configs/titanic.yaml"
    main = Main(config)
    main.initialize()

    if len(sys.argv) > 2:
        mode = sys.argv[2]
        if mode == "etl":
            main.etl()
        elif mode == "train":
            print(main.train())
        elif mode == "predict":
            print(main.predict(None))
        elif mode == "tune":
            best_params, best_value = main.tune()
            print(f"Best params: {best_params}\nBest value: {best_value}")
        elif mode == "serve":
            from eigenframework.modules.core.api_server import ApiServer
            server = ApiServer(host="0.0.0.0", port=8000, title="Toy Models API")
            server.serve_main(main)
            server.start()
    else:
        print("=== Smoke test ===")
        main.etl()
        result = main.train()
        print(f"✓ smoke test completado: {result}")

