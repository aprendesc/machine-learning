"""Toy Models — Backend Facade

Entry point for the Toy Models application.
Delegates all logic to the ML orchestrator.
"""

import sys


class Main:
    """Facade del backend — delega toda la lógica en orquestadores."""

    def __init__(self, config_path: str = "machinelearning/apps/toy_models/configs/titanic.yaml"):
        self.config_path = config_path
        self.orchestrator = None

    def initialize(self, **kwargs) -> None:
        from machinelearning.apps.toy_models.orchestrators.ml_orchestrator import MLOrchestrator
        self.orchestrator = MLOrchestrator(self.config_path)
        self.orchestrator.initialize()

    def etl(self, **kwargs):
        return self.orchestrator.etl(**kwargs)

    def train(self, **kwargs):
        return self.orchestrator.train(**kwargs)

    def predict(self, X, **kwargs):
        import pandas as pd
        if isinstance(X, (dict, list)):
            X = pd.DataFrame(X)
        return self.orchestrator.predict(X, **kwargs)

    def tune(self, **kwargs) -> tuple:
        return self.orchestrator.tune(**kwargs)


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not required in cluster environments
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
