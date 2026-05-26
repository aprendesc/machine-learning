# %% [0] Bootstrap
from machinelearning.apps.toy_models.main import Main

# %% [1] initialize
# ── Config ──
cfg = {"config_path": "machinelearning/apps/toy_models/configs/titanic.yaml"}
# ── Execute ──
main = Main(**cfg)
main.initialize()
# ── Inspect ──
print(f"✓ initialized | orchestrator: {type(main.orchestrator).__name__}")

# %% [2] etl
# ── Config ──
cfg = {"config_path": "machinelearning/apps/toy_models/configs/titanic.yaml"}
# ── Execute ──
main = Main(**cfg)
main.initialize()
main.etl()
# ── Inspect ──
print("✓ etl completado")

# %% [3] train
# ── Config ──
cfg = {"config_path": "machinelearning/apps/toy_models/configs/titanic.yaml"}
# ── Execute ──
main = Main(**cfg)
main.initialize()
scores = main.train()
# ── Inspect ──
print(f"scores:\n{scores}")

# %% [4] predict
# ── Config ──
cfg = {"config_path": "machinelearning/apps/toy_models/configs/titanic.yaml"}
# ── Execute ──
main = Main(**cfg)
main.initialize()
preds = main.predict(None)
# ── Inspect ──
print(f"predictions:\n{preds}")

