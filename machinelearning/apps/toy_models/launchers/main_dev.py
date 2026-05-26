# %% [0] Bootstrap
from dotenv import load_dotenv
load_dotenv()

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
import pandas as pd
# ── Config ──
cfg = {"config_path": "machinelearning/apps/toy_models/configs/titanic.yaml"}
# ── Sample — post-SPP features (after feature engineering + selection, before DPP encoding/scaling) ──
# Example: Rose (1st class, female, 17yo) → expected survival = 1
X_sample = pd.DataFrame([{
    "Pclass": 1, "Sex": "female", "Age": 17.0, "SibSp": 1, "Parch": 2,
    "Fare": 263.0, "Embarked": "S", "Title": "Miss", "FamilySize": 4,
    "IsAlone": 0, "Deck": "C", "FareBin": 3, "AgeBin": 1,
}])
# ── Execute ──
main = Main(**cfg)
main.initialize()
preds = main.predict(X_sample)
# ── Inspect ──
print(f"predictions (0=no survived, 1=survived):\n{preds}")

