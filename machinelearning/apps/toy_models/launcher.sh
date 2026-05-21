#!/bin/bash
cd "$(dirname "$0")"
cd ../../..  # navegar al project root

# Override: si se pasan argumentos, lanzar directo sin menú
if [ $# -gt 0 ]; then
  case $1 in
    titanic)  uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml "$2" ;;
    mnist)    uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/mnist.yaml "$2" ;;
    yahoo)    uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/yahoo_finance.yaml "$2" ;;
    *)        echo "Unknown model: $1" ;;
  esac
  exit 0
fi

# Sin argumentos: menú interactivo
echo ""
echo "=== Toy Models ==="
echo ""
echo "  [1] Smoke test (Titanic)"
echo "  [2] Train Titanic"
echo "  [3] Train MNIST"
echo "  [4] Train Yahoo Finance"
echo "  [5] Serve API (Titanic)"
echo "  [0] Exit"
echo ""
read -p "Select option: " option

case $option in
  1) uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml ;;
  2) uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml train ;;
  3) uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/mnist.yaml train ;;
  4) uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/yahoo_finance.yaml train ;;
  5) uv run python -m machinelearning.apps.toy_models.main machinelearning/apps/toy_models/configs/titanic.yaml serve ;;
  0) echo "Bye!" ;;
  *) echo "Unknown option: $option" ;;
esac

