#!/bin/bash
# deploy.sh — despliega los endpoints del proyecto a partir de los entrypoints

set -e
cd "$(dirname "$0")"
cd ..  # navegar al project root

# Cargar variables de entorno
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

# Override: si se pasa una app específica, desplegar solo esa
if [ $# -gt 0 ]; then
  APP=$1
  echo "=== Deploying $APP (${HOST}:${PORT}) ==="
  uv run python -m machinelearning.apps.${APP}.main serve
  exit 0
fi

# Sin argumentos: menú interactivo con todas las apps disponibles
echo ""
echo "=== Deploy — Endpoints disponibles ==="
echo ""

i=1
declare -a apps
for app_dir in machinelearning/apps/*/; do
  app_name=$(basename "$app_dir")
  echo "  [$i] $app_name"
  apps[$i]=$app_name
  i=$((i + 1))
done
echo "  [0] Exit"
echo ""
read -p "Select app to deploy: " option

if [ "$option" = "0" ]; then
  echo "Bye!"
  exit 0
fi

selected=${apps[$option]}
if [ -z "$selected" ]; then
  echo "Opción desconocida: $option"
  exit 1
fi

echo "=== Deploying $selected (${HOST}:${PORT}) ==="
uv run python -m machinelearning.apps.${selected}.main serve

