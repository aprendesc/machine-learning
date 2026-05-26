#!/bin/bash
# launcher.sh — lanza una fachada de la app con la config seleccionada

cd "$(dirname "$0")"
cd ..  # navegar de launchers/ a la app root
APP_DIR=$(pwd)
cd ../../..  # navegar al project root
PROJECT_ROOT=$(pwd)

PACKAGE="machinelearning"
APP="toy_models"

# --- Pantalla 1: Seleccionar fachada ---
echo ""
echo "=== Toy Models ==="
echo ""
echo "Fachadas disponibles:"
echo ""

i=1
for facade in "$APP_DIR"/*.py; do
  [ -f "$facade" ] || continue
  name=$(basename "$facade" .py)
  [[ "$name" == "__init__" ]] && continue
  facades[$i]="$name"
  echo "  [$i] $name"
  i=$((i + 1))
done
echo "  [0] Exit"
echo ""
read -p "Selecciona fachada: " f_option

[ "$f_option" = "0" ] && echo "Bye!" && exit 0
selected_facade=${facades[$f_option]}
[ -z "$selected_facade" ] && echo "Opción desconocida: $f_option" && exit 1

# --- Pantalla 2: Seleccionar configuración ---
echo ""
echo "Configuraciones disponibles:"
echo ""

i=1
for config in "$APP_DIR"/configs/*.yaml; do
  [ -f "$config" ] || continue
  name=$(basename "$config")
  configs[$i]="$name"
  echo "  [$i] $name"
  i=$((i + 1))
done
echo "  [0] Exit"
echo ""
read -p "Selecciona config: " c_option

[ "$c_option" = "0" ] && echo "Bye!" && exit 0
selected_config=${configs[$c_option]}
[ -z "$selected_config" ] && echo "Opción desconocida: $c_option" && exit 1

# --- Lanzar ---
echo ""
echo "=== Launching $selected_facade (config: $selected_config) ==="
uv run python -m ${PACKAGE}.apps.${APP}.${selected_facade} ${PACKAGE}/apps/${APP}/configs/${selected_config}

