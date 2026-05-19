#!/bin/bash
# build_project.sh — build del paquete y generación del .whl instalable

set -e
cd "$(dirname "$0")"
cd ..  # navegar al project root

echo "=== Limpiando builds anteriores ==="
rm -rf dist/ build/ *.egg-info

echo "=== Ejecutando tests ==="
uv run pytest tests/ -q

echo "=== Building package ==="
uv build

echo "=== Build completado ==="
ls -lh dist/*.whl
echo "✓ Wheel generado en dist/"

