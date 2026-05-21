"""machinelearning — Personal ML engineering framework."""

import os as _os
import platform as _platform

# ── OpenMP compatibility (macOS) ──────────────────────────────────────────────
# PyTorch, LightGBM, and XGBoost each bundle their own copy of libomp.
# On macOS, loading multiple copies in the same process triggers a fatal
# "pthread_mutex_init" error (OMP Error #179).
# Fix: preload the system libomp so all libraries share the same instance.
if _platform.system() == "Darwin":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import ctypes as _ctypes

    for _path in ("/opt/homebrew/opt/libomp/lib/libomp.dylib",
                  "/usr/local/opt/libomp/lib/libomp.dylib"):
        try:
            _ctypes.CDLL(_path)
            break
        except OSError:
            continue

