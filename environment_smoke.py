"""Import and in-memory serialization smoke test. No fitting or explanations."""

import argparse
from importlib import import_module
import io
import json
from pathlib import Path
import platform
import sys
import sysconfig

from colab_preflight import runtime_inventory


MODULES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
    "lightgbm": "lightgbm", "shap": "shap", "numba": "numba", "llvmlite": "llvmlite",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "joblib": "joblib",
}
ROOT = Path(__file__).resolve().parent


def platform_supported():
    libc, version = platform.libc_ver()
    return (sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 13)
            and platform.system() == "Linux" and platform.machine() == "x86_64"
            and not sysconfig.get_config_var("Py_GIL_DISABLED")
            and libc == "glibc" and tuple(map(int, version.split("."))) >= (2, 28))


def serialization_smoke():
    import joblib
    import numpy as np
    from sklearn.preprocessing import FunctionTransformer
    from ids_utils import log_clip

    stream = io.BytesIO()
    joblib.dump(FunctionTransformer(log_clip), stream)
    stream.seek(0)
    transformer = joblib.load(stream)
    if transformer.func is not log_clip:
        raise RuntimeError("Serialized project function did not round-trip.")
    np.testing.assert_allclose(transformer.transform(np.array([-1.0, 0.0, 1.0])),
                               [0.0, 0.0, np.log(2.0)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true", help="Inventory an existing environment without qualifying it")
    parser.add_argument("--without-shap", action="store_true",
                        help="Verify SHAP's installed pin without importing it for Stage A")
    args = parser.parse_args()
    if not args.diagnose and not platform_supported():
        parser.exit(1, "Expected regular CPython 3.13 on Linux x86_64 with glibc >= 2.28.\n")

    inventory = runtime_inventory(ROOT / "environment/colab-py313-v1.in")
    errors = {}
    for package, module in MODULES.items():
        if args.without_shap and package == "shap":
            continue
        try:
            import_module(module)
        except Exception as exc:
            errors[package] = f"{type(exc).__name__}: {exc}"
    backend = str(import_module("matplotlib").get_backend()) if "matplotlib" not in errors else None
    serialization = "passed"
    try:
        serialization_smoke()
    except Exception as exc:
        serialization = f"{type(exc).__name__}: {exc}"
    mismatches = {name: spec for name, spec in inventory["packages"].items() if not spec["matches"]}
    print(json.dumps({"python": platform.python_version(), "machine": platform.machine(),
                      "libc": platform.libc_ver(), "platform_supported": platform_supported(),
                      "pin_mismatches": mismatches, "import_errors": errors,
                      "matplotlib_backend": backend, "serialization": serialization, "diagnostic_only": args.diagnose}, indent=2))
    if not args.diagnose and (mismatches or errors or serialization != "passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
