#!/usr/bin/env python3
"""Zero-dependency test runner.

``pytest tests/`` is the normal way to run this suite. This runner exists so the
tests also execute on a bare Python install (no pytest), which keeps the
"nothing is claimed to work unless tested" rule enforceable anywhere.

Usage:
    python run_tests.py            # run everything
    python run_tests.py chunking   # run files matching a substring
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from typing import Callable

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(ROOT, "tests")


def _load_module(path: str):
    name = "t_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    sys.path.insert(0, ROOT)
    pattern = argv[0] if argv else ""

    files = sorted(
        os.path.join(TESTS_DIR, f)
        for f in os.listdir(TESTS_DIR)
        if f.startswith("test_") and f.endswith(".py") and pattern in f
    )

    passed = failed = 0
    failures: list[tuple[str, str]] = []

    for path in files:
        rel = os.path.relpath(path, ROOT)
        try:
            module = _load_module(path)
        except Exception:
            failed += 1
            failures.append((rel, traceback.format_exc()))
            print(f"  ERROR importing {rel}")
            continue

        tests: list[tuple[str, Callable]] = [
            (n, getattr(module, n))
            for n in sorted(dir(module))
            if n.startswith("test_") and callable(getattr(module, n))
        ]
        print(f"\n{rel}  ({len(tests)} tests)")
        for name, fn in tests:
            try:
                fn()
            except Exception:
                failed += 1
                failures.append((f"{rel}::{name}", traceback.format_exc()))
                print(f"  FAIL  {name}")
            else:
                passed += 1
                print(f"  pass  {name}")

    print("\n" + "=" * 60)
    print(f"{passed} passed, {failed} failed")
    if failures:
        print("=" * 60)
        for label, tb in failures:
            print(f"\n--- {label} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
