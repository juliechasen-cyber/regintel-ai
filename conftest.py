"""Make the project importable in tests regardless of how pytest is invoked.

`python -m pytest` puts the working directory on sys.path, but the bare `pytest`
console script does not -- so tests that `import regintel` pass locally and fail
in CI. A root conftest.py is imported by pytest before collection, which fixes
the path for every invocation style and every pytest version.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
