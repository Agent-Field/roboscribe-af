"""Thin entry shim. Real logic lives in src/roboscribe_af/app.py.

This file makes the project runnable as `python main.py` from the repo root,
matching the pattern used by other AgentField examples (pr-af, sec-af).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from roboscribe_af.app import main  # noqa: E402

if __name__ == "__main__":
    main()
