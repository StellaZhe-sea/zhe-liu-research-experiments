"""Compatibility entry point for the canonical modular implementation.

The model is defined only in helmholtz_multigrade/models.py.
"""

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent / "helmholtz_multigrade"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from main import main


if __name__ == "__main__":
    main()
