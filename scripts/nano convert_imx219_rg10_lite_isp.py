#!/usr/bin/env python3
"""Compatibility wrapper for the cleaned IMX219 lite ISP script.

Use scripts/convert_imx219_rg10_lite_isp.py for new commands. This file is kept
because it existed during camera bring-up and may still be referenced in notes.
"""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_imx219_rg10_lite_isp import main


if __name__ == "__main__":
    main()
