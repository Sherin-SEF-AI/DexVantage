"""Whether PyTorch sees a CUDA device, asked in a child process (the Splat views page, ui/pages/splat.py), so that the
window never imports torch only to find out:

    python -m dexvantage.core.cuda_probe        exit 0: a CUDA device; 3: none, or PyTorch is not installed

Importing torch in the window's own process would add about half a gigabyte to it and hold the GIL long enough to
stall the window (535 MB and 155 ms, measured with torch 2.14.1 on 2026-11-08), whether or not the page is used.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        return 0 if torch.cuda.is_available() else 3
    except Exception:                                    # noqa: BLE001 — not installed or broken: no CUDA here
        return 3


if __name__ == '__main__':
    sys.exit(main())
