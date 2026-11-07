"""Fewer malloc arenas, so memory freed by worker threads goes back to the system.

glibc gives each thread that allocates its own malloc arena (up to eight per core: 192 on a 24-core machine). Memory
freed in those arenas stays mapped as fragments, so a long-running process with worker threads (the window's studies,
the validation suite, the test suite) grows although nothing in it is alive any more: measured on 2026-11-07, seven
test modules left 1.64 GB resident with the default arenas and 0.84 GB with two. limit_arenas() calls glibc's
mallopt(M_ARENA_MAX, n) before the threads start; MALLOC_ARENA_MAX in the environment, when set, takes precedence.

Known limits: glibc only (elsewhere this does nothing); arenas created before the call stay; threads then share the
arenas and may wait on each other's allocation locks (no difference was measurable in the studies, which allocate in
large NumPy blocks).
"""
from __future__ import annotations

import os
import sys

M_ARENA_MAX = -8                 # glibc malloc.h
DEFAULT_ARENAS = 2


def limit_arenas(n: int = DEFAULT_ARENAS) -> bool:
    """Limit glibc's malloc arenas to n; True when it was set here."""
    if not sys.platform.startswith('linux') or os.environ.get('MALLOC_ARENA_MAX'):
        return False
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        return bool(libc.mallopt(ctypes.c_int(M_ARENA_MAX), ctypes.c_int(int(n))))
    except (OSError, AttributeError):
        return False
