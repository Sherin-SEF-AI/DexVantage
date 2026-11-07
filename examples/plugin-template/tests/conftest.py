"""Throw-away config, data and cache folders: the plugin's tests never touch the user's DexVantage settings."""
import atexit
import os
import shutil
import tempfile

for _k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
    os.environ[_k] = tempfile.mkdtemp(prefix=f'dv-plugin-test-{_k[4:8].lower()}-')
    atexit.register(shutil.rmtree, os.environ[_k], True)    # thrown away when the session ends
