"""Where the API keys live: the system keyring, and nowhere else. One key per provider (Claude, Groq), entered in
the Assistant settings dialog. Never in an environment variable the app reads, a project file, the settings file or a
file of its own; never in a log or a chat.

Without the optional keyring package (or with only keyring's "fail" backend, or with DEXVANTAGE_NO_KEYRING set, as
the tests and the self-test set it) there is nowhere to keep a key: the dialog says so and how to install it.

Tests never touch a real keyring: they set DEXVANTAGE_NO_KEYRING and, where a test needs a stored key, install the
in-memory backend below with set_backend(MemoryKeyring()).

Ported from Vantage assistant/secrets.py, without its environment-variable and private-file fallbacks (the
specification keeps keys in the OS keyring only)."""
from __future__ import annotations

import os

from . import providers as PR

SERVICE = 'DexVantage'
ACCOUNT = {PR.CLAUDE: 'anthropic-api-key', PR.GROQ: 'groq-api-key'}
NO_KEYRING_ENV = 'DEXVANTAGE_NO_KEYRING'

_override = None                # a backend object set by set_backend (tests); None = the system keyring


class KeyStoreError(Exception):
    """A key could not be stored or removed; the message says why in plain words."""


class MemoryKeyring:
    """A keyring that keeps passwords in this process only (the tests' backend)."""

    def __init__(self):
        self.store: dict = {}

    def get_password(self, service, account):
        return self.store.get((service, account))

    def set_password(self, service, account, password):
        self.store[(service, account)] = password

    def delete_password(self, service, account):
        self.store.pop((service, account), None)


def set_backend(backend):
    """Use this backend instead of the system keyring (None goes back to the system keyring)."""
    global _override
    _override = backend


def _account(provider) -> str:
    if provider not in ACCOUNT:
        raise KeyError(f'unknown provider {provider!r}')
    return ACCOUNT[provider]


def backend():
    """The keyring to use, or None when there is none."""
    if _override is not None:
        return _override
    if os.environ.get(NO_KEYRING_ENV):
        return None
    try:
        import keyring
        from keyring.backends import fail
        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
        return keyring
    except Exception:                       # noqa: BLE001 — no package, a broken D-Bus session: no keyring
        return None


def available() -> bool:
    return backend() is not None


def why_unavailable() -> str:
    """A plain sentence: why keys cannot be stored here ('' when they can)."""
    if available():
        return ''
    if os.environ.get(NO_KEYRING_ENV) and _override is None:
        return 'The system keyring is switched off for this run (DEXVANTAGE_NO_KEYRING is set).'
    try:
        import keyring  # noqa: F401
    except ImportError:
        return PR.INSTALL['keyring']
    return ('No system keyring service answered (on Linux: GNOME Keyring, KWallet or another Secret Service). Start '
            'one, then open the settings again.')


def load(provider) -> str | None:
    kr = backend()
    if kr is None:
        return None
    try:
        k = (kr.get_password(SERVICE, _account(provider)) or '').strip()
    except KeyError:
        raise
    except Exception:                       # noqa: BLE001 — a locked keyring reads as no key
        return None
    return k or None


def save(provider, key: str) -> None:
    key = (key or '').strip()
    if not key:
        raise KeyStoreError('The key is empty: paste the whole key.')
    kr = backend()
    if kr is None:
        raise KeyStoreError(why_unavailable())
    try:
        kr.set_password(SERVICE, _account(provider), key)
    except Exception as e:                  # noqa: BLE001
        raise KeyStoreError(f'The system keyring refused to store the key ({type(e).__name__}).') from None


def delete(provider) -> None:
    kr = backend()
    if kr is None:
        return
    try:
        kr.delete_password(SERVICE, _account(provider))
    except Exception:                       # noqa: BLE001 — nothing stored is not an error
        pass


def mask(key: str | None) -> str:
    """The first and last four characters, for the settings dialog."""
    if not key:
        return ''
    return f'{key[:4]}…{key[-4:]}' if len(key) > 12 else '…' + key[-2:]
