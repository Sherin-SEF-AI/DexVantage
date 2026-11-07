"""The providers the assistant can use, chosen in the Assistant settings: Claude (Anthropic's Messages API through the
official anthropic SDK, claude.py) and Groq (an OpenAI-compatible HTTPS API, client.py), plus the fake provider every
test uses (fake.py: scripted replies and tool calls, nothing leaves the machine). Each real provider has its own key
(secrets.py, the system keyring only), its own model list and its own usage line.

Ported from Vantage assistant/providers.py: the Claude model list gains Claude Haiku 5.5; the install message names
DexVantage's optional extra.

Prices: Claude's are constants with their source and date (Anthropic's model table as quoted in the Claude API
reference, cached 2026-10-06, read 2026-11-07). A price that table does not state (Claude Haiku 5.5's cache prices)
is None, and a conversation that used it shows its tokens without a cost. Groq's prices are not bundled: no source was
read for them, so a Groq conversation shows tokens only."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

CLAUDE, GROQ, FAKE = 'claude', 'groq', 'fake'
PROVIDERS = (CLAUDE, GROQ)                      # what the settings offer; FAKE is for the tests and the self-test
LABEL = {CLAUDE: 'Claude', GROQ: 'Groq', FAKE: 'Fake provider (tests)'}
KEYS_URL = {CLAUDE: 'https://platform.claude.com/', GROQ: 'https://console.groq.com/keys'}
KEY_HINT = {CLAUDE: 'Paste your Anthropic API key', GROQ: 'Paste your Groq API key'}
INSTALL = {
    CLAUDE: ('The Claude provider needs Anthropic\'s Python SDK, which this installation does not have. In the source '
             'folder: pip install ".[assistant]" (it installs anthropic 2.6.0 and keyring 25.7.0).'),
    'keyring': ('Keys are kept only in the system keyring, and this installation has no keyring package. In the source '
                'folder: pip install ".[assistant]" (it installs keyring 25.7.0).'),
}


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    note: str = ''
    vision: bool = False
    rank: int = 50                  # lower first in the picker
    fallbacks: bool = False         # Claude: the server-side refusal fallback ("default") is offered for this model


CLAUDE_DEFAULT = 'claude-opus-5-5'
# Model ids from Anthropic's model table (Claude API reference, cached 2026-10-06, read 2026-11-07). Server-side
# fallback on a refusal: Opus 5.5 and Sonnet 5.5 take fallbacks: "default"; Haiku 5.5 has none (same source).
CLAUDE_MODELS = {
    'claude-opus-5-5': ModelInfo('claude-opus-5-5', 'Claude Opus 5.5', 'Most capable · the default', True, 0, True),
    'claude-sonnet-5-5': ModelInfo('claude-sonnet-5-5', 'Claude Sonnet 5.5', 'Faster, lower cost', True, 10, True),
    'claude-haiku-5-5': ModelInfo('claude-haiku-5-5', 'Claude Haiku 5.5', 'Fastest, lowest cost', True, 20, False),
}
GROQ_DEFAULT = 'openai/gpt-oss-120b'
# The Groq models Vantage lists from Groq's documentation (October 2026); the picker adds what the key's /models
# endpoint returns when the settings dialog asks for it. None is documented to take images.
GROQ_MODELS = {
    'openai/gpt-oss-120b': ModelInfo('openai/gpt-oss-120b', 'GPT-OSS 120B', 'Most capable on Groq', False, 0),
    'openai/gpt-oss-20b': ModelInfo('openai/gpt-oss-20b', 'GPT-OSS 20B', 'Faster', False, 10),
    'llama-3.3-70b-versatile': ModelInfo('llama-3.3-70b-versatile', 'Llama 3.3 70B', 'No reasoning', False, 20),
}
FAKE_MODELS = {'fake-model': ModelInfo('fake-model', 'Fake model', 'Scripted replies for the tests')}

# USD per million tokens. Source: Anthropic's model table in the Claude API reference (cached 2026-10-06, read
# 2026-11-07): Opus 5.5 $4 in / $20 out, cache reads $0.20; Sonnet 5.5 $2 / $10, cache reads $0.20; Haiku 5.5
# $0.10 / $0.50 (prompts up to 100K tokens; $0.50 / $2.50 beyond), cache prices not stated there (None).
# Cache writes are not stated in that table for any of them (None).
PRICES = {
    'claude-opus-5-5': {'input': 4.00, 'output': 20.00, 'cache_write': None, 'cache_read': 0.20},
    'claude-sonnet-5-5': {'input': 2.00, 'output': 10.00, 'cache_write': None, 'cache_read': 0.20},
    'claude-haiku-5-5': {'input': 0.10, 'output': 0.50, 'cache_write': None, 'cache_read': None},
    'claude-haiku-5-5 (prompt over 100K)': {'input': 0.50, 'output': 2.50, 'cache_write': None, 'cache_read': None},
}
# a request whose prompt (input + cache tokens) is longer than this is priced at the long-prompt row (same source)
LONG_PROMPT = {'claude-haiku-5-5': (100_000, 'claude-haiku-5-5 (prompt over 100K)')}
PRICE_SOURCE = ('Anthropic\'s model table (Claude API reference, cached 2026-10-06, read 2026-11-07), USD per '
                'million tokens')


def claude_available() -> bool:
    """The anthropic SDK can be imported (an optional dependency)."""
    try:
        return importlib.util.find_spec('anthropic') is not None
    except (ImportError, ValueError):
        return False


def table(provider) -> dict:
    return {CLAUDE: CLAUDE_MODELS, GROQ: GROQ_MODELS, FAKE: FAKE_MODELS}.get(provider, {})


def models(provider) -> list:
    return sorted(table(provider), key=lambda m: (table(provider)[m].rank, m))


def model_info(provider, model_id) -> ModelInfo:
    return table(provider).get(model_id) or ModelInfo(model_id, str(model_id).split('/')[-1], '', provider == CLAUDE,
                                                      60)


def default_model(provider) -> str:
    return {CLAUDE: CLAUDE_DEFAULT, GROQ: GROQ_DEFAULT, FAKE: 'fake-model'}.get(provider, CLAUDE_DEFAULT)


@dataclass
class Usage:
    """Tokens of one conversation (input, output, the prompt cache's reads and writes) and the cost when every price
    it needs is known."""
    prompt: int = 0
    completion: int = 0
    cache_read: int = 0
    cache_write: int = 0
    requests: int = 0
    provider: str = CLAUDE
    model: str = ''
    by_model: dict = field(default_factory=dict)        # model -> [input, output, cache_write, cache_read]

    def add(self, model='', input_tokens=0, output_tokens=0, cache_write=0, cache_read=0):
        i, o, w, r = (int(x or 0) for x in (input_tokens, output_tokens, cache_write, cache_read))
        self.prompt += i + w + r
        self.completion += o
        self.cache_read += r
        self.cache_write += w
        self.requests += 1
        key = model or self.model
        lp = LONG_PROMPT.get(key)
        if lp and i + w + r > lp[0]:
            key = lp[1]                                  # priced at the long-prompt rates
        row = self.by_model.setdefault(key, [0, 0, 0, 0])
        for k, v in enumerate((i, o, w, r)):
            row[k] += v

    def add_openai(self, u: dict):
        """A Groq usage dict (OpenAI field names)."""
        if u:
            self.add(self.model, u.get('prompt_tokens'), u.get('completion_tokens'))

    @property
    def tokens(self) -> int:
        return self.prompt + self.completion

    def cost(self) -> float | None:
        """USD, or None when a price it needs is not known (Groq; a cache price not stated)."""
        if self.provider != CLAUDE:
            return None
        total = 0.0
        for m, (i, o, w, r) in self.by_model.items():
            pr = PRICES.get(m)
            if pr is None:
                return None
            for n, k in ((i, 'input'), (o, 'output'), (w, 'cache_write'), (r, 'cache_read')):
                if n:
                    if pr.get(k) is None:
                        return None
                    total += n * pr[k] / 1e6
        return total

    def line(self) -> str:
        """'12.3k tokens in (9.8k from the cache), 1.2k out · about $0.04': the dock's usage line."""
        if not self.requests:
            return ''
        bits = f'{self.prompt / 1000:.1f}k tokens in'
        if self.cache_read:
            bits += f' ({self.cache_read / 1000:.1f}k from the cache)'
        bits += f', {self.completion / 1000:.1f}k out'
        c = self.cost()
        if c is not None:
            bits += f' · about ${c:.3f}' if c < 1 else f' · about ${c:.2f}'
        elif self.provider == GROQ:
            bits += ' · cost not estimated (Groq\'s prices are not bundled)'
        elif self.provider == CLAUDE:
            bits += ' · cost not estimated (a price is not stated in the bundled table)'
        return bits
