"""Automatic model choice (off unless the user turns it on in Assistant settings): which provider and model answers a
request, decided on this machine, deterministically, from what the request is and which keys there are. Nothing is
sent to decide it.

The table (route()):
- a Copilot plan (a run or an export to plan, or a follow-up to the plan on screen) -> deep ("Copilot plan")
- a question about changing the rig (optimise, improve, move, re-place cameras)   -> deep ("rig change")
- anything else                                                                    -> fast ("quick question")
deep = Claude Opus 5.5; fast = Groq's GPT-OSS 120B, or Claude Haiku 5.5 when there is no Groq key. When the tier's model
cannot be used (no Claude key, or the anthropic SDK not installed) the best available model answers and the route says
why (fallback), so the dock can say it once. With automatic choice off, the model picked in the dock answers.

One conversation keeps one provider's history (Groq's message list and Claude's content blocks are never mixed). A
request routed to the other provider starts a new conversation with a hand-over: a short local summary of the
conversation so far (handover_text), sent in front of the first question.

Ported from Vantage assistant/routing.py: the picture and whole-rig rules are DexVantage's rig-change rule; the fast
Claude model is Haiku 5.5."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import providers as PR
from .copilot import is_plan_request

FAST, DEEP = 'fast', 'deep'
DEEP_MODEL = (PR.CLAUDE, 'claude-opus-5-5')
FAST_GROQ = (PR.GROQ, 'openai/gpt-oss-120b')
FAST_CLAUDE = (PR.CLAUDE, 'claude-haiku-5-5')
REASONS = {'plan': 'Copilot plan', 'rig': 'rig change', 'quick': 'quick question', 'manual': 'chosen in the dock'}

_RIG = re.compile(r'\b(optimi[sz]e|improve|tune|move|re-?place|re-?arrange|re-?position|re-?mount|where should)\b'
                  r'[^.?!]{0,60}\b(rig|cameras?|sensors?|mounts?|wrist|head|layout)\b', re.IGNORECASE)


@dataclass
class Route:
    provider: str
    model: str
    tier: str
    reason: str
    fallback: str = ''

    @property
    def label(self) -> str:
        return f'{PR.model_info(self.provider, self.model).label} · {REASONS.get(self.reason, self.reason)}'

    def to_dict(self) -> dict:
        return {'provider': self.provider, 'model': self.model, 'tier': self.tier, 'reason': self.reason,
                'label': self.label, 'fallback': self.fallback}


def classify(text: str, planning=False) -> tuple[str, str]:
    if planning or is_plan_request(text):
        return DEEP, 'plan'
    if _RIG.search(str(text or '')):
        return DEEP, 'rig'
    return FAST, 'quick'


def usable(keys: dict, claude_sdk: bool) -> dict:
    return {PR.GROQ: bool(keys.get(PR.GROQ)), PR.CLAUDE: bool(keys.get(PR.CLAUDE)) and bool(claude_sdk)}


def route(text: str, planning=False, keys: dict | None = None, claude_sdk=True) -> Route | None:
    """The provider and model for one request; None when no provider can be used."""
    ok = usable(keys or {}, claude_sdk)
    tier, reason = classify(text, planning)
    if tier == DEEP:
        if ok[PR.CLAUDE]:
            return Route(*DEEP_MODEL, tier, reason)
        if ok[PR.GROQ]:
            why = 'there is no Claude key' if not (keys or {}).get(PR.CLAUDE) else 'the anthropic SDK is not installed'
            return Route(*FAST_GROQ, tier, reason, fallback=f'Claude Opus 5.5 would answer this, but {why}: GPT-OSS '
                                                            '120B on Groq answers instead.')
        return None
    if ok[PR.GROQ]:
        return Route(*FAST_GROQ, tier, reason)
    if ok[PR.CLAUDE]:
        return Route(*FAST_CLAUDE, tier, reason)
    return None


HANDOVER_CHARS = 2400


def handover_text(transcript_md: str, from_provider: str, cap=HANDOVER_CHARS) -> str:
    """A short local summary of a conversation from its Markdown transcript (### You / ### Assistant), the latest
    exchanges kept first, capped."""
    turns, cur, mode = [], None, 'q'
    for line in (transcript_md or '').splitlines():
        if line.startswith('### You'):
            cur = {'q': [], 'a': []}
            turns.append(cur)
            mode = 'q'
        elif line.startswith('### Assistant'):
            if cur is None:
                cur = {'q': [], 'a': []}
                turns.append(cur)
            mode = 'a'
        elif cur is not None and line.strip():
            cur[mode].append(line.strip())
    bits, used = [], 0
    for t in reversed(turns):
        q = ' '.join(t['q'])[:300]
        a = ' '.join(t['a'])
        a = a[:420] + ('…' if len(a) > 420 else '')
        item = f'- You asked: {q}' + (f'\n  Answer: {a}' if a else '')
        if used + len(item) > cap:
            break
        bits.append(item)
        used += len(item)
    bits.reverse()
    skipped = len(turns) - len(bits)
    who = PR.LABEL.get(from_provider, from_provider)
    head = (f'[Earlier in this conversation (with {who}; a summary made by DexVantage, the full figures are in the '
            'tools'
            + (f'; {skipped} earlier exchange{"s" if skipped != 1 else ""} left out' if skipped else '') + '):')
    return f'{head}\n' + ('\n'.join(bits) if bits else '- (nothing yet)') + '\n]'
