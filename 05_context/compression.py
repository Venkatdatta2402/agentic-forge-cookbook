"""Making information smaller. Selection chooses between items; compression changes them.

Compression is not a stage every context passes through. It is an operation, run when
something calls for it, and the caller decides when:

    something must go in and cannot fit     one item bigger than any budget (a log dump), or
                                            pinned items that overflow it on their own
    the run has grown past a threshold      conversation compression: `compact()`
    someone asks                            a user's "compact this", a harness between sessions

Every method here loses information. What differs is WHAT it loses, whether you chose that,
and whether the result says it is incomplete:

    truncate()       free and exact about what it kept; loses everything outside the cut, and
                     says so in the text
    summarize()      keeps the gist of all of it; loses detail it judged unimportant -- judged
                     for the purpose it was given, which may not be the next caller's purpose
    extract_facts()  keeps individual statements, word for word where it matters; loses the
                     connective tissue -- order, who asked what, why
    salvage()        extraction from superseded items, shown beside their replacements, so it
                     keeps what the new version doesn't restate and not the values it replaced

`compact()` combines them for a conversation: old stretches summarized for the story, facts
extracted for the numbers and rules, superseded items salvaged, and recent turns, task state
and pinned items left untouched.

`why_compress()` reports which of the triggers above apply; it never compresses anything itself.
`missing()` is how the notebooks measure the loss, against a list of facts that should survive.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from chat import _encoding
from context import Item, render, tokens
from llm import chat, extract
from selection import supersede, words


def truncate(item, max_tokens, keep="head"):
    """A copy of `item` cut to `max_tokens`, keeping the "head", the "tail", or both "ends".

    Chapter 3's `invoke.truncate()` keeps the head only, which suits a tool result read top to
    bottom. A log is the opposite: the newest lines are at the end. Which end matters is a fact
    about the content, so it is the caller's choice. The note saying what was cut is not
    optional -- without it a model reads a cut result as a complete one.
    """
    ids = _encoding.encode(item.content)
    if len(ids) <= max_tokens:
        return item
    if keep not in ("head", "tail", "ends"):
        raise ValueError(f"keep must be 'head', 'tail' or 'ends', not {keep!r}")

    # Cut at a line break where there is one. A token boundary lands mid-line, and the first
    # version of this ended a log on a dangling "07:" -- half a timestamp, which reads as data.
    def head(n):
        text = _encoding.decode(ids[:n])
        return text[: text.rfind("\n")] if "\n" in text else text

    def tail(n):
        text = _encoding.decode(ids[-n:])
        return text[text.find("\n") + 1:] if "\n" in text else text

    front = head(max_tokens if keep == "head" else max_tokens // 2) if keep != "tail" else ""
    back = tail(max_tokens if keep == "tail" else max_tokens // 2) if keep != "head" else ""
    cut = len(ids) - len(_encoding.encode(front)) - len(_encoding.encode(back))
    note = f"[... {cut:,} of {len(ids):,} tokens cut ...]"
    text = "\n".join(part for part in (front, note, back) if part)
    return Item(item.kind, text, item.turn, item.about, f"{item.source}, truncated", item.pinned)


def _span(items):
    turns = [i.turn for i in items]
    return f"t{min(turns)}" if min(turns) == max(turns) else f"t{min(turns)}-t{max(turns)}"


def summarize(items, focus=None, max_words=120):
    """Several items as one `summary` item, written by the model.

    `focus` says what the summary is FOR. Without one, the model decides what matters, and it
    decides for a reader it has to imagine. Numbers are asked for verbatim: a summary that
    rounds "2.9 s" to "about 3 seconds" has changed a fact, not shortened it.
    """
    purpose = (f"The summary is for this purpose, so keep everything it needs: {focus}\n"
               if focus else "")
    text = chat([{"role": "user", "content": (
        "Summarize the material below so that someone continuing the work can rely on the "
        f"summary instead of the material.\n{purpose}"
        "Copy every number exactly as written, with its unit -- never round or convert. Keep "
        "names, versions and times. Say what was asked, found and concluded, in order. "
        f"At most {max_words} words. No preamble.\n\n{render(items)}"
    )}])
    return Item("summary", text.strip(), max(i.turn for i in items),
                source=f"summary of {_span(items)}")


class Fact(BaseModel):
    kind: Literal["fact", "constraint"]
    text: str


class Facts(BaseModel):
    facts: list[Fact]


def extract_facts(items, known=()):
    """The individual statements in `items` worth keeping, as `fact` and `constraint` items.

    Constraints come back pinned, and are asked for WORD FOR WORD. A rule restated in the
    model's own words is a different rule -- "don't change production" and "do not restart,
    scale or redeploy anything in production" permit different things.

    `known` is text already in hand (other facts, the pinned rules), so the same thing is not
    extracted twice.
    """
    # Both halves of this prompt were rewritten after a live run. With `known` listed casually
    # above the material, the model extracted FROM it -- a salvage of three superseded items came
    # back with search-api's latency and orders-db's CPU, neither of which was in them. And with
    # "rule" left loose, a line of the agent's own plan ("the database has to be ruled out
    # before blaming the deploy") came back as a pinned constraint, which nothing would ever drop.
    result = extract(_facts_prompt(
        "Below is part of an agent's working session. List every fact in it that someone "
        "continuing the work could need later: measurements with their numbers and units, "
        "times, versions, causes, what was ruled in or out, and decisions.",
        f"MATERIAL:\n{render(items)}", known), Facts)
    return _as_items(result, items, render(items), known)


def salvage(old, newer, known=()):
    """Facts from superseded items that their replacements do not cover.

    Not `extract_facts(old)`. Tried first, that brought back exactly what supersession removed:
    the replaced reading came out as "checkout-api p95 latency is 2400 ms in the last 15
    minutes", present tense, beside the current 2900 ms -- and its stale p99 and request rate
    turned up in a final answer as if they were current. An old item is mostly out of date, and
    an extractor that cannot see the new version has no way to know which parts.

    So the replacement is shown alongside, and only what it does NOT restate is kept: the
    baseline in "2.4 s, up from 0.3 s yesterday", which "now 2.9 s" says nothing about.
    """
    result = extract(_facts_prompt(
        "The OLD items below have each been replaced by a NEWER version of the same thing, so "
        "most of what they say is out of date. List only facts from the OLD items that the "
        "NEWER items neither restate nor give a new value for -- a baseline, a comparison, a "
        "cause the newer version leaves out. Never list a measurement the newer version "
        "updates. Say when each fact applied (\"yesterday\", \"at 07:05\") so it cannot be "
        "read as current. If nothing survives, return an empty list.",
        f"OLD:\n{render(old)}\n\nNEWER:\n{render(newer)}", known), Facts)
    return _as_items(result, old, render(old), known)


def _facts_prompt(task, material, known):
    already = "\n".join(f"- {k}" for k in known) or "(nothing)"
    return (
        f"{task}\n\n"
        "Each fact is one short sentence that makes sense on its own: name what it is about "
        "(\"checkout-api\", never \"it\"), and copy numbers exactly as written, in the unit "
        "written -- never converted, even if the session asks for another unit.\n\n"
        # Without this, a log came back one fact per line: 17 of a compaction's 28 facts were
        # single log lines, for what the log's own observation said in one sentence.
        "Entries that repeat the same pattern -- log lines, readings over time -- are ONE "
        "fact: when the pattern starts, how it changes, and where it ends. Never one fact per "
        "entry.\n\n"
        "kind \"constraint\" is ONLY for a rule the USER stated about how the work must be "
        "done, copied word for word. The agent's own plans, goals and reasoning are never "
        "constraints. Everything else is kind \"fact\".\n\n"
        f"{material}\n\n"
        "ALREADY KNOWN -- this is not part of the material. Do not extract from it, and do "
        "not repeat anything it says:\n"
        f"{already}\n\n"
        f"Return JSON matching this schema:\n{Facts.model_json_schema()}"
    )


def _norm(text):
    return " ".join(text.lower().split()).rstrip(".")


def _as_items(result, items, material, known):
    """The model's facts as items -- after the checks a prompt could not make stick.

    Told plainly that the ALREADY KNOWN block was not material and that plans are never
    constraints, the model still returned a line of the agent's plan, copied out of that block,
    as a pinned constraint. Prompting lowered how often; only code makes it never. Two checks,
    both possible only because of how the facts were asked for:

      - anything that is a verbatim copy of known text is dropped: it was already in hand. A
        NEAR copy is dropped too, on word overlap: a live run returned the read-only rule as a
        fresh constraint beside the pinned one it was extracted from, differing by a comma,
        which the substring test let through
      - a constraint must appear word for word in the material. It was asked for word for word,
        so one that doesn't was either paraphrased or came from somewhere else -- and either way
        must not be pinned. It is kept as an ordinary fact, where selection can still drop it.
    """
    known_text = [_norm(k) for k in known]
    body = _norm(material)
    kept, seen = [], set()
    turn, source = max(i.turn for i in items), f"extracted from {_span(items)}"
    for f in result.facts:
        text = f.text.strip()
        key = _norm(text)
        if not text or key in seen or any(key in k for k in known_text):
            continue
        # near-duplicate of something already in hand: same words, different punctuation or
        # word order. Only checked against `known`, never between two new facts, because two
        # facts about one thing are often both worth keeping.
        mine = words(text)
        if mine and any(len(mine & words(k)) / len(mine) >= 0.8 for k in known_text):
            continue
        seen.add(key)
        kind = f.kind if f.kind != "constraint" or key in body else "fact"
        kept.append(Item(kind, text, turn, source=source, pinned=kind == "constraint"))
    return kept


@dataclass
class Compaction:
    items: list
    summary: Item = None
    facts: list = field(default_factory=list)
    replaced: list = field(default_factory=list)      # what the summary and facts stand in for

    @property
    def tokens(self):
        return tokens(self.items)

    def report(self):
        lines = [f"{tokens(self.replaced)} tokens in {len(self.replaced)} items replaced by "
                 f"{tokens(self.facts) + (self.summary.tokens if self.summary else 0)} tokens: "
                 f"a summary and {len(self.facts)} facts"]
        if self.summary:
            lines.append(f"\n[summary, {self.summary.source}]\n{self.summary.content}")
        lines.append("")
        lines += [f"[{f.kind}{', pinned' if f.pinned else ''}] {f.content}" for f in self.facts]
        return "\n".join(lines)


def compressible(items, keep_recent=6):
    """The items `compact()` would actually replace -- what there is to gain by running it.

    Worth asking before paying for it. A run can be over a size limit and still have almost
    nothing compactable, because everything in it is either pinned, task state, or inside the
    recent window that `compact()` leaves alone. Measured in a live loop: compaction fired twice
    on a growing run, spent two model calls each time, and returned a context the same size it
    started.
    """
    if not items:
        return []
    cutoff = max(i.turn for i in items) - keep_recent
    return [i for i in items
            if not i.pinned and i.kind != "state" and i.turn <= cutoff]


def compact(items, keep_recent=6):
    """Compress everything older than the last `keep_recent` turns. Leave the rest alone.

    What is never touched, and why:
      - pinned items: a summarizer that paraphrases a rule has changed it
      - task state (the current plan, the scratchpad): it is already the run's own summary of
        where things stand, and the next step starts from it
      - the last `keep_recent` turns: what the next step is most likely to build on, verbatim

    What happens to the rest:
      - current messages and observations are SUMMARIZED -- they carry the story
      - facts are EXTRACTED from those, and from old raw tool results, whose numbers a summary
        would smooth away
      - superseded items are SALVAGED: shown beside their replacements, so only what the new
        version does not restate is kept. `supersede()` drops a whole item when one thing in
        it is replaced, and anything else it said goes with it unless it is pulled out first --
        but pulled out blind, the stale values come back too (see `salvage()`)
    """
    now = max(i.turn for i in items)
    cutoff = now - keep_recent
    current, superseded = supersede(items)
    superseded = [i for i, _ in superseded]

    keep, story, raw = [], [], []
    for item in current:
        if item.pinned or item.kind == "state" or item.turn > cutoff:
            keep.append(item)
        elif item.kind == "tool_result":
            raw.append(item)
        else:
            story.append(item)

    if not (story or raw or superseded):
        return Compaction(list(items))
    known = [i.content for i in keep if i.pinned or i.kind in ("state", "fact")]
    summary = summarize(story) if story else None
    facts = extract_facts(story + raw, known=known) if story or raw else []
    if superseded:
        versions = {(i.kind, i.about) for i in superseded}
        newer = [i for i in current if (i.kind, i.about) in versions]
        facts += salvage(superseded, newer, known=known + [f.content for f in facts])

    new = sorted(keep + facts + ([summary] if summary else []), key=lambda i: i.turn)
    return Compaction(new, summary, facts,
                      replaced=sorted(story + raw + superseded, key=lambda i: i.turn))


def why_compress(items, budget, history_limit=None):
    """The reasons, if any, that `items` call for compression. [] means select() is enough.

    Reports rather than acts. Deciding to compress is the caller's call -- a user asking for it
    is a reason this function cannot see -- and each reason points at a different operation:
    a pinned overflow at the pinned items, one oversized item at that item, a long history at
    `compact()`.
    """
    current, _ = supersede(items)
    pinned = [i for i in current if i.pinned]
    reasons = []
    if tokens(pinned) > budget:
        reasons.append(f"pinned items are {tokens(pinned)} tokens, over the budget of {budget}")
    for i in current:
        if not i.pinned and i.tokens > budget:
            reasons.append(f"t{i.turn} {i.kind} is {i.tokens} tokens and can never fit {budget}")
    if history_limit is not None and tokens(current) > history_limit:
        reasons.append(f"the run holds {tokens(current)} tokens, past the limit of {history_limit}")
    return reasons


def missing(items, checks):
    """Which of `checks` ({name: regex}) no longer appear anywhere in `items`' text."""
    text = "\n".join(i.content for i in items)
    return [name for name, pattern in checks.items() if not re.search(pattern, text, re.I)]
