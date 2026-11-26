"""Which groups a task asks figures for: one small model call, made by code's task check.

Code finds the names a task gives (lead.names_in) but not how the sentence puts them together:
"Muslim women in Kerala" is one group, "Muslim women against women in Kerala" two, with the same
names in both. A small model reads only that, from the task and the names, and answers with each
group as an object, one value per field (a field left out is everyone); code then asks the server
whether each is published (no tokens). It is called only when the names come from two fields or
more: names of one field are always separate groups. It also leaves out a name the task doesn't
use for people ("and others" is no social group).

Each group is an object because a list of names was read two ways: as one group (["female",
"male", X] for women in X and men in X) and as a field's names (["graduate", "post graduate"],
["male", "female"]), so code couldn't tell a group from a list of names (eval, 2 runs of 67:
117 and then 106 of 134 groups right, and 2 valid tasks refused).
"""

import itertools
import json
import re

from .. import config
from ..core import llm, meter, trace

INSTRUCTIONS = (
    "A task asks for figures about groups of people. You get the task (or a question) and the names "
    "in it that are groups in the data, by field. Say which groups it asks figures for: each group is the "
    "people one figure is about, given by the field values that describe them; a field left out "
    "means everyone. Names that describe the same people go in one group, with every name that "
    "applies to them. Names set against each other, or asked for as separate figures, go in "
    "separate groups; a name can be in several groups. Leave out a name the task doesn't use for "
    "people. Reply by calling `groups`.")


def label(field, value):
    return f"{value} ({field})"


def _fields(names):
    out = {}
    for f, v in names:
        out.setdefault(f, []).append(v)
    return out


def schema(names):
    """The reply's form: groups, each an object of the fields found in the task, each field one of
    the values found."""
    fields = _fields(names)
    group = {"type": "object", "additionalProperties": False,
             "properties": {f: {"type": "string", "enum": vs} for f, vs in fields.items()}}
    return {"type": "function", "function": {
        "name": "groups", "description": "the groups the task asks figures for",
        "parameters": {"type": "object", "required": ["groups"], "properties": {
            "groups": {"type": "array", "description": "each group: the people one figure is about",
                       "items": group}}}}}


def _value(given, field, written):
    """A value as found in the task, from how the reply wrote it: as given, any case, with or
    without the number in front ("post graduate & above")."""
    want = str(written).strip().lower()
    for v in given.get(field, []):
        if want in (v.lower(), re.sub(r"^\d+\.", "", v).lower()):
            return v
    return None


def _groups_from(text):
    """The groups list read on its own from a reply that isn't valid JSON as a whole. Eval: 4 of
    67 calls came back so, each with a stray quote after the list ('...}]"}')."""
    m = re.search(r'"groups"\s*:\s*(\[.*\])', text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m[1])
    except ValueError:
        return None


async def groups_in(task: str, names: list[tuple[str, str]], keep: list | None = None) -> list[dict] | None:
    """[{field: value}] for each group the task (or question) asks figures for, or None if the
    reply can't be read (the check then lets the task through: a misread must never refuse a
    task). Which names go together is the model's to read from the sentence; code only reads the
    reply's form. A rule in code joining names the reply left apart was tried and dropped: it
    reads the reply's shape, not the sentence, and with more fields it would guess. `keep`: the
    model's own reply is appended to it (the eval saves it, to score again offline)."""
    given = _fields(names)
    listed = "; ".join(f"{f}: {', '.join(vs)}" for f, vs in given.items())
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Task: {task}\nNames, by field: {listed}"}]
    from ..core.runtime import parse_call
    # the lead's: it runs inside ask_analysts, and the page counts it with the lead's calls
    with trace.span("task check: groups", input=task), meter.acting_as("lead > task check"):
        try:
            # reasoning counts against max_tokens (gpt-oss): room for medium effort too
            reply = await llm.complete(config.GROUPING_MODEL, messages, tools=[schema(names)], max_tokens=1500,
                                       reasoning_effort=config.GROUPING_EFFORT, tool_choice="required")
            text = reply.tool_calls[0].arguments if reply.tool_calls else reply.content
        except llm.BadRequestError as e:
            # Groq turned the call down and says what was written: read the groups from that
            # (eval: 15 of 134 calls came back so, with nothing read from them)
            text = llm.failed_generation(e)
            if keep is not None:
                keep.append({"rejected": llm._describe(e), "generated": text})
        except Exception as e:                  # noqa: BLE001 -- a limit, a timeout: the check lets the tasks through
            trace.note("task check: grouping failed", error=type(e).__name__)
            return None
        found = parse_call(text)
        raw = found[1].get("groups") if found else _groups_from(text)
    if keep is not None:
        keep.append(raw)
    if not isinstance(raw, list):
        return None
    out = []
    for group in raw:
        if not isinstance(group, dict):
            continue
        # a list as a field's value is that field's groups side by side
        options = {f: [x for x in (w if isinstance(w, list) else [w]) if _value(given, f, x)]
                   for f, w in group.items() if f in given}
        options = {f: [_value(given, f, x) for x in xs] for f, xs in options.items() if xs}
        for combo in itertools.product(*options.values()):
            one = dict(zip(options, combo))
            if one and one not in out:
                out.append(one)
    return out
