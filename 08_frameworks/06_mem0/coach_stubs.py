"""Stand-ins for every model and embedding call, so both coaches run offline -- through real Mem0, real Qdrant, real
SQLite and real Chroma.

    groq          chapter 1's client patched: the scratch coach calls `remember` with what a careful coach would
                  keep, then replies (round 2: replies, and its write step gets the same as a form); the Mem0
                  coach just replies. Every request is kept for reading
    FakeMemoryLLM Mem0's model: for each check-in, the sentences a careful extractor would write, in Mem0's style
    FakeEmbedder  words hashed into 768 numbers, so texts sharing words are near each other -- enough for search
                  to mean something, which random numbers are not (Mem0 drops anything under 0.1 similarity)
    embeddings    the same, for chapter 4's `stores.embed`
"""

import hashlib
import json
import math
import re
from contextlib import contextmanager

import runners as R

DATES = {s["date"]: s["id"] for s in R.SESSIONS}

# What a careful coach keeps from each check-in, through the scratch coach's `remember`
WRITES = {
    "maya-1": {"profile": [{"field": "race", "value": "Riverside Half Marathon, Sunday 18 October"},
                           {"field": "goal", "value": "under 1:50"},
                           {"field": "days", "value": "Tuesday, Thursday, Saturday, and a long run on Sunday"},
                           {"field": "time_of_day", "value": "evenings, after work"},
                           {"field": "injuries", "value": "left knee sore after long downhill stretches"}],
               "runs": [{"on": "2026-08-30", "kind": "long", "km": 14, "minutes": 86}]},
    "tom-1": {"profile": [{"field": "race", "value": "City Marathon, Sunday 8 November (first marathon)"},
                          {"field": "days", "value": "five days a week, long run on Saturdays"},
                          {"field": "food", "value": "vegetarian; long runs on water alone so far"}],
              "runs": [{"on": "2026-08-29", "kind": "long", "km": 24, "felt": "fine"}]},
    "maya-2": {"runs": [{"on": "2026-09-06", "kind": "long", "km": 16, "minutes": 98, "felt": "good, knee fine"},
                        {"on": "2026-09-01", "kind": "easy", "km": 6}, {"on": "2026-09-03", "kind": "easy", "km": 6}]},
    "maya-3": {"profile": [{"field": "race", "value": "Riverside Half Marathon, Sunday 1 November (moved from 18 October)"},
                           {"field": "time_of_day", "value": "mornings, before work"},
                           {"field": "injuries", "value": "left knee sore after hill repeats on 10 September"}],
               "runs": [{"on": "2026-09-13", "kind": "long", "km": 17, "minutes": 105}]},
    "tom-2": {"profile": [{"field": "food", "value": "vegetarian; gels on long runs worked much better than water"},
                          {"field": "injuries", "value": "right hamstring tight since the 28 km run"}],
              "runs": [{"on": "2026-09-12", "kind": "long", "km": 28, "felt": "much better with gels"}],
              "notes": ["Gets stomach cramps if he eats less than two hours before a run."]},
    "maya-4": {"profile": [{"field": "injuries", "value": "none: knee cleared by a physio on 22 September (tight IT band)"},
                           {"field": "days", "value": "Tuesday, Friday and Sunday (new job)"}],
               "runs": [{"on": "2026-09-20", "kind": "long", "km": 18, "minutes": 111}]},
    "maya-5": {"profile": [{"field": "goal", "value": "1:45"}],
               "runs": [{"on": "2026-09-26", "kind": "race", "km": 10, "minutes": 49.5, "felt": "personal best"}],
               "notes": ["Goes out far too fast in the first kilometre at race starts, and pays for it at the end."]},
}

# What a careful extractor writes from each check-in, in the style Mem0's prompt asks for
EXTRACTS = {
    "maya-1": ["Maya is training for the Riverside Half Marathon on Sunday 18 October 2026 and wants to finish under 1:50",
               "Maya runs four days a week -- Tuesday, Thursday, Saturday, and a long run on Sunday -- in the evening after work",
               "Maya's longest run is 14 km, run on 30 August 2026 in 1:26",
               "Maya's left knee gets sore after long downhill stretches"],
    "tom-1": ["Tom is running his first marathon, the City Marathon, on Sunday 8 November 2026",
              "Tom runs five days a week with a long run on Saturdays, and ran 24 km on 29 August 2026",
              "Tom is vegetarian and has done his long runs on water alone"],
    "maya-2": ["Maya ran a 16 km long run on 6 September 2026 in 1:38; it felt good and her knee was fine"],
    "maya-3": ["The Riverside Half Marathon was moved from 18 October to Sunday 1 November 2026 because of roadworks",
               "Maya switched from evening runs to running in the mornings before work",
               "Maya ran 17 km on 13 September 2026 in 1:45",
               "Maya's knee was sore after hill repeats on Thursday 10 September 2026"],
    "tom-2": ["Tom ran 28 km on 12 September 2026 using gels, which went much better than water alone",
              "Tom's right hamstring has been tight since the 28 km run",
              "Tom gets stomach cramps when he eats less than two hours before a run"],
    "maya-4": ["A physio cleared Maya's knee on 22 September 2026: it was a tight IT band, and she can run normally",
               "Maya started a new job and now runs three days a week: Tuesday, Friday and Sunday",
               "Maya ran 18 km on 20 September 2026 in 1:51"],
    "maya-5": ["Maya ran a 10k race on 26 September 2026 in 49:30, a personal best",
               "Maya tends to go out too fast in the first kilometre at race starts",
               "Maya changed her half-marathon goal from 1:50 to 1:45"],
}


# Round 4: the runner's own words for each run's day and time (in WRITES' order), the old `food` split in two, and
# one field added where none fits
SAID = {"maya-1": [("last Sunday", "1:26")], "tom-1": [("last Saturday", "")],
        "maya-2": [("Sunday's long run", "1:38"), ("Tuesday", ""), ("Thursday", "")],
        "maya-3": [("Sunday's long run", "1:45")], "tom-2": [("Saturday", "")],
        "maya-4": [("Sunday's long run", "1:51")], "maya-5": [("on Saturday", "49:30")]}
FOOD = {"tom-1": {"diet": "vegetarian", "fuel": "water alone on long runs so far"},
        "tom-2": {"fuel": "gels on long runs, much better than water alone"}}
NEW = {"tom-2": [{"name": "pre-run eating", "about": "how long before a run he can eat",
                  "value": "stomach cramps if he eats less than two hours before a run"}]}


# Round 5: notes with a kind and the fields they depend on -- one that will go stale when Maya's days change, and
# her habit -- and a field the run log could work out, which the check should refuse
NOTES5 = {"maya-1": [{"text": "Her long run is on Sundays, after the three shorter days", "kind": "fact",
                      "depends_on": ["days"]}],
          "maya-5": [{"text": "Goes out far too fast in the first kilometre at race starts", "kind": "habit",
                      "depends_on": []}]}
NEW5 = {"maya-1": [{"name": "longest run", "about": "her longest run", "value": "14 km"}]}


def _profile_fields(kwargs):
    """The profile slots the card's schema asks for, read from the request itself."""
    schema = kwargs["response_format"]["json_schema"]["schema"]
    profile = schema["$defs"][schema["properties"]["profile"]["$ref"].split("/")[-1]]
    return list(profile["properties"])


def _session_of(text, pattern):
    found = re.search(pattern, text)
    return DATES[found.group(1)] if found else None


def completion(content=None, calls=(), prompt=100, output=50):
    from openai.types.chat import ChatCompletion
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]
    return ChatCompletion.model_validate({
        "id": "stub", "object": "chat.completion", "created": 0, "model": "stub",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
                     "message": {"role": "assistant", "content": content, "tool_calls": tool_calls or None}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}})


@contextmanager
def groq(fail_on=None):
    """Chapter 1's client, answered by a script. `fail_on` names a check-in whose first call is refused, as Groq
    refuses one over its daily limit."""
    import httpx
    import llm
    import openai
    from chat import count_tokens
    seen = []
    real = llm.client.chat.completions.create

    def create(**kwargs):
        messages = kwargs["messages"]
        text = "\n".join(str(m.get("content") or "") for m in messages)
        session = _session_of(text, r"Today is (?:[A-Z][a-z]+ )?(\d{4}-\d{2}-\d{2})")
        seen.append({"session": session, "tools": bool(kwargs.get("tools")), "messages": messages,
                     "tokens": count_tokens(json.dumps(messages)) + count_tokens(json.dumps(kwargs.get("tools") or []))})
        if session == fail_on:
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            raise openai.RateLimitError("tokens per day (TPD): Limit 200000", body=None,
                                        response=httpx.Response(429, request=request))
        kept = any(str(m.get("content") or "").startswith(("Kept:", "Nothing new kept")) for m in messages)
        writes = {k: v for k, v in WRITES.get(session, {}).items() if v}
        if kwargs.get("response_format") and "statements:" in text:                 # round 5's card
            profile = {c["field"]: c["value"] for c in writes.get("profile", []) if c["field"] != "food"}
            profile.update(FOOD.get(session, {}))
            notes = NOTES5.get(session, [{"text": n, "kind": "fact", "depends_on": []} for n in writes.get("notes", [])])
            runs = [{"when": when, "kind": r["kind"], "km": r["km"], "time": time, "felt": r.get("felt", "")}
                    for r, (when, time) in zip(writes.get("runs", []), SAID.get(session, []))]
            card = {"profile": {f: profile.get(f, "unchanged") for f in _profile_fields(kwargs)},
                    "new_fields": NEW.get(session, []) + NEW5.get(session, []), "runs": runs, "notes": notes,
                    "statements": [{"said": v, "went_to": "profile", "field": f} for f, v in profile.items()] +
                                  [{"said": r["when"], "went_to": "events"} for r in runs] +
                                  [{"said": n["text"], "went_to": "note"} for n in notes]}
            return completion(json.dumps(card))
        if kwargs.get("response_format") and "new_fields:" in text:                 # round 4's card
            profile = {c["field"]: c["value"] for c in writes.get("profile", []) if c["field"] != "food"}
            profile.update(FOOD.get(session, {}))
            card = {"profile": {f: profile.get(f, "unchanged") for f in _profile_fields(kwargs)},
                    "new_fields": NEW.get(session, []),
                    "runs": [{"when": when, "kind": r["kind"], "km": r["km"], "time": time, "felt": r.get("felt", "")}
                             for r, (when, time) in zip(writes.get("runs", []), SAID.get(session, []))],
                    "notes": writes.get("notes", [])}
            return completion(json.dumps(card))
        if kwargs.get("response_format") and "Fill in the record form" in text:     # round 3's form
            message = next(s["message"] for s in R.SESSIONS if s["id"] == session)
            new = {c["field"]: c["value"] for c in writes.get("profile", [])}
            return completion(json.dumps({
                "profile": {f: ({"value": new[f], "quote": message} if f in new else {"value": "unchanged"})
                            for f in ("race", "goal", "days", "time_of_day", "injuries", "food")},
                "runs": [{**r, "quote": message} for r in writes.get("runs", [])],
                "notes": [{"text": n, "quote": message} for n in writes.get("notes", [])]}))
        if kwargs.get("response_format"):                      # round 2's write step, through extract()
            return completion(json.dumps({"profile": [], "runs": [], "notes": [], **writes}))
        if kwargs.get("tools") and writes and not kept:
            return completion(calls=[("remember", writes)])
        return completion(f"Coach's reply for {session}.")

    llm.client.chat.completions.create = create
    try:
        yield seen
    finally:
        llm.client.chat.completions.create = real


class FakeMemoryLLM:
    """Mem0's model: for each check-in, the sentences in EXTRACTS -- found by the date the check-in's custom
    instructions carry, which is also the proof that they reached the model."""

    def __init__(self, fail_on=None):
        self.prompts, self.fail_on = [], fail_on

    def generate_response(self, messages, response_format=None, **kwargs):
        self.prompts.append(messages)
        session = _session_of(messages[-1]["content"], r"took place on (\d{4}-\d{2}-\d{2})")
        if session == self.fail_on:
            import coach_mem0 as M
            raise M.MemoryQuotaSpent("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        return json.dumps({"memory": [{"id": str(i), "text": t, "attributed_to": "user"}
                                      for i, t in enumerate(EXTRACTS.get(session, []))]})


def vector(text, dim=768):
    """Words hashed into `dim` numbers, unit length: texts sharing words point the same way."""
    out = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        out[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def embed(self, text, memory_action=None):
        self.texts.append(text)
        return vector(text)

    def embed_batch(self, texts, memory_action="add"):
        self.texts.extend(texts)
        return [vector(t) for t in texts]


@contextmanager
def embeddings():
    """Chapter 4's `stores.embed`, offline."""
    import stores
    real = stores._embed_uncached
    stores._embed_uncached = lambda texts, task, dim: [vector(t, dim) for t in texts]
    try:
        yield
    finally:
        stores._embed_uncached = real
