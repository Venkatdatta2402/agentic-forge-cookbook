"""A judge that checks an answer claim by claim, against the repo's own lines, on a different model from the one
that wrote it.

    judge()      one answer, in two calls to Gemini Flash:
                   1. extract  the answer's claims, each with its own words quoted; shown the question and the
                               answer only
                   2. check    each of those claims marked supported, contradicted or not in the sources, against
                               the reference points and the evidence lines; and the points the answer makes
    scores()     a verdict as numbers: claims wrong, claims not in the sources, share of the points made
    judge_all()  every saved answer of an experiment that has no verdict yet, each verdict saved as it arrives

Why claim by claim. 05's checks -- a must-contain word, a cited file -- passed an answer that put facts on the
wrong side of a comparison. A verdict on the whole answer ("is this faithful?") has the same blind spot: most of it
is right. A claim that attaches the right fact to the wrong thing is one claim, and it is wrong.

Why two calls. Asked in one call to list an answer's claims while looking at the answer key, the judge listed the
KEY: it rewrote the answer's claims into the reference points' shape -- "in both, durability stops at the
workflow", which the answer never said -- and left out every claim the key had no point for, among them one that
was wrong. A judge that cannot see the key while it reads the answer cannot do that, and every claim it lists must
quote the answer: a quote not in the answer is caught by code (`_unanchored`), as are claims left unmarked.

Why a different model. The answers are gpt-oss-120b's on Groq; a judge from the same family grades writing like its
own, and 05's gpt-oss-20b judge misread its own prompt on 3 answers of 15. Gemini also has its own quota.

Rate limits. Gemini's free tier limits requests per minute and per day. A per-minute 429 says how long to wait, and
is waited out; a per-day one stops the run. Verdicts are saved one by one, so a stopped run loses nothing: the next
judges only what is missing. The answers it reads were saved before any judging began, so a judge that cannot run
never costs a Groq token.
"""

import json
import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import eval_set as E

load_dotenv(E.CB.REPO / ".env")

# Not the newest Flash. gemini-3.8-flash answered "503 high demand" to every try for over three minutes on the day
# this was built, and gemini-3.5-flash allows about 20 requests a day on the free tier -- ten answers, at two calls
# each, which ran out 8 answers into the first round. An evaluation loop judges every answer of every round.
JUDGE_MODEL = "gemini-2.5-flash"
TRIES = 6


# --------------------------------------------------------------------------- 1. extract

class Claim(BaseModel):
    claim: str = Field(description="One statement the answer makes about the repo, in a few plain words")
    quote: str = Field(description="The answer's own words that make it, copied exactly: a phrase or sentence")


class Claims(BaseModel):
    claims: list[Claim]


EXTRACT = """Below is an answer to a question about a code repository. List every claim it makes about the
repository: what something does, which part does it, a number, a measured result, a reason, a limitation, a file or
line it points to -- including claims inside tables, lists and code comments, and including claims about which file
a piece of code comes from. One claim per statement; a table cell or bullet that says two things is two claims.
Skip headings, restatements of the question, and general advice that says nothing about this repository.

For each claim, copy the answer's own words that make it, exactly as they appear.

QUESTION
{question}

ANSWER
{answer}"""


# --------------------------------------------------------------------------- 2. check

class Mark(BaseModel):
    n: int = Field(description="The claim's number")
    verdict: Literal["supported", "contradicted", "not in the sources"]
    source: str = Field(description="What it was checked against: evidence lines (e.g. 'runtime.py 185') or a "
                                    "reference point (e.g. 'point 3'); empty when not in the sources")
    why: str = Field(description="For a contradicted claim, what the sources say instead; otherwise empty")


class Marks(BaseModel):
    marks: list[Mark]
    points_made: list[int] = Field(description="The numbers of the reference points the answer makes")


CHECK = """You check claims made in an answer about a code repository against the repository's own words.

QUESTION
{question}

REFERENCE POINTS -- what a right answer says. Each is true of the repo.
{points}

EVIDENCE -- the repo's lines the reference points come from, numbered.
{evidence}

THE ANSWER
{answer}

ITS CLAIMS, numbered, each with the answer's words that make it
{claims}

1. Mark every numbered claim, once:
   supported           the evidence or a reference point says it. Plausible is not enough: what neither of them
                       actually says is "not in the sources"
   contradicted       the evidence or a reference point says otherwise. This includes the right fact attached to the
                       wrong thing -- something one design does, credited to the other; a feature one has, said to be
                       missing from it; code from one file, said to be from another -- a wrong number, and the
                       reverse of what is said
   not in the sources  neither says it, nor says otherwise
   Read each claim as the answer makes it, in its place: a claim under a heading or in a column about one design
   is about that design, and in an answer comparing two things, "only X" means only X of those two.
2. List the reference points the answer makes. A point counts when the answer states its substance, in any words;
   not when it only mentions the topic.

Judge only against the evidence and the reference points, not against what you know of other software."""


def _numbered(question):
    return "\n".join(f"{i}. {point}" for i, (point, _) in enumerate(question["points"], 1))


def prompts(question, answer, claims=None):
    """The two prompts: extracting (no key in it), and checking the extracted claims (the key and evidence in it)."""
    extract = EXTRACT.format(question=question["question"], answer=answer)
    if claims is None:
        return extract, None
    listed = "\n".join(f"{i}. {c['claim']}\n   words: \"{c['quote']}\"" for i, c in enumerate(claims, 1))
    return extract, CHECK.format(question=question["question"], points=_numbered(question),
                                 evidence=E.evidence(question), answer=answer, claims=listed)


def _flat(text):
    return " ".join(re.sub(r"[*`_]", "", text or "").lower().split())


def _unanchored(claims, answer):
    """Claims whose quoted words are not in the answer (ignoring case, spacing and markdown marks)."""
    return [c for c in claims if _flat(c["quote"]) not in _flat(answer)]


class JudgeQuotaSpent(RuntimeError):
    """The judge model's daily quota is used up. Nothing more can be judged until it resets."""


def _wait(error):
    """Seconds to wait before trying again, from a Gemini error; raises JudgeQuotaSpent for a per-day limit."""
    code = getattr(error, "code", None)
    body = getattr(error, "details", None) or {}
    details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    if code == 429:
        quotas = [v.get("quotaId", "") for d in details for v in d.get("violations", [])]
        if any("PerDay" in q for q in quotas):
            raise JudgeQuotaSpent(f"{JUDGE_MODEL}: daily quota spent ({', '.join(quotas)})") from error
        delay = next((d["retryDelay"] for d in details if "retryDelay" in d), "30s")
        return float(re.sub(r"[^\d.]", "", delay) or 30) + 2
    if code in (500, 503, 504):
        return 30.0          # overloaded or unavailable: not a quota, try again shortly
    return None              # anything else is a real error


def _groq(model):
    return model.startswith(("qwen/", "openai/", "meta-llama/"))


# Groq's free tier refuses a single request over 8,000 tokens (413) -- and for gpt-oss-20b it counts the reply's
# `max_tokens` as well as the prompt ("Requested 9,242" for a 1,050-token prompt and max_tokens of 8,192). The reply
# needs room (its hidden reasoning counts as output), so each request asks for what the prompt leaves, and a check
# prompt -- evidence, answer and every claim -- is kept to 4,500 tokens by checking the claims in batches, each batch
# with the whole answer, evidence and key, so no claim is checked out of its context.
GROQ_LIMIT = 8000
GROQ_REQUEST_TOKENS = 4500
_ENCODING = None


def _size(text):
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("o200k_base")
    return len(_ENCODING.encode(text))


def _batches(question, answer, claims):
    """The claims in groups whose check prompts fit one Groq request, in order."""
    groups, current = [], []
    for claim in claims:
        if current and _size(prompts(question, answer, current + [claim])[1]) > GROQ_REQUEST_TOKENS:
            groups.append(current)
            current = []
        current.append(claim)
    return groups + ([current] if current else [])


def _ask_groq(client, model, contents, schema, log):
    """The same call to a model on Groq, through its OpenAI-compatible API. JSON mode, the schema in the prompt, the
    reply validated here; one more try if it does not validate. Groq's per-minute limits are waited out by the
    OpenAI client (`max_retries`); its per-day limit stops the run, as Gemini's does."""
    import openai
    asked = f"{contents}\n\nReply with JSON only, matching this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    for attempt in (1, 2, 3):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": asked}], temperature=0,
                response_format={"type": "json_object"}, max_tokens=GROQ_LIMIT - _size(asked) - 200)
            return schema.model_validate_json(response.choices[0].message.content), response.usage.total_tokens
        except openai.RateLimitError as error:
            if "per day" in str(error):
                raise JudgeQuotaSpent(f"{model}: daily token limit spent") from error
            raise
        except (openai.BadRequestError, ValueError) as error:
            # JSON mode's own refusal of an empty or broken reply (400 json_validate_failed), or a reply that is
            # JSON but not the schema: the model's slip, asked again; any other 400 is a real error
            if attempt == 3 or (isinstance(error, openai.BadRequestError) and "json_validate_failed" not in str(error)):
                raise
            log(f"    {model}: reply was not the JSON asked for, asking again")


def _ask(client, model, contents, schema, sleep, log):
    """One structured call, waiting out what can be waited out. Returns (parsed, tokens)."""
    if _groq(model):
        return _ask_groq(client, model, contents, schema, log)
    from google.genai import errors, types
    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json", response_schema=schema,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    for attempt in range(1, TRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            parsed = response.parsed if isinstance(response.parsed, schema) else \
                schema.model_validate_json(response.text)
            return parsed, getattr(response.usage_metadata, "total_token_count", None) or 0
        except errors.APIError as error:
            wait = _wait(error)
            if wait is None or attempt == TRIES:
                raise
            log(f"    {model}: {error.code}, waiting {wait:.0f}s (try {attempt} of {TRIES})")
            sleep(wait)


def judge(question, answer, client=None, model=JUDGE_MODEL, sleep=time.sleep, log=print):
    """One answer's verdict: every claim with its quote and mark, the points made, and what judging cost."""
    if client is None and _groq(model):
        import openai
        client = openai.OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1",
                               max_retries=8)
    elif client is None:
        # its own key if there is one: Google stopped offering gemini-2.5-flash to new keys after this judge was
        # checked against the hand marks, and a different model would be a different judge, to be checked again
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY_JUDGE") or os.environ["GEMINI_API_KEY"])
    extract, _ = prompts(question, answer)
    found, used = _ask(client, model, extract, Claims, sleep, log)
    claims = [c.model_dump() for c in found.claims]
    groups = _batches(question, answer, claims) if _groq(model) else [claims]
    marks, points, more, start = {}, None, 0, 0
    for group in groups:
        _, check = prompts(question, answer, group)
        marked, used = _ask(client, model, check, Marks, sleep, log)
        more += used
        marks.update({start + m.n: m for m in marked.marks if 1 <= m.n <= len(group)})
        # every batch sees the whole answer; the points it makes are read from the first
        points = marked.points_made if points is None else points
        start += len(group)
    for i, c in enumerate(claims, 1):
        m = marks.get(i)
        c.update(verdict=m.verdict if m else "unmarked", source=m.source if m else "", why=m.why if m else "")
    return {"claims": claims,
            "points made": sorted({p for p in (points or []) if 1 <= p <= len(question["points"])}),
            "check calls": len(groups),
            "unanchored": len(_unanchored(claims, answer)),
            "unmarked": sum(c["verdict"] == "unmarked" for c in claims),
            "judge": model, "judge tokens": used + more}


def scores(verdict, question):
    """A verdict as numbers. An answer that never came gets no scores: it is counted as unanswered instead."""
    if verdict.get("no answer"):
        return {}
    marks = [c["verdict"] for c in verdict["claims"]]
    return {"claims wrong": marks.count("contradicted"),
            "claims not in the sources": marks.count("not in the sources"),
            "claims": len(marks),
            "points made": round(len(verdict["points made"]) / len(question["points"]), 3)}


def judge_all(answers, verdicts_path, questions=None, client=None, sleep=time.sleep, log=print):
    """Judge every saved answer that has no verdict yet, saving each verdict as it arrives.

    `answers` maps a question to its saved output ({"answer": ..., "stopped": ...}). Returns the verdicts, and
    whether it finished: False when the judge's daily quota ran out, and the rest waits for the next run.
    """
    questions = {q["question"]: q for q in (questions or E.QUESTIONS)}
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8")) if verdicts_path.exists() else {}
    for text, output in answers.items():
        if text in verdicts:
            continue
        if not output.get("answer"):
            # nothing to judge, and no point paying to be told so
            verdicts[text] = {"no answer": output.get("stopped") or "no answer"}
        else:
            try:
                verdicts[text] = judge(questions[text], output["answer"], client=client, sleep=sleep, log=log)
            except JudgeQuotaSpent as spent:
                log(f"    stopped: {spent}. {len(verdicts)} of {len(answers)} judged; run again after the reset.")
                return verdicts, False
        verdicts_path.parent.mkdir(parents=True, exist_ok=True)
        verdicts_path.write_text(json.dumps(verdicts, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"    judged {len(verdicts)} of {len(answers)}: {text[:60]}")
    return verdicts, True
