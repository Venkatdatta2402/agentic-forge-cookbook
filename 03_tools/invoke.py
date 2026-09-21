import concurrent.futures
from dataclasses import dataclass, field, replace

import tiktoken
from pydantic import ValidationError

# imported for its side effect as much as for the function: chat.py points tiktoken at a local
# cache directory, so counting tokens works without reaching for the network
from chat import count_tokens
from verify import VerificationError

_encoding = tiktoken.get_encoding("cl100k_base")


class ToolError(Exception):
    """Raised by a tool to say: I ran, and the answer is no.

    The difference that matters is not how bad the failure is, it is who can do something about
    it. "No weather station for Narnia" is something the model can act on -- try another city,
    or say it can't be done. A missing API key is not: the model will retry, invent workarounds,
    and burn the budget on a problem that was never its to fix.

    So a tool raises ToolError for the first kind, and anything else it raises is treated as the
    second.
    """


# The one kind that does not belong in the conversation: a bug in the tool. The model cannot fix
# it, and telling it about it invites a workaround for something it cannot see.
OURS_TO_FIX = "broken"


@dataclass
class Result:
    """What came back from one tool call, and what can be done about it."""

    tool: str
    kind: str  # ok | bad_arguments | refused | unverified | timeout | broken
    content: str  # the text the conversation sees
    # The whole result, before any cutting down. Kept off `content` on purpose: the program can
    # have all of it, while the conversation gets only as much as it can afford to carry.
    value: object = None
    truncated: bool = False
    repeatable: bool = True

    @property
    def ok(self):
        return self.kind == "ok"

    @property
    def for_the_model(self):
        """Whether this belongs in the conversation at all.

        True for a result, and for a failure the model can do something about -- wrong
        arguments, a tool that ran and said no, a call that took too long, a claim that did
        not check out. False only when the tool itself is broken, which is ours to fix and
        not something to tell it about.
        """
        return self.kind != OURS_TO_FIX

    @property
    def safe_to_retry(self):
        # A timeout is the one case where nobody knows whether the tool ran. For a tool that only
        # reads something, calling it again costs a little time. For one that sends an email, it
        # is a second email -- so the tool has to have said in advance whether repeating it is
        # safe, because after the timeout there is no way to find out.
        # Same reasoning for "unverified": the tool claimed to have done the job, so some of
        # it may have happened. How much, nobody knows.
        if self.kind in ("timeout", "unverified"):
            return self.repeatable
        return self.kind in ("bad_arguments", "refused")

    def __str__(self):
        return self.content


def truncate(text, max_tokens):
    """Cut text down to a token budget, and say so in the text itself."""
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    # The note matters as much as the cut. Without it the model reads a sentence that stops in
    # the middle and treats it as the whole answer -- it has no way to tell a short result from a
    # long one with the end missing.
    kept = _encoding.decode(tokens[:max_tokens])
    return f"{kept}\n[... cut short: {len(tokens):,} tokens in all, {max_tokens:,} kept]", True


def _call(function, args, timeout):
    if timeout is None:
        return function(**args)

    # Threads, not asyncio, for the same reason chapter 2 used them: notebooks are already
    # running an event loop, and this work is blocking anyway.
    #
    # Not a `with` block. Leaving one waits for the thread to finish, which is exactly what the
    # timeout is trying not to do -- the call would return only after the slow tool was done,
    # having timed out and waited anyway.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(function, **args).result(timeout=timeout)
    finally:
        # Honest about what this does and doesn't do: it stops US waiting. The thread is still
        # running, because Python cannot kill a thread from outside, and it will keep whatever it
        # holds until it finishes on its own. Stopping a tool for real needs a separate process,
        # which is `09_sandbox`.
        pool.shutdown(wait=False)


def invoke(tool, args, timeout=None, max_tokens=None):
    """Call one tool, and come back with a Result whatever happens.

    Three decisions live here, and nowhere else: how long to wait, how much of the answer the
    conversation is allowed to carry, and what to say when it goes wrong.

    Nothing is raised. Every ending is a Result with a `kind`, so the decision about what to do
    next belongs to the caller -- the same split chapter 2 draws when it says Act carries things
    out and never decides.
    """
    try:
        checked = tool.validate(args)
    except ValidationError as e:
        # Nothing ran, so nothing happened, and the message names the field and the problem --
        # which chapter 1's notebook 6 showed a model correcting on its very next attempt.
        return Result(tool.name, "bad_arguments", str(e), repeatable=tool.repeatable)

    try:
        value = _call(tool.function, checked, timeout)
    except concurrent.futures.TimeoutError:
        waited = f"'{tool.name}' took longer than {timeout} seconds and was given up on."
        advice = (" It only reads, so calling it again is safe."
                  if tool.repeatable else
                  " It may or may not have gone through. Do not call it again -- check first.")
        return Result(tool.name, "timeout", waited + advice, repeatable=tool.repeatable)
    except ToolError as e:
        return Result(tool.name, "refused", str(e), repeatable=tool.repeatable)
    except VerificationError as e:
        # The tool said it worked and a check of the world disagreed. Goes to the model, because
        # it is about to build on something that did not happen -- and the message says what is
        # actually true, not merely that something went wrong.
        return Result(tool.name, "unverified", str(e), repeatable=tool.repeatable)
    except Exception as e:
        # The tool itself is broken -- a missing key, a bad import, a bug. Saying so in the
        # conversation invites the model to work around something it cannot see, so the message
        # is written for whoever is reading the run, and `for_the_model` is False.
        return Result(tool.name, "broken", f"{tool.name} is broken: {type(e).__name__}: {e}",
                      repeatable=tool.repeatable)

    content = str(value)
    truncated = False
    if max_tokens is not None:
        content, truncated = truncate(content, max_tokens)
    return Result(tool.name, "ok", content, value=value, truncated=truncated,
                  repeatable=tool.repeatable)


def invoker(tool, timeout=None, max_tokens=None):
    """Wrap a tool as a plain function, so chapter 2's `Act(functions=...)` can call it.

    `execute_tool_call` finishes with `str(result)`, and `Result.__str__` is its content -- so
    what reaches the conversation is the message, while the Result kept everything else.

    Which makes this the one place that has to honour `for_the_model`. It is computed by
    `invoke()` and read by nobody: `str(result)` does not consult it, so a `broken` result --
    the single kind that is supposed to stay out of the conversation -- went into it anyway.
    Harmless while `broken` meant a bug in a function in this repo. Not harmless from
    `07_mcp` on, where `broken` is the honest reading of any failure a server we did not write
    declines to explain, which is most of them.
    """
    def call(**kwargs):
        result = invoke(tool, kwargs, timeout=timeout, max_tokens=max_tokens)
        if result.for_the_model:
            return result
        # The conversation gets an acknowledgement and nothing to work around; the cause moves
        # to `value`, where whoever is reading the run can still find it. Deliberately not
        # done in `Result.__str__`: printing a broken result and seeing what it actually says
        # is how `03_tools` notebook 3 makes this distinction visible in the first place.
        return replace(result, content=f"{result.tool} could not be completed.",
                       value=result.content)

    return call
