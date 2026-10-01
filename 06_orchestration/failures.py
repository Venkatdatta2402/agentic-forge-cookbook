"""What to do when a worker fails, loops, or hands back something unusable.

Every notebook in this chapter produced one of these by accident:

    it failed        a tool raised inside a worker, and `Act` quietly retried the whole worker --
                     three model calls to repair an argument, three more agent runs behind them
    it ran out       a specialist hit its iteration cap and returned "Stopping: reached the
                     maximum of 8 iterations... This is partial progress, not a final answer."
                     The supervisor read that as an answer and carried on. (Runtime now ends a
                     stopped run on the agent's best answer instead, which reads even more like a
                     real one -- all the more reason the stop has to arrive as a failure form.)
    it cannot be used  an agent wrote prose instead of calling `submit`; another answered
                     `confident: true` with nothing behind it; a third marked its reply
                     `internal` and the supervisor forwarded it to a journalist anyway

The shape of the fix is the same each time. **A failure has to arrive in the same form as a
success**, so the thing that asked can see it and decide, instead of discovering it as an
exception in the wrong place or as a sentence that happens to start with "Stopping".

Nothing here is clever. It is a try/except, a string check, and a function that says whether a
reply may leave the building -- which is exactly the kind of code that multi-agent systems are
short of, because the interesting part looks like it is happening inside the agents.
"""

from messages import STOPPED, Response, answer

# What `messages.answer` writes into `unknowns` when Runtime stopped the agent. It used to be a
# check on the answer text itself, for Runtime's "Stopping: reached the maximum..." message. A
# stopped run now ends with the agent's best answer instead, which is a real answer and reads
# like one -- so the text can no longer say it was cut short, and the conversation's `stopped`
# is what does.


def failure(reason, detail=""):
    """A failure, in the same shape as an answer."""
    return Response(answer=f"Could not do this: {reason}", confident=False, visibility="internal",
                    unknowns=detail or reason, ask_next="")


def attempt(agent, request, model=Response, retries=0):
    """Run an agent and come back with a form whatever happens.

    Three things can go wrong, and all three come back as a `Response` the caller can read:
    the agent raises, the agent runs out of budget, or the agent answers without filling in the
    form. `retries` counts EXTRA attempts and is 0 by default. Retrying a worker is not like
    retrying an HTTP request: it is another whole agent run, and most of what goes wrong here --
    a missing argument, a budget too small, an agent that will not use the form -- comes out the
    same way the second time. Ask for a retry where you have reason to expect a different answer.
    """
    last = None
    for attempt_number in range(retries + 1):
        try:
            response = answer(agent, request, model=model)
        except Exception as raised:                      # noqa: BLE001 -- the caller decides, not us
            last = failure(f"{agent.name} raised {type(raised).__name__}", str(raised))
            continue
        if isinstance(response, Response) and response.unknowns.startswith(STOPPED):
            # the reason, then what it had got to -- kept, because it is often most of the answer
            last = failure(f"{agent.name} was stopped before finishing",
                           f"{response.unknowns[len(STOPPED) + 2:]} Its answer so far: {response.answer}")
            continue
        return response
    return last


def sendable(response, audience="customer"):
    """Whether this reply may go to somebody outside the company, and why not.

    The checks are dull on purpose. Each one is a failure this chapter actually produced:
    notebook 2's supervisor forwarding a specialist's answer to a journalist; notebook 3's
    engineer reporting `confident: true` about a unit it had misread; notebook 1's one agent
    putting the supplier's name in a document meant for the regulator.
    """
    # Ordered so the reason a reader gets is the most useful one. A failure is also marked
    # internal and also not confident, and being told the least specific of the three is how a
    # caller ends up handling the wrong problem.
    if not response.answer.strip():
        return False, "there is no answer in it"
    if response.answer.startswith("Could not do this"):
        return False, f"it is a failure, not an answer: {response.unknowns[:80]}"
    if not response.confident:
        return False, f"the agent is not confident: {response.unknowns or 'no reason given'}"
    if response.visibility == "internal":
        return False, f"marked internal, and the {audience} is outside the company"
    return True, "ok"


def guard_watchers(board, limit=3):
    """Stop a chain of board watchers from running away.

    A watcher may write, and a write fires watchers, so two fields that watch each other loop
    until the stack gives out. This wraps `set` so a chain deeper than `limit` is refused and
    said out loud, rather than ending in a RecursionError a long way from the cause.
    """
    depth = {"value": 0}
    original = board.set

    def counted(author, field, value):
        if depth["value"] >= limit:
            raise RuntimeError(
                f"watcher chain deeper than {limit} while {author} was setting {field}; "
                "something is watching a field it also writes")
        depth["value"] += 1
        try:
            return original(author, field, value)
        finally:
            depth["value"] -= 1

    board.set = counted
    return board
