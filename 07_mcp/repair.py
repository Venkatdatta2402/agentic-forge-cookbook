"""Reconnecting a dead server, as a decision rather than as a reflex.

`Connection.reconnect()` is the mechanism, and it answers nothing about *when*. How many times
to try, how long to wait, when to give up, and who gets told -- those are policy, and in this
repo policy lives in a component, because `02_agent_runtime` says a component is anything with
`.run(conversation)` and the loop is where decisions belong.

Which is why nothing in `02_agent_runtime` changes to support this. The extension point was
built four chapters ago; `Reconnect` just uses it, the same way `04_memory`'s `Recall` does.

Two decisions in here are worth arguing with rather than reading past.

**It asks the server instead of reading the conversation.** By the time a failure reaches
`conversation.pending` it is a string -- `execute_tool_call` finishes with `str(result)`, so
`Result.kind` is already gone and "the connection died" is indistinguishable from "the tool has
a bug in it". So `Reconnect` does not try to infer. It pings. Checking the world rather than
trusting the report is the same move `verified()` makes in `03_tools`.

**A successful repair is not told to the model.** It goes in the run log and not in the
conversation, for exactly the reason `for_the_model` exists: the model cannot act on "we
reconnected", and spending context on it invites it to reason about infrastructure it does not
control. A repair that *fails* is the opposite -- "the weather service is unreachable" is
something a model can work with, so that one becomes a message.
"""

import time


class Reconnect:
    """A component that repairs a named resource on the conversation when it stops answering.

        runtime = Runtime(loop=[Observe(...), Reconnect(), Think(...), Act(...), Observe()], ...)

    Placement is a real choice, not a detail:

    *Before* `Think`/`Act`, as above, the connection is checked before it is needed, so a call
    rarely fails at all -- at the cost of one ping per iteration. Over stdio that is a round
    trip down a local pipe and effectively free; over HTTP it is a network request per turn.

    *After* the final `Observe`, nothing is paid while things work, and each outage costs one
    wasted iteration -- a model call, a failed tool call, and a turn in the transcript saying so.

    The default is `every_turn=True` because the cheap case is the common one in this chapter and
    a wasted turn is more expensive than a ping. Set it False for HTTP servers or slow links.
    """

    def __init__(self, name="mcp", attempts=3, wait=0.5, backoff=2.0, budget=None,
                 every_turn=True, timeout=2):
        self.name = name
        self.attempts = attempts          # tries per outage
        self.wait = wait                  # seconds before the first retry
        self.backoff = backoff            # multiplier on each subsequent wait
        # Total repairs allowed for the whole run. Without it, a server that dies on startup
        # every time gets `attempts` tries per iteration forever, and the run burns its whole
        # budget restarting a process that is never going to work.
        self.budget = budget
        self.every_turn = every_turn
        self.timeout = timeout
        self.repairs = 0
        self.log = []

    def _note(self, message):
        self.log.append(message)

    def run(self, conversation):
        connection = (getattr(conversation, "resources", None) or {}).get(self.name)
        if connection is None:
            # No such resource on this conversation. Not an error -- the same loop should be
            # usable with local tools, and a component that refused to run without its resource
            # would make the loop's shape depend on how the tools happen to be implemented.
            return None

        if not self.every_turn and not self._suspect(conversation):
            return None

        if connection.healthy(timeout=self.timeout):
            return None

        if self.budget is not None and self.repairs >= self.budget:
            self._note(f"{self.name}: out of repair budget after {self.repairs}")
            return self._give_up(connection)

        wait = self.wait
        for attempt in range(1, self.attempts + 1):
            try:
                connection.reconnect()
            except Exception as e:  # noqa: BLE001 -- a factory that cannot build is the case
                self._note(f"{self.name}: attempt {attempt} failed to start: "
                           f"{type(e).__name__}: {e}")
            else:
                if connection.healthy(timeout=self.timeout):
                    self.repairs += 1
                    self._note(f"{self.name}: reconnected on attempt {attempt}")
                    # Deliberately no message. The repair worked; the conversation is unchanged
                    # from the model's point of view, which is the truth.
                    return None
                self._note(f"{self.name}: attempt {attempt} started but did not answer")

            if attempt < self.attempts:
                time.sleep(wait)
                wait *= self.backoff

        self._note(f"{self.name}: gave up after {self.attempts} attempts")
        return self._give_up(connection)

    def _give_up(self, connection):
        """What the conversation is told when the server is not coming back.

        This one does go to the model, because it changes what is worth attempting. Written as
        a fact about the world rather than as an error, for the same reason `ToolError` messages
        are: the useful half is what to do now, not what went wrong.
        """
        try:
            connection.close()
        except Exception:  # noqa: BLE001 -- already giving up; a failed close changes nothing
            pass
        return {
            "role": "tool",
            "content": (f"The {self.name} service is unreachable and its tools cannot be used. "
                        f"Answer from what you already know, or say what you cannot do."),
        }

    @staticmethod
    def _suspect(conversation):
        """Whether the last turn looks like it hit a dead server.

        A guess, and known to be one. The `kind` is gone by the time a result reaches `pending`,
        so this matches on the phrasing `invoker()` uses for a withheld failure. Good enough to
        skip pings on turns where nothing went wrong, and not good enough to be the only check --
        which is why a match is followed by a ping rather than believed.
        """
        pending = getattr(conversation, "pending", None)
        if isinstance(pending, dict) and "could not be completed" in str(pending.get("content")):
            return True
        last = conversation.messages[-1] if conversation.messages else {}
        return "could not be completed" in str(last.get("content", ""))
