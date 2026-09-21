from llm import chat
from think import branch_label, render, transcript


class Observe:
    # branch_detail decides how much of a forked branch's work reaches the conversation:
    #   None         -- follow each branch's own `explain` setting (the default, see below)
    #   "answer"     -- just what each branch concluded (cheap, and usually all the caller needs)
    #   "transcript" -- each branch's goal, the calls it made and why, what came back, and its
    #                   answer. Costs context, but lets the caller judge the work rather than
    #                   take it on trust -- and makes a branch that ends on a vague sentence
    #                   still useful, because its actual results are right there in the block.
    #
    # Defaulting to None because otherwise this is the SAME decision twice. `explain` is set per
    # branch by the Think that delegated it -- asking for a branch's reasoning is asking to read
    # its working. Requiring the loop declaration to say "transcript" as well let the two
    # disagree, both ways and both silently: explain=True with the default Observe generated
    # rationale inside every branch and then threw it away, and "transcript" over explain=False
    # printed calls and results with the reasons missing.
    #
    # It also fixes a granularity mismatch. One Observe handles a whole fork, but `explain` is
    # per branch -- so a fork of three branches, one of which needs auditing, could not be
    # expressed at all. Now each branch is rendered by its own setting, and branch_detail is an
    # override for when the caller genuinely wants to force one or the other.
    def __init__(self, event=None, extractors=None, branch_detail=None):
        self.event = event
        self.extractors = extractors or {}
        self.branch_detail = branch_detail

    def run(self, conversation):
        if self.event is not None:
            return {"role": "user", "content": self.event}

        effect = getattr(conversation, "pending", None)
        conversation.pending = None

        # Nothing to observe. Act leaves pending empty whenever the decision it applied changed
        # the conversation rather than the world -- a replan is the clear case. Returning None
        # means Runtime appends nothing, which is the honest record: no effect, no observation.
        if effect is None:
            return None

        if isinstance(effect, list):
            if not effect:
                # a fork that only allocated empty branches: nothing ran, so there are no results
                # to report. Saying which branches are now waiting is the useful thing -- it is
                # exactly what the next Think needs in order to fill them in.
                waiting = list(getattr(conversation, "branches", None) or {})
                return {"role": "tool",
                        "content": f"Branches waiting for a thought: {', '.join(waiting)}."}
            # a fork's (or advance_branch's) results: one entry per branch. Whatever the detail
            # level, this stays a SINGLE message -- splicing a branch's turns into the outer
            # conversation as separate messages would let the next Think read them as its own
            # prior turns, and two branches' turns would interleave into nonsense.
            # tagged so the next Think can tell branch results from an ordinary tool result. It
            # uses that to withhold `delegate` for exactly one turn -- see Think._can_fork.
            return {"role": "tool", "source": "branches",
                    "content": "\n".join(self._render(b) for b in effect)}

        extractor = self.extractors.get(effect["tool"])
        if extractor is not None:
            content = extractor(effect)
        else:
            content = chat([{
                "role": "user",
                "content": (
                    f"Conversation so far:\n{render(conversation)}\n\n"
                    f"The tool `{effect['tool']}` was just called and returned:\n{effect['content']}\n\n"
                    "State this result as one plain factual sentence.\n\n"
                    "Copy every value exactly as the tool gave it, in the units the tool gave it. "
                    "Do not convert, calculate, round, combine or reformat anything -- not even "
                    "when the conversation asks for a different unit or form. Working a value out "
                    "is a later step's job and it has tools for exactly that; doing it quietly "
                    "here means the number reaching the conversation came from nowhere and nothing "
                    "checked it. Use the conversation only to say what the value refers to.\n\n"
                    "Keep anything the TOOL itself said about what to do next -- a suggestion, a "
                    "constraint, an alternative to try, the reason it refused. That is part of "
                    "the result, and the agent reading this has to act on it.\n\n"
                    "Add nothing of your own: no speculation about why the result matters, how it "
                    "will be used, or what should happen now. No preamble, no commentary."
                ),
            }])

        step_outputs = getattr(conversation, "step_outputs", None)
        if step_outputs is None:
            step_outputs = conversation.step_outputs = []
        step_outputs.append(effect["content"])

        return {"role": "tool", "content": content}

    def _render(self, branch):
        if self.branch_detail is None:
            detailed = branch.get("explain", False)     # whatever the delegating Think asked for
        else:
            detailed = self.branch_detail == "transcript"
        return self._transcript(branch) if detailed else self._answer(branch)

    @staticmethod
    def _answer(branch):
        return f"[{branch_label(branch)}] {branch['content']}"

    # the compacted form lives in think.py beside render(), because ExpandThink needs it too --
    # a child inherits exactly this block as the state it grows from
    _transcript = staticmethod(transcript)
