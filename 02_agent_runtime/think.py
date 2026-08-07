import json
import re

from openai import BadRequestError
from pydantic import BaseModel

from llm import complete, current_model, extract


def render(conversation):
    return "\n".join(f"{m['role']}: {m['content']}" for m in conversation.messages)


_BUDGET_LABELS = {
    "seconds": "seconds",
    "iterations": "iterations",
    "tool_calls": "tool calls",
    "tokens": "tokens",
}


def _delegate_tool(conversation):
    # built fresh each call so its description can mention whatever budget actually remains --
    # a plain data string, not a decision; Think still does all the judging about how to split it.
    budget = getattr(conversation, "remaining_budget", None) or {}
    parts = [f"{v:.0f} {_BUDGET_LABELS[k]}" for k, v in budget.items() if v is not None]
    budget_note = (
        f" The following remain, shared across every branch: {', '.join(parts)}. Split each of "
        "these between branches as whole numbers, in proportion to how hard each sub-goal looks."
        if parts else ""
    )
    return {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Delegate to independent sub-agents. Use persistent=true, broadcast=true for an "
                "ongoing exchange like a debate, where branches should see each other's turns. "
                "Use persistent=true, broadcast=false for independent specialists you may return "
                "to individually later. Leave both false for a one-off split of work, or for "
                "running the same goal several times with different sampling to check "
                f"consistency (vary temperature/top_p, not the goal, for that).{budget_note}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "persistent": {"type": "boolean"},
                    "broadcast": {"type": "boolean"},
                    "branches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "short name, needed if persistent"},
                                "goal": {
                                    "type": "string",
                                    "description": (
                                        "A complete, self-contained instruction for an agent with NO "
                                        "other context -- it will not see this conversation or any "
                                        "other branch's goal. Include everything it actually needs "
                                        "(e.g. the specific question or values involved). Never write "
                                        "it as if the reader already knows the broader request, and "
                                        "never reference 'the other task' or another branch."
                                    ),
                                },
                                "temperature": {"type": "number"},
                                "top_p": {"type": "number"},
                                "max_seconds": {"type": "number"},
                                "max_iterations": {"type": "integer"},
                                "max_tool_calls": {"type": "integer"},
                                "max_tokens": {"type": "integer"},
                            },
                            "required": ["goal"],
                        },
                    },
                },
                "required": ["branches"],
            },
        },
    }


class Think:
    # the default, general-purpose Think: decides a tool call, a delegation (fork), or a final
    # answer, all through the one native tool-calling mechanism -- "delegate" is just one more
    # tool in the list, so no separate ToolCallThink/ForkThink is needed for this. Sampling
    # (model/temperature/top_p) is per-instance, so a forked branch can run with its own settings.
    def __init__(self, tools=None, retries=3, model=None, temperature=None, top_p=None, allow_fork=True):
        self.tools = tools or []
        self.retries = retries
        # left as None, the active provider's model is looked up at request time rather than
        # frozen here -- so llm.use() still takes effect on Thinks constructed before it.
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        # leaf-level branches doing one concrete task shouldn't be tempted to delegate further --
        # only Thinks that should be able to fork (usually the outer/orchestrating one) need this.
        self.allow_fork = allow_fork

    def _sampling_kwargs(self):
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs

    def _build_messages(self, conversation):
        # reconstructs the REAL native tool-calling shape (an assistant message carrying a
        # structured tool_calls entry, followed by a tool-role reply with a matching id) --
        # rather than flattening everything to prose. Models are trained to recognize "I already
        # did this" from exactly this structure; narrated text of the same information turned out
        # unreliable (the model would sometimes just call the same tool again). A "decision" of
        # type tool_call is always immediately followed by exactly one "tool" observation in our
        # own conversation, so pairing them up by position is enough -- no id needs to be stored.
        #
        # What DOES need storing is whatever the model attached to its own tool call. Thinking
        # models return an opaque reasoning token there (Gemini 3.x calls it a thought_signature,
        # under extra_content) and reject a replayed function call that arrives without it, since
        # without it they cannot resume the chain of thought that produced the call. We keep it on
        # the decision and hand it straight back -- see run() below. Verified against the API: the
        # id may be freely re-synthesized, but dropping extra_content is a hard 400.
        messages = [{
            "role": "system",
            "content": (
                "If the user's request has already been fully answered by a tool result shown "
                "below, respond in plain text with that answer -- do not call the tool again."
            ),
        }]
        pending_call_id = None
        for m in conversation.messages:
            role = m["role"]
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": m["content"]})
            elif role == "decision" and m.get("type") == "tool_call":
                pending_call_id = f"call_{len(messages)}"
                tool_call = {
                    "id": pending_call_id,
                    "type": "function",
                    "function": {"name": m["tool"], "arguments": m["content"]},
                }
                if m.get("extra_content"):
                    tool_call["extra_content"] = m["extra_content"]
                messages.append({
                    "role": "assistant",
                    # the model's own words about why it made this call, when the provider emits
                    # any alongside a tool call -- most don't, and None is the normal case
                    "content": m.get("reasoning"),
                    "tool_calls": [tool_call],
                })
            elif role == "tool" and pending_call_id is not None:
                messages.append({"role": "tool", "tool_call_id": pending_call_id, "content": m["content"]})
                pending_call_id = None
            else:
                # plan/critique/fork decisions, or a stray tool observation with no pending
                # call (e.g. a fork's merged result) -- narrated as the agent's own past turn.
                messages.append({"role": "assistant", "content": m["content"]})
        return messages

    def run(self, conversation):
        messages = self._build_messages(conversation)
        all_tools = [*self.tools, _delegate_tool(conversation)] if self.allow_fork else self.tools
        for attempt in range(self.retries):
            try:
                response = complete(
                    model=self.model or current_model(), messages=messages, tools=all_tools,
                    **self._sampling_kwargs()
                )
            except BadRequestError:
                messages.append({
                    "role": "user",
                    "content": "That call could not be processed. Try again, calling one of the available tools correctly.",
                })
                continue

            message = response.choices[0].message
            if not message.tool_calls:
                if not message.content:
                    messages.append({"role": "assistant", "content": message.content or ""})
                    messages.append({"role": "user", "content": "Write out your final answer as text."})
                    continue
                return {"role": "decision", "type": "respond", "content": message.content}

            call = message.tool_calls[0]
            if call.function.name == "delegate":
                args = json.loads(call.function.arguments)
                branches = args["branches"]
                return {
                    "role": "decision",
                    "type": "fork",
                    "content": "Forked into: " + "; ".join(b["goal"] for b in branches),
                    "persistent": args.get("persistent", False),
                    "broadcast": args.get("broadcast", False),
                    "branches": branches,
                }

            decision = {
                "role": "decision",
                "type": "tool_call",
                "tool": call.function.name,
                "content": call.function.arguments,
            }
            # Keep whatever the model produced alongside the call rather than discarding it.
            # `extra_content` is the provider's own opaque reasoning state (Gemini 3.x's
            # thought_signature lives here) and _build_messages hands it back verbatim on the
            # next turn -- without it a thinking model 400s the moment it sees its own replayed
            # call. `reasoning` is the human-readable counterpart for providers that emit one;
            # it is real content for later Thinks to read, not just plumbing.
            if getattr(call, "extra_content", None):
                decision["extra_content"] = call.extra_content
            if message.content:
                decision["reasoning"] = message.content
            return decision
        raise ValueError(f"Failed to get a valid decision after {self.retries} attempts")


class PlanStep(BaseModel):
    tool: str
    args: dict[str, str]


class Plan(BaseModel):
    steps: list[PlanStep]


class ContinueOrFinish(BaseModel):
    done: bool
    final_answer: str = ""
    additional_steps: list[PlanStep] = []


def _latest_plan(conversation):
    plan_index, plan = None, None
    for i, m in enumerate(conversation.messages):
        if "steps" in m:
            plan_index, plan = i, m
    return plan_index, plan


def plan_progress(conversation):
    # the *latest* plan only -- a replan appends a new "plan" message, and progress on it
    # starts back at 0, even though earlier plans' decisions are still sitting in history.
    # Counts *decided* steps (a decision exists) -- correct for picking the next step to
    # decide, but NOT the same as *executed* (Act has actually run for it) -- see plan_done.
    plan_index, plan = _latest_plan(conversation)
    if plan is None:
        return None, 0
    decided = sum(1 for m in conversation.messages[plan_index + 1:] if m.get("role") == "decision")
    return plan, decided


_STEP_REF = re.compile(r"\$(?:step)?(\d+)(?:\.(\w+))?")


def step_dependencies(step):
    # which earlier step indices this step's args reference -- read straight off the same
    # "$stepN"/"$stepN.field" syntax Observe already resolves, no new plan format needed.
    deps = set()
    for value in step["args"].values():
        match = _STEP_REF.fullmatch(value) if isinstance(value, str) else None
        if match is not None:
            deps.add(int(match.group(1)))
    return deps


def resolve_args(step, results):
    # results: {step_index: raw_content}. A step is only ever resolved once every index in
    # step_dependencies(step) is already a key in results -- the caller's job, not this one's.
    args = {}
    for key, value in step["args"].items():
        match = _STEP_REF.fullmatch(value) if isinstance(value, str) else None
        if match is None:
            args[key] = value
            continue
        index, field = match.groups()
        raw = results[int(index)]
        args[key] = json.loads(raw)[field] if field is not None else raw
    return args


def plan_done(conversation):
    # deliberately counts *executed* steps (a "tool" observation exists), not decided ones --
    # right after the last step's decision is produced but before its Act has run, decided
    # already equals len(steps); using that here would end a repeating loop one beat too early.
    plan_index, plan = _latest_plan(conversation)
    if plan is None:
        return True
    executed = sum(1 for m in conversation.messages[plan_index + 1:] if m.get("role") == "tool")
    return executed >= len(plan["steps"])


class PlannerThink(Think):
    # decides the full plan, including every argument -- either a literal value already known
    # from the request, or "$stepN" referencing an earlier step's not-yet-known result. Later
    # steps' arguments are then pure substitution (Observe's job), never a fresh decision. Called
    # again after a plan finishes (e.g. as the target of an outer repeat_from), it switches to
    # deciding whether the request is actually answered yet, or another round of steps is needed.
    def __init__(self, tools, outputs=None):
        self.tools = tools
        self.outputs = outputs or {}

    def _tool_descriptions(self):
        return "\n".join(
            f"- {t['function']['name']}({', '.join(t['function']['parameters']['properties'])}) "
            f"returns: {self.outputs.get(t['function']['name'], 'a single value')}"
            for t in self.tools
        )

    def run(self, conversation):
        plan, _ = plan_progress(conversation)
        if plan is None:
            return self._make_plan(conversation)
        return self._continue_or_finish(conversation)

    def _make_plan(self, conversation):
        request = conversation.messages[-1]["content"]
        plan = extract(
            f"A user asked: {request!r}\n"
            f"Available tools, with their exact argument names and what they return:\n{self._tool_descriptions()}\n\n"
            "Produce, in order, every tool call needed to fully answer this request. For each "
            "step, give the tool name and a value for every one of that tool's argument names "
            "listed above, as strings -- the args object's keys must be exactly those argument "
            "names, nothing else. If an "
            "argument's value is already known from the request, use that literal value. If an "
            "argument's value can only be known after an earlier step's tool actually runs, "
            'reference it with exactly the string "$stepN" (0-indexed, e.g. "$step0" for the '
            'first step\'s result) instead of guessing -- never abbreviate this to "$0" or '
            'anything else. If that earlier step returns a single value, "$stepN" alone is the '
            'whole value. If it returns a JSON object with named fields (see "returns" above), '
            'reference just the one field you need with "$stepN.fieldname" -- never forward the '
            "whole object when only one field is actually needed. "
            "Return JSON matching the given schema.",
            Plan,
        )
        conversation.messages.append({
            "role": "plan",
            "content": plan.model_dump_json(),
            "steps": [s.model_dump() for s in plan.steps],
        })
        first = plan.steps[0]
        return {
            "role": "decision",
            "type": "tool_call",
            "tool": first.tool,
            "content": json.dumps(first.args),
        }

    def _continue_or_finish(self, conversation):
        start_index = len(getattr(conversation, "step_outputs", []))
        result = extract(
            f"Conversation so far:\n{render(conversation)}\n\n"
            f"Available tools, with their exact argument names and what they return:\n{self._tool_descriptions()}\n\n"
            "All previously planned tool calls have finished running. Decide whether the "
            'conversation so far fully answers the user\'s original request. If it does, set '
            '"done" to true and write the final answer in "final_answer". If it does not, set '
            '"done" to false and list the additional tool call steps still needed, in '
            f'"additional_steps", the same format as before. So far {start_index} tool call(s) '
            f'have completed, referenceable as "$step0" through "$step{start_index - 1}". Any '
            "new step you add now runs in order starting immediately, so if a later new step "
            f'needs an earlier new step\'s result, reference it starting from "$step{start_index}" '
            "onward, continuing that same count. Return JSON matching the given schema.",
            ContinueOrFinish,
        )
        if result.done:
            return {"role": "decision", "type": "respond", "content": result.final_answer}
        conversation.messages.append({
            "role": "plan",
            "content": json.dumps([s.model_dump() for s in result.additional_steps]),
            "steps": [s.model_dump() for s in result.additional_steps],
        })
        first = result.additional_steps[0]
        return {
            "role": "decision",
            "type": "tool_call",
            "tool": first.tool,
            "content": json.dumps(first.args),
        }


class Selection(BaseModel):
    keep_indices: list[int]
    reasoning: str


class SelectThink(Think):
    # pruning: reads the RAW, not-yet-merged results of a fork (still sitting in
    # conversation.pending as a list) and decides which survive. Produces a "prune" decision;
    # Act applies it mechanically (drop everything not in keep_indices) -- SelectThink judges,
    # it doesn't touch the list itself.
    def __init__(self, keep=1):
        self.keep = keep

    def run(self, conversation):
        candidates = getattr(conversation, "pending", None) or []
        listing = "\n".join(f"{i}. [{c.get('goal', '(no goal)')}] {c['content']}" for i, c in enumerate(candidates))
        result = extract(
            f"Conversation so far:\n{render(conversation)}\n\n"
            f"Candidate results from {len(candidates)} independent branches:\n{listing}\n\n"
            f"Score each against the original goal, and pick the best {self.keep} to continue "
            "exploring, by their index in the list above. Discard the rest.\n"
            # the schema has to be shown, not just referred to -- without it the model invents its
            # own field names (e.g. a list of scored items) and only recovers via extract()'s
            # retry loop, if at all. Same reason DebugThink.fix() embeds its schema.
            f"Return JSON matching this schema:\n{Selection.model_json_schema()}",
            Selection,
        )
        # the model sometimes returns an index that doesn't exist, or more than it was asked for.
        # Act indexes straight into pending, so the guarantee that these are real indices has to
        # be made here, where the candidate list is actually in scope.
        valid = [i for i in dict.fromkeys(result.keep_indices) if 0 <= i < len(candidates)]
        return {
            "role": "decision",
            "type": "prune",
            "content": result.reasoning,
            "keep_indices": valid[: self.keep] or list(range(min(self.keep, len(candidates)))),
        }


class DebugThink:
    # not a Think: no run(conversation) -- Act owns this privately, it's never declared into a Runtime's loop
    def __init__(self, schemas):
        self.schemas = schemas

    def fix(self, tool_name, args, error, attempts):
        schema = self.schemas[tool_name]
        history = "\n".join(f"- args {a['args']} failed with: {a['error']}" for a in attempts)
        corrected = extract(
            f"Calling the tool `{tool_name}` with arguments {args} failed with this error:\n{error}\n\n"
            f"Earlier failed attempts on this same call:\n{history}\n\n"
            f"Fix the arguments to match this JSON schema:\n{schema.model_json_schema()}\n"
            "Return corrected JSON matching the schema.",
            schema,
        )
        return corrected.model_dump_json()


class Reflection(BaseModel):
    critique: str
    improvements: str
    good_enough: bool


class ReflectionThink(Think):
    # deliberately doesn't write a final answer -- it only judges and suggests. Its output isn't
    # even a "decision" (nothing for Act to execute); it's a "critique" message that goes straight
    # back into the conversation, and the loop returns to a real Think (e.g. plain Think) next,
    # which reads the critique/good_enough as context and makes its own ordinary decision --
    # try again differently, or respond -- with no separate retry mechanism needed anywhere.
    def run(self, conversation):
        result = extract(
            f"Conversation so far:\n{render(conversation)}\n\n"
            "Critique whether the most recent result actually answers the original request and "
            "was carried out well. Say what was good and what wasn't. If it's not good enough, "
            "describe concrete improvements to try next. Do not write the final answer yourself.\n"
            f"Return JSON matching this schema:\n{Reflection.model_json_schema()}",
            Reflection,
        )
        return {
            "role": "critique",
            "content": result.critique,
            "improvements": result.improvements,
            "good_enough": result.good_enough,
        }
