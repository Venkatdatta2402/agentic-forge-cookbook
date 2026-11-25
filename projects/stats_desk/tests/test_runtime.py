import asyncio
import time

from pydantic import BaseModel

from stats_desk.agents.agent import Agent, Stopped, answer
from stats_desk.core import meter
from stats_desk.core.runtime import Act, AsyncRuntime, Conversation, Observe, Think
from stats_desk.core.tools import Registry, Tool, ToolError, invoke
from stubs import call, calls, say


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def refuse(x: str) -> str:
    raise ToolError(f"no such thing as {x}")


async def crash() -> str:
    raise KeyError("GROQ_API_KEY")


async def test_invoke_sorts_every_ending_into_a_kind():
    assert (await invoke(Tool(add), {"a": 1, "b": 2})).content == "3"
    assert (await invoke(Tool(add), {"a": 1, "b": 2, "c": 3})).kind == "bad_arguments"
    assert (await invoke(Tool(refuse), {"x": "narnia"})).kind == "refused"
    broken = await invoke(Tool(crash), {})
    assert broken.kind == "broken"
    assert "GROQ_API_KEY" not in broken.as_seen          # ours to fix, not the model's to read


def loop(tools, **kw):
    reg = Registry(tools)
    return AsyncRuntime([Think(reg, "m"), Act(reg), Observe()], **kw)


async def test_tool_then_answer(model):
    model.steps += [call("add", a=2, b=3), lambda req: say(f"It is {model.last_tool_result(req)}.")]
    convo = Conversation(system="s", messages=[{"role": "user", "content": "2+3?"}])
    await loop([Tool(add)]).run(convo)
    assert convo.messages[-1]["content"] == "It is 5."
    # history went back to the model in the native shape: assistant tool_calls, then tool
    sent = model.requests[1]["messages"]
    assert sent[-2]["tool_calls"][0]["function"]["name"] == "add"
    assert sent[-1]["role"] == "tool"


def test_old_discovery_results_are_shortened_but_numbers_never_are():
    from stats_desk.core.runtime import history
    msgs = [{"role": "user", "content": "q"}]
    for i, (name, content) in enumerate([("fetch", "table t (8 rows)\ncolumns...\nrows..."),
                                         ("sql", "2 rows\nmonth | value\nJan | 16.0"),
                                         ("fetch", "table u (3 rows)\nmore"),
                                         ("sql", "1 row\nx\n1")]):
        msgs.append({"role": "decision", "type": "tool_calls", "calls": [
            {"id": f"c{i}", "name": name, "arguments": "{}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": content})
    out = [m["content"] for m in history(Conversation(messages=msgs), shorten=("fetch",)) if m["role"] == "tool"]
    assert out[0] == "table t (8 rows)\ncolumns...  [earlier result, shortened]"
    assert out[1] == "2 rows\nmonth | value\nJan | 16.0"           # sql: the numbers stay
    assert out[2] == "table u (3 rows)\nmore"                      # recent: kept whole


async def test_two_calls_in_one_turn_run_concurrently(model):
    async def slow(name: str) -> str:
        await asyncio.sleep(0.3)
        return name

    model.steps += [calls(("slow", {"name": "a"}), ("slow", {"name": "b"})), say("done")]
    convo = Conversation(messages=[{"role": "user", "content": "go"}])
    started = time.monotonic()
    await loop([Tool(slow)]).run(convo)
    assert time.monotonic() - started < 0.55          # two 0.3 s calls, overlapping
    assert [m["content"] for m in convo.messages if m["role"] == "tool"] == ["a", "b"]


async def test_repeated_identical_call_stops_and_answers_from_results(model):
    model.steps += [call("add", a=1, b=1), call("add", a=1, b=1),
                    say("From the results: 2.")]           # finish(), no tools offered
    convo = Conversation(messages=[{"role": "user", "content": "1+1"}])
    await loop([Tool(add)]).run(convo)
    assert "identical" in convo.stopped
    assert convo.messages[-1]["content"] == "From the results: 2."
    assert "tools" not in model.requests[-1]


async def test_a_form_filling_agent_gets_no_prose_answer_when_stopped(model):
    # live run 5: the prose written at a budget stop invented 4 of 16 values
    model.steps += [call("add", a=1, b=1), call("add", a=1, b=1)]      # no step for finish()
    convo = Conversation(messages=[{"role": "user", "content": "1+1"}])
    await loop([Tool(add)], answer_on_stop=False).run(convo)
    assert convo.messages[-1]["content"].startswith("Stopping:")
    assert len(model.requests) == 2


async def test_low_budget_tells_the_model_to_finish(model):
    model.steps += [call("add", a=1, b=2), say("3")]
    convo = Conversation(messages=[{"role": "user", "content": "1+2"}])
    await loop([Tool(add)], max_tokens=4_200).run(convo)
    assert "Finish now" in model.requests[1]["messages"][0]["content"]
    assert "Finish now" not in model.requests[0]["messages"][0]["content"]


async def test_token_budget_stops_the_run(model):
    model.steps += [call("add", a=1, b=2), call("add", a=3, b=4), say("3 and 7")]
    convo = Conversation(messages=[{"role": "user", "content": "sums"}])
    await loop([Tool(add)], max_tokens=600).run(convo)      # each stub call costs 340
    assert "token budget" in convo.stopped


async def test_usage_is_attributed_to_the_agent(model):
    model.steps += [say("hi")]
    with meter.metered() as usage, meter.acting_as("labour"):
        await loop([]).run(Conversation(messages=[{"role": "user", "content": "x"}]))
    assert usage.by("agent") == {"labour": 340}


class Form(BaseModel):
    total: int


async def test_agent_finishes_by_submitting_its_form(model):
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m")
    model.steps += [call("add", a=2, b=2), call("json", total=4)]
    result = await answer(agent, "2+2", Form)
    assert result == Form(total=4)


async def test_prose_instead_of_the_form_gets_one_reminder(model):
    agent = Agent("calc", "You add.", model="m")
    model.steps += [say("The total is 4."), call("json", total=4)]
    assert await answer(agent, "2+2", Form) == Form(total=4)


async def test_an_agent_that_never_hands_in_comes_back_as_stopped(model):
    agent = Agent("calc", "You add.", model="m")
    model.steps += [say("4"), say("4"), say("4")]                 # three tries, then it stops
    result = await answer(agent, "2+2", Form)
    assert isinstance(result, Stopped) and len(model.requests) == 3


async def test_a_submit_decided_on_the_call_that_crosses_the_budget_still_counts(model):
    # live run 2: a skeptic's verdict was thrown away because its hand-in call crossed the budget
    agent = Agent("calc", "You add.", model="m", budget=300)       # one stub call costs 340
    model.steps += [call("json", total=4)]
    assert await answer(agent, "2+2", Form) == Form(total=4)


def rejected(generated):
    """Groq's 400 for a malformed tool call, carrying what the model generated."""
    import httpx
    from openai import BadRequestError

    def step(request):
        raise BadRequestError("tool call validation failed", response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")),
            body={"error": {"code": "tool_use_failed", "message": "attempted to call tool which was not in request.tools",
                            "failed_generation": generated}})
    return step


async def test_a_hand_in_made_under_the_wrong_name_is_made_as_the_right_tool(model):
    # baseline runs: gpt-oss called 'json' or 'commentary' 9 times with a full hand-in
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m")
    model.steps += [rejected('{"name": "commentary", "arguments": {"total": 4}}')]
    assert await answer(agent, "2+2", Form) == Form(total=4)
    assert len(model.requests) == 1                                  # no second model call needed


async def test_a_rejected_call_whose_contents_fit_no_tool_gets_the_real_names(model):
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m")
    model.steps += [rejected('{"name": "commentary", "arguments": We cannot answer that from the tables}'),
                    call("json", total=4)]
    assert await answer(agent, "2+2", Form) == Form(total=4)
    told = model.requests[1]["messages"][-1]["content"]
    assert "Your tools are exactly: `add`, `json`" in told


async def test_a_step_named_by_code_is_asked_for_once_then_left_to_the_model(model):
    """The lead's fixed steps: code names the tool for one try. A model set on
    another tool is refused by Groq, and that call is not rescued as the other tool (span tests:
    it was, as the brief); the next try is the model's own choice, told which tool to call."""
    def must_add(conversation):
        return None if any(m.get("role") == "tool" for m in conversation.messages) else "add"
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m", choose=must_add)
    model.steps += [rejected('{"name": "json", "arguments": {"total": 4}}'),          # fits json, not add
                    call("add", a=2, b=2), call("json", total=4)]
    assert await answer(agent, "2+2", Form) == Form(total=4)
    first, second, third = (r.get("tool_choice") for r in model.requests)
    assert first == {"type": "function", "function": {"name": "add"}} and second == third == "required"
    assert model.requests[1]["messages"][-1]["content"].endswith("Call `add` now.")


async def test_prose_where_a_tool_call_was_required_gets_its_own_words_back_and_the_next_move(model):
    """Live run: an analyst retried on Groq's "issue the call again" wrote "this
    cannot be computed with a single tool" three times, and spent its budget on no analysis."""
    import httpx
    from openai import BadRequestError

    def prose(request):
        raise BadRequestError("Tool choice is required, but model did not call a tool", response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")),
            body={"error": {"code": "tool_use_failed", "message": "Tool choice is required, but model did not call a tool",
                            "failed_generation": "This cannot be computed with a single tool."}})
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m", when_stuck="Run the closest analysis now.")
    model.steps += [prose, call("json", total=4)]
    assert await answer(agent, "2+2", Form) == Form(total=4)
    told = model.requests[1]["messages"][-1]["content"]
    assert told == "You wrote this instead of calling a tool:\nThis cannot be computed with a single tool.\n\nRun the closest analysis now."


def test_an_analyst_stopped_before_handing_in_has_its_results_handed_in_by_code():
    """Span test: code forced an out-of-budget analyst to hand in, the model called another
    analysis 3 times instead, each rejected and paid for. Now code hands in what it found."""
    from types import SimpleNamespace

    from stats_desk.agents.analyst import Finding, handed_in_by_code
    analyses = SimpleNamespace(results={"value-123456": "result value-123456\nUR, 2025 | everyone\nvalue: 3.1%"})
    transcript = [{"role": "tool", "kind": "ok", "content": "result value-123456\nUR, 2025 | everyone\nvalue: 3.1%"},
                  {"role": "tool", "kind": "refused", "content": "no education 'All ages'"}]
    stopped = Stopped(agent="analyst f1", reason="reached its token budget of 12,000")
    found = handed_in_by_code(analyses, "f1", "the task", transcript, stopped)
    assert isinstance(found, Finding) and found.results == ["value-123456"]
    assert found.claim.startswith("no claim: the analyst was stopped (reached its token budget")
    assert found.figures == {"value-123456": "UR, 2025 | everyone\nvalue: 3.1%"} and "cut short" in found.flags[0]
    # nothing found: nothing to hand in
    assert handed_in_by_code(analyses, "f1", "the task", transcript[1:], stopped) is stopped
    # a list of the tables is no figure
    analyses.results["tables-abcdef"] = "result tables-abcdef\nrelease: ..."
    only_tables = [{"role": "tool", "kind": "ok", "content": "result tables-abcdef\nrelease: ..."}]
    assert handed_in_by_code(analyses, "f1", "the task", only_tables, stopped) is stopped


def test_a_hand_in_must_say_when_its_figures_are_on_another_setting_than_asked():
    """Span tests: 4 of 4 analysts used figures on PS+SS and never said so, the task asking CWS."""
    from types import SimpleNamespace

    import pytest

    from stats_desk.agents.analyst import check_settings
    from stats_desk.core.tools import ToolError
    analyses = SimpleNamespace(results={
        "gap-123456": "result gap-123456\nnote: these figures are for measure PS+SS, not CWS as asked "
                      "(CWS isn't published for this)\nnote: these figures are for age 15 years and above, "
                      "not all ages as asked (all ages isn't published for this)\nUR, 2025 | ...",
        "value-abcdef": "result value-abcdef\nUR, 2025 | everyone\nvalue: 3.1%"})
    with pytest.raises(ToolError) as refused:
        check_settings(analyses, ["gap-123456"], "UR is 11.2% against 3.1% overall.")
    assert "measure PS+SS, not CWS" in str(refused.value) and "age 15 years and above" in str(refused.value)
    with pytest.raises(ToolError) as refused:      # one of the two said
        check_settings(analyses, ["gap-123456"], "On usual status, UR is 11.2% against 3.1%.")
    assert "measure" not in str(refused.value) and "age 15" in str(refused.value)
    check_settings(analyses, ["gap-123456"], "On PS+SS, ages 15+, UR is 11.2% against 3.1%.")
    check_settings(analyses, ["gap-123456"], "For persons 15 and above, usual status: 11.2%.")
    check_settings(analyses, ["value-abcdef"], "UR is 3.1%.")       # no note, nothing to say


async def test_a_step_with_no_usable_reply_ends_the_agent_instead_of_crashing(model):
    """Span test: an analyst's 4 rejected tries raised out of the whole question."""
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m", answer_on_stop=False)
    model.steps += [rejected("We cannot answer that")] * 4
    result = await answer(agent, "2+2", Form)
    assert isinstance(result, Stopped) and result.reason.startswith("no usable reply")


async def test_retries_stop_at_the_budget(model):
    """Every try is paid in full: an analyst's 4 rejected tries cost 17k on a 12k budget."""
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m", answer_on_stop=False, budget=50)
    model.steps += [rejected("We cannot answer that")] * 4
    result = await answer(agent, "2+2", Form)
    assert isinstance(result, Stopped) and "rejected" in result.reason and len(model.requests) == 1


async def test_half_a_thought_sent_as_the_reply_gets_its_own_words_back_too(model):
    """Groq calls it "could not be parsed", not "did not call a tool": an eval analyst sent "We have
    change result. Need claim." and got the generic message."""
    import httpx
    from openai import BadRequestError

    def note(request):
        raise BadRequestError("Parsing failed", response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")),
            body={"error": {"code": "output_parse_failed",
                            "message": "Parsing failed. The model generated output that could not be parsed.",
                            "failed_generation": "We have change result. Need claim."}})
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m", when_stuck="Hand in now.")
    model.steps += [note, call("json", total=4)]
    assert await answer(agent, "2+2", Form) == Form(total=4)
    assert model.requests[1]["messages"][-1]["content"].startswith(
        "You wrote this instead of calling a tool:\nWe have change result. Need claim.")


async def test_a_step_code_insists_on_stays_named_after_a_refused_try(model):
    """An eval analyst out of budget, told to hand in, wrote another analysis; Groq refused it, and
    the next try, left to the model, was another analysis that stopped it with its figures lost."""
    agent = Agent("calc", "You add.", tools=Registry([Tool(add)]), model="m", choose=lambda c: ("json", True))
    model.steps += [rejected('{"name": "add", "arguments": {"a": 2, "b": 2}}'), call("json", total=4)]
    assert await answer(agent, "2+2", Form) == Form(total=4)
    assert [r["tool_choice"] for r in model.requests] == [{"type": "function", "function": {"name": "json"}}] * 2


async def test_a_hand_in_written_as_text_is_read_as_the_form_with_the_same_checks(model):
    async def at_most_ten(form):
        if form.total > 10:
            raise ToolError("total must be at most 10")
        return form
    agent = Agent("calc", "You add.", model="m", form_build=at_most_ten)
    model.steps += [say('```json\n{"total": 40}\n```'), say('{"total": 4}')]
    assert await answer(agent, "2+2", Form) == Form(total=4)
    assert "read as your hand-in, and refused: total must be at most 10" in model.requests[1]["messages"][-1]["content"]
