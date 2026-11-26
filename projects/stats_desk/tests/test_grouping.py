"""The task check's grouping call (agents/grouping.py): what code makes of the reply. The model
is scripted; how well the real one groups is evals/grouping.py's to measure."""

from stats_desk.agents.grouping import groups_in
from stubs import call, say
from test_runtime import rejected

NAMES = [("religion", "Islam"), ("gender", "female"), ("gender", "male"), ("state", "Kerala"),
         ("education", "8.post graduate & above")]


async def test_each_group_is_one_value_per_field_and_a_list_is_groups_side_by_side(model):
    model.steps += [call("groups", groups=[
        {"religion": "Islam", "gender": "female", "state": "Kerala"},
        {"gender": ["female", "male"], "state": "Kerala"},          # women and men in Kerala
        {"education": "post graduate & above", "gender": "Female"},  # no number, other case
        {"state": "Nowhere"}, {"caste": "x"}, "not a group"]),
    ]
    got = await groups_in("t", NAMES)
    assert got == [{"religion": "Islam", "gender": "female", "state": "Kerala"},
                   {"gender": "female", "state": "Kerala"}, {"gender": "male", "state": "Kerala"},
                   {"education": "8.post graduate & above", "gender": "female"}]
    # the reply's form allows only the fields and values found in the task
    group = model.requests[0]["tools"][0]["function"]["parameters"]["properties"]["groups"]["items"]
    assert group["properties"]["gender"]["enum"] == ["female", "male"] and group["additionalProperties"] is False


async def test_a_reply_groq_turned_down_is_read_from_what_it_says_was_written(model):
    model.steps += [rejected('{"name": "groups", "arguments": {"groups": [{"gender": "female", "state": "Kerala"}]}}')]
    kept = []
    assert await groups_in("t", NAMES, keep=kept) == [{"gender": "female", "state": "Kerala"}]
    assert "rejected" in kept[0] and len(model.requests) == 1


async def test_a_reply_that_cant_be_read_lets_the_task_through(model):
    model.steps += [say("I think these are two groups.")]
    assert await groups_in("t", NAMES) is None


async def test_the_groups_are_read_past_a_stray_quote_after_them(model):
    # eval: 4 of 67 replies, each Groq refused as not JSON
    model.steps += [rejected('{"name": "groups", "arguments": {"groups":[{"gender":"female","state":"Kerala"}]"}')]
    assert await groups_in("t", NAMES) == [{"gender": "female", "state": "Kerala"}]


async def test_a_call_that_fails_lets_the_task_through(model):
    def down(request):
        raise TimeoutError("no reply")
    model.steps += [down]
    assert await groups_in("t", NAMES) is None
