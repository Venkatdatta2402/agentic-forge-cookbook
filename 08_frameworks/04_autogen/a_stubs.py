"""Stand-ins for every model call, so both teams run offline -- with real code really executed.

    CODE        a careful analysis per question: duplicates dropped, returns handled, figures printed
    replay      AutoGen's own `ReplayChatCompletionClient`, playing back a planner, the analyst's code
                block, the analyst's report, an approval and the finding, in that order
    scratch     the same script for the scratch team; its analyst really calls `run_python`
"""

import brightwater as B

CODE = {
    "Q1": """import pandas as pd
df = pd.read_csv("orders.csv")
print("duplicate order_ids:", df.order_id.duplicated().sum())
df = df.drop_duplicates("order_id")
df["net"] = df.units * df.unit_price * (1 - df.discount)
df = df[df.returned == 0]
df["q"] = pd.to_datetime(df.order_date).dt.quarter
by = df.groupby(["region", "q"]).net.sum().unstack()
print(((by[3] / by[2] - 1) * 100).round(1).to_string())""",
    "Q2": """import pandas as pd
df = pd.read_csv("orders.csv").drop_duplicates("order_id")
q3 = df[df.order_date >= "2026-07-01"]
print((q3.groupby("category").returned.mean() * 100).round(1).to_string())""",
    "Q3": """import pandas as pd
df = pd.read_csv("orders.csv").drop_duplicates("order_id")
t = df[df.category == "tents"]
during = t[(t.order_date >= "2026-07-01") & (t.order_date <= "2026-07-14")].units.sum() / 14
before = t[(t.order_date >= "2026-06-17") & (t.order_date <= "2026-06-30")].units.sum() / 14
print("per day before:", round(before, 2), "during:", round(during, 2), "lift %:", round((during / before - 1) * 100, 1))""",
}


def finding(qid):
    want = B.truth()[qid]
    return B.Finding(entity=str(want["entity"]), value=want["value"], answer=f"{want['entity']}: {want['value']}%.",
                     data_issues=["70 duplicate order rows dropped"])


def replay(qid, approve=True, skills=False):
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, ModelInfo, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient
    review = (B.Review(approved=True, fix_by="nobody") if approve else
              B.Review(approved=False, problems=["check the data again"], fix_by="analyst")).model_dump_json()
    replies = ["1. Check the data for duplicates. 2. Compute what the question defines.",
               f"```python\n{CODE[qid]}\n```", "The figures are printed above.", review, finding(qid).model_dump_json()]
    if skills:
        # the planner reads a procedure first, then writes its plan from what came back
        read = CreateResult(finish_reason="function_calls", usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                            cached=False, content=[FunctionCall(id="1", name="read_skill",
                                                                arguments='{"name": "deduplicate-records"}')])
        replies = [read] + replies
    if not approve:
        replies = [replies[0]] + [replies[1], replies[2], review] * 10     # routing is the review form's now
    return ReplayChatCompletionClient(replies, model_info=ModelInfo(vision=False, function_calling=True,
                                                                    json_output=True, structured_output=True,
                                                                    family="unknown"))


def scratch(team, qid, approve=True):
    """An `answer` for `team_scratch.Team`, whose analyst runs CODE through the team's real `run_python`."""
    def answer(name, request, model):
        if model is B.Finding:
            return finding(qid)
        if model is B.Review:
            return (B.Review(approved=True, fix_by="nobody") if approve else
                    B.Review(approved=False, problems=["check the data again"], fix_by="analyst"))
        if name == "analyst":
            team.run_python(CODE[qid])
            return "The figures are in what the program printed."
        if team.skills:
            team.read_skill("deduplicate-records")
        return "1. Check the data for duplicates. 2. Compute what the question defines."
    return answer
