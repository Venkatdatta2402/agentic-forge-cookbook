"""Offline tests for the AP desk: both builds, the whole inbox, and every check in `ap_checks`.

    python test_ap.py          (from this folder; costs nothing)

Run before spending tokens. Each assertion is a behaviour the notebook claims; the ones marked
"found live" are failures a real run produced first, kept here so they cannot come back unnoticed.
"""

import copy
import os
import sqlite3
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

import ap_checks as C  # noqa: E402
import ap_graph as G  # noqa: E402
import ap_scratch as A  # noqa: E402
import ap_stubs as S  # noqa: E402
import payables as P  # noqa: E402

FIELDS = ("status", "amount", "signed_by")


def as_policy_says(outcomes):
    return {k: o for k, o in outcomes.items() if any(o[f] != P.EXPECTED[k][f] for f in FIELDS)}


def sku_in_description(email):
    # the model's other fair reading of "CB-7 controller board": code left in the text, sku empty (found live)
    invoice = copy.deepcopy(S.extract(email))
    for line in invoice["lines"]:
        if line["sku"]:
            line["description"], line["sku"] = f"{line['sku']} {line['description']}", ""
    return invoice


def inbox(build, settle, **parts):
    P.reset()
    desk = build(**parts)
    return {key: P.outcome(settle(desk, key)) for key in P.INBOX}


def test_both_builds_follow_the_policy():
    for extract in (S.extract, sku_in_description):
        graph = inbox(lambda **p: G.build_desk(InMemorySaver(), **p), G.settle,
                      extract=extract, model=S.Scripted(), draft=S.draft)
        scratch = inbox(A.build_desk, A.settle, extract=extract, investigate=S.investigate, draft=S.draft)
        assert not as_policy_says(graph), as_policy_says(graph)
        assert not as_policy_says(scratch), as_policy_says(scratch)


def test_checks():
    sent = C.double_send()
    assert sent["LangGraph"] == {"emails sent while waiting": 1, "emails sent after the reply": 2}
    assert sent["case file"] == {"emails sent while waiting": 1, "emails sent after the reply": 1}

    swapped = C.reordered_answers()
    assert swapped["LangGraph recorded"] == {"tomasz": "REFUSED", "ines": "approved"}
    assert swapped["case file recorded"] == swapped["what happened"]

    mira = C.wrong_person()
    assert mira["LangGraph, after Mira answered"] == {"status": "scheduled", "amount": 10300.0, "signed_by": ["tomasz"]}
    assert "tomasz's to set, not mira's" in mira["case file, when Mira answered"]
    assert mira["case file, still waiting on"] == "approval:tomasz"

    runaway = C.runaway_agent()
    assert runaway["LangGraph default recursion_limit"] == 10007
    assert runaway["LangGraph, limit 17"]["status"] == "held" and runaway["scratch"]["status"] == "held"
    assert runaway["scratch"]["model calls"] < runaway["LangGraph, limit 17"]["model calls"]

    loop = C.endless_vendor_loop(rounds=10)
    assert loop["vendor rounds"] == 11 and loop["still waiting for the vendor"]

    assert C.flaky_provider()["attempts at reading"] == 2

    refused = C.refused_tool_call()                    # found live
    assert refused["thread parked before"] == ("investigate",)
    assert refused["invoke(None) then reached"] == ["vendor"]
    assert refused["recommendation"] == "query_vendor"

    rewound = C.misread_iban()
    assert rewound["waiting on"] == ["callback"] and rewound["after the fix"]["status"] == "scheduled"


def test_finish_elsewhere():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "ap.sqlite"
        P.reset()
        desk = G.build_desk(SqliteSaver(sqlite3.connect(db, check_same_thread=False)),
                            extract=S.extract, model=S.Scripted(), draft=S.draft)
        G.start(desk, "B")
        said = C.finish_elsewhere(db, "B")
        assert "after answering: {'status': 'scheduled'" in said, said
        assert desk.get_state(G.config("B")).values["status"] == "scheduled"
        desk.checkpointer.conn.close()


def test_the_case_file():
    import case as CF
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "cases.sqlite"
        P.reset()
        desk = A.build_desk(path=str(db), extract=S.extract, investigate=S.investigate, draft=S.draft)
        waiting = desk.open("invoice-C", {"key": "C", "email": P.INBOX["C"], "vendor_rounds": 0})
        assert waiting["field"] == "vendor_reply" and len(desk.state("invoice-C")["outbox"]) == 1

        # a second desk on the same file -- another process, as far as the case can tell -- carries it on
        later = A.build_desk(path=str(db), extract=S.extract, investigate=S.investigate, draft=S.draft)
        waiting = later.answer("invoice-C", "vendor", "vendor_reply", P.human("C", "vendor"))
        assert waiting["field"] == "approval:mira" and "INV-88251-R" in waiting["packet"]
        assert "Investigation:" not in waiting["packet"]             # the old invoice's reasons stay with it
        later.answer("invoice-C", "mira", "approval:mira", P.human("C", "approval", "mira"))
        state = later.state("invoice-C")
        assert P.outcome(state) == {"status": "scheduled", "amount": 10320.0, "signed_by": ["mira"]}
        assert len(state["outbox"]) == 1                              # the vendor was written to once

        case = later.case("invoice-C")
        for author, field in [("investigator", "amount"), ("reader", "status"), ("mira", "approval:tomasz"),
                              ("desk", "invoice"), ("desk", "anything_new")]:
            try:
                case.set(author, field, 0)
                assert False, f"{author} wrote {field}"
            except CF.Denied:
                pass
        authors = {w["author"] for w in case.history}
        assert authors == {"desk", "reader", "investigator", "vendor", "mira"}, authors
        later.log.db.close()
        desk.log.db.close()


if __name__ == "__main__":
    for test in (test_both_builds_follow_the_policy, test_checks, test_finish_elsewhere, test_the_case_file):
        test()
        print("ok ", test.__name__)
