import asyncio

import pytest

from stats_desk.core.governor import DailyLimitReached, Governor, TooLarge

# small_limits: a 1,000-token bucket refilling at 1000/60 = 16.67 tokens a second


def gov(clock, limits):
    return Governor(limits={"m": limits}, clock=clock, wall=clock, sleep=clock.sleep)


async def test_admits_while_the_bucket_has_room(clock, small_limits):
    g = gov(clock, small_limits)
    await g.reserve("m", 400)
    await g.reserve("m", 500)
    assert clock.slept == []


async def test_waits_only_as_long_as_the_refill_needs(clock, small_limits):
    # Groq refills continuously: a 94-token call reset in 705 ms. Waiting a full minute, as a
    # sliding window does, cost one live run 415 of its 425 seconds.
    g = gov(clock, small_limits)
    await g.reserve("m", 700)            # 300 left
    await g.reserve("m", 700)            # needs 400 more: 400 / 16.67 = 24 s
    assert sum(clock.slept) == pytest.approx(24, abs=0.5)


async def test_a_wait_is_told_once_with_its_length_for_the_page(clock, small_limits):
    g, told = gov(clock, small_limits), []
    await g.reserve("m", 700, on_wait=told.append)          # room: nothing to tell
    await g.reserve("m", 700, on_wait=told.append)          # 24 s, however many sleeps it takes
    assert told == [24]


def test_one_governor_serves_one_event_loop_after_another(small_limits):
    """The eval runs each question in its own asyncio.run with the same governor (the
    second question's analysts, waiting in line, got "bound to a different event loop")."""
    class Clock:                         # time passes when slept, and a sleep lets the other in line run
        t = 1000.0

        def __call__(self):
            return self.t

        async def sleep(self, seconds):
            self.t += seconds
            await asyncio.sleep(0)
    c = Clock()
    g = Governor(limits={"m": small_limits}, clock=c, wall=c, sleep=c.sleep)

    async def two_in_line():
        await asyncio.gather(g.reserve("m", 600), g.reserve("m", 600))      # the second waits its turn
    asyncio.run(two_in_line())
    asyncio.run(two_in_line())                                               # a new loop, the same governor


async def test_refuses_a_request_over_the_per_request_cap_before_sending(clock, small_limits):
    with pytest.raises(TooLarge):
        await gov(clock, small_limits).reserve("m", 801)


async def test_daily_limit_is_refused_not_waited_on(clock, small_limits):
    g = gov(clock, small_limits)
    for _ in range(4):                   # 4 x 700 = 2,800 of 3,000
        await g.reserve("m", 700)
    with pytest.raises(DailyLimitReached) as e:
        await g.reserve("m", 700)
    assert e.value.retry_after > 3600


async def test_settle_refunds_what_the_call_did_not_use(clock, small_limits):
    g = gov(clock, small_limits)
    ticket = await g.reserve("m", 800)   # prompt + max_tokens; 200 left
    g.settle(ticket, used=300)           # the model wrote less than its cap: 500 back
    await g.reserve("m", 700)
    assert clock.slept == []


async def test_the_request_cap_counts_max_tokens_but_the_bucket_takes_a_typical_reply(clock, small_limits):
    g = gov(clock, small_limits)
    with pytest.raises(TooLarge):
        await g.reserve("m", 300, request=900)       # prompt + max_tokens over the 800 cap
    await g.reserve("m", 300, request=700)
    assert g.snapshot()["models"]["m"]["minute_available"] == 700


async def test_a_call_that_used_more_than_reserved_leaves_a_debt(clock, small_limits):
    g = gov(clock, small_limits)
    ticket = await g.reserve("m", 300)
    g.settle(ticket, used=900)                       # 600 over: level 1000 - 900 = 100
    await g.reserve("m", 400)                        # needs 300 more: 18 s
    assert sum(clock.slept) == pytest.approx(18, abs=0.5)


async def test_groq_header_lower_than_ours_is_believed(clock, small_limits):
    g = gov(clock, small_limits)
    ticket = await g.reserve("m", 100)
    g.settle(ticket, used=100, remaining=200)    # someone else on the account spent 700
    await g.reserve("m", 500)                    # needs 300 more: 18 s
    assert sum(clock.slept) == pytest.approx(18, abs=0.5)


async def test_concurrent_requests_are_admitted_in_turn(clock, small_limits):
    g = gov(clock, small_limits)
    await asyncio.gather(*(g.reserve("m", 600) for _ in range(3)))
    # 400 left after the first; the second waits 12 s; the third needs a full 600: 36 s more
    assert sum(clock.slept) == pytest.approx(48, abs=1)


# ---- one bucket per user's key (the API)

async def test_each_users_key_draws_on_its_own_bucket(clock, small_limits):
    g = gov(clock, small_limits)
    await g.reserve("m", 700, account="alice")
    await g.reserve("m", 700, account="bob")      # alice's 700 is not bob's: no wait
    assert clock.slept == []


async def test_a_users_key_counts_no_day_and_takes_its_minute_from_groqs_header(clock, small_limits):
    g = gov(clock, small_limits)
    for _ in range(5):               # 3,500 tokens: past the 3,000 a day the server's key would allow
        ticket = await g.reserve("m", 700, account="paid")
        g.settle(ticket, used=700, limit=2000)    # Groq says this key's minute holds 2,000
    assert g._models[("paid", "m")].limits.tpm == 2000
    assert g.snapshot()["user_buckets"] == 1 and "m" not in g.snapshot()["models"]


def test_an_account_is_a_hash_never_the_key():
    from stats_desk.core.governor import account_of
    assert account_of("gsk_secret") != "gsk_secret" and "secret" not in account_of("gsk_secret")
    assert account_of("gsk_secret") == account_of("gsk_secret")
