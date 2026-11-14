"""The MoSPI client the release job uses: paging, errors that arrive as success, rate limits."""

import pytest

from stats_desk.release.mospi import Breaker, Mospi, MospiError


class Fake:
    """A transport answering from a function of (tool, args), recording what it was asked."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def call(self, tool, args):
        self.calls.append((tool, args))
        result = self.answer(tool, args)
        if isinstance(result, Exception):
            raise result
        return result


async def no_wait(_):
    pass


def mospi(answer, **kw):
    fake = Fake(answer)
    return Mospi(fake, sleep=no_wait, **kw), fake


PLFS = {"indicator_code": 3, "frequency_code": 1}


async def test_an_error_returned_as_success_is_raised_with_mospis_hint():
    body = {"error": "Invalid parameters", "valid": False, "invalid_params": ["level"],
            "valid_params": ["base_year", "series", "year"]}
    m, _ = mospi(lambda t, a: body)
    with pytest.raises(MospiError, match="valid_params: base_year, series, year"):
        await m.data("CPI", {"base_year": "2024", "level": "Group"})


async def test_plfs_without_frequency_is_refused_before_mospi_can_default_it():
    # MoSPI answers the ANNUAL table when frequency_code is left out
    m, fake = mospi(lambda t, a: {"data": []})
    with pytest.raises(MospiError, match="needs frequency_code"):
        await m.data("PLFS", {"indicator_code": 3, "year": "2026"})
    assert fake.calls == []


async def test_user_query_is_never_sent():
    m, fake = mospi(lambda t, a: {"datasets": {}})
    await m._call("list_datasets", {"user_query": "my private question"})
    assert fake.calls == [("list_datasets", {})]


async def test_pages_are_fetched_at_100_rows_one_after_another():
    def answer(tool, args):
        page = int(args["filters"].get("page", 1))
        return {"data": [{"v": page}] * (100 if page < 3 else 50),
                "meta_data": {"totalRecords": 250, "totalPages": 3}}
    m, fake = mospi(answer)
    rows = await m.data("PLFS", {**PLFS, "limit": "9"})
    assert len(rows) == 250
    assert [a["filters"].get("page") for _, a in fake.calls] == [None, "2", "3"]
    assert all(a["filters"]["limit"] == "100" for _, a in fake.calls)


async def test_too_many_rows_is_refused_before_downloading_them():
    m, fake = mospi(lambda t, a: {"data": [{"v": 1}] * 100,
                                  "meta_data": {"totalRecords": 21996, "totalPages": 220}})
    with pytest.raises(MospiError, match="21,996 rows"):
        await m.data("PLFS", PLFS, max_rows=5_000)
    assert len(fake.calls) == 1


async def test_too_many_requests_is_waited_out_not_reported():
    # measured: MoSPI's backend answered 429 after ~150 quick page requests
    replies = iter([{"error": "too many 429 error responses"},
                    {"error": "too many 429 error responses"},
                    {"data": [{"v": 1}], "meta_data": {"totalRecords": 1, "totalPages": 1}}])
    m, fake = mospi(lambda t, a: next(replies))
    assert await m.data("PLFS", PLFS) == [{"v": 1}]
    assert len(fake.calls) == 3


async def test_breaker_stops_calling_a_service_that_keeps_failing():
    m, fake = mospi(lambda t, a: ConnectionError("down"), breaker=Breaker(threshold=3, cooldown=60))
    for _ in range(3):
        with pytest.raises(MospiError, match="could not be reached"):
            await m._call("list_datasets", {})
    with pytest.raises(MospiError, match="not responding"):
        await m._call("list_datasets", {})
    assert len(fake.calls) == 3


async def test_repeat_calls_are_served_from_the_cache():
    m, fake = mospi(lambda t, a: {"datasets": {}})
    await m._call("list_datasets", {})
    await m._call("list_datasets", {})
    assert len(fake.calls) == 1
