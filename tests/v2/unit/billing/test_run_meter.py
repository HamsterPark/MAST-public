"""The spend probes that finally connect the ledger to the orchestrator's gate.

Context (2026-07-30): ``budget_remaining_usd`` had a working hard gate in the
supervisor and NO writer anywhere in the tree, so it had never fired once. The
only real bound on one run's cost was ``recursion_limit`` — about $50 worth — and
the ledger holds a run that spent $30.66 in an hour on a loop of SUCCESSES, which
StallGuard cannot see because it keys off repeated FAILURE signatures.

What these tests hold in place:
  * ``None`` means "unreadable", never 0.0 and never infinity — that distinction
    is what keeps a billing hiccup from either killing a healthy run or silently
    disabling the ceiling;
  * mixed-currency rows are summed into ONE comparable number, because the ledger
    stores each row in the provider's own currency on purpose;
  * ``RunMeter`` measures from run start, and the over-attribution that implies is
    documented rather than papered over;
  * ``make_budget_probe(0)`` yields an inert probe rather than a $0 ceiling that
    would end every run on its first hop.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import time  # noqa: E402

import pytest  # noqa: E402

from mast.billing import run_meter as rm  # noqa: E402
from mast.billing.ledger import UsageLedger, UsageRecord, set_ledger_for_test  # noqa: E402


@pytest.fixture()
def ledger(tmp_path):
    """A REAL ledger on a temp file, installed as the process ledger.

    A real sqlite ledger rather than a stub: the probe's whole job is to read that
    schema correctly, and a stub would let a wrong column name pass. Torn down so
    no test can leak into the operator's actual usage database — this repo has
    polluted real user data four times, always through exactly this kind of
    redirection gap.
    """
    led = UsageLedger(tmp_path / "usage.sqlite")
    set_ledger_for_test(led)
    try:
        yield led
    finally:
        set_ledger_for_test(None)
        led.close()


def _book(led: UsageLedger, cost: float, currency: str = "USD",
          ts: float | None = None, *, known: bool = True) -> None:
    led.record(UsageRecord(
        kind="llm", provider="p", model="m", source="test",
        input_tokens=10, output_tokens=10, cost=cost, currency=currency,
        cost_known=known, ts=(time.time() if ts is None else ts),
    ))


class TestSpendUsd:
    def test_empty_ledger_is_zero_not_none(self, ledger):
        """An empty ledger is a KNOWN zero. Only an unreadable one is unknown, and
        conflating the two would make the gate inert whenever nothing had been
        spent yet — i.e. at the start of every run."""
        assert rm.spend_usd() == 0.0

    def test_sums_usd_rows(self, ledger):
        _book(ledger, 1.25)
        _book(ledger, 0.75)
        assert rm.spend_usd() == pytest.approx(2.0)

    def test_converts_non_usd_rows(self, ledger, monkeypatch):
        monkeypatch.setattr(rm, "_usd_to_cny", lambda: 8.0)
        _book(ledger, 8.0, currency="CNY")     # → $1
        _book(ledger, 2.0, currency="USD")
        assert rm.spend_usd() == pytest.approx(3.0)

    def test_estimated_rows_are_included(self, ledger):
        """A row whose price could only be estimated still cost money. For a
        CEILING an approximate number beats a silent zero."""
        _book(ledger, 4.0, known=False)
        assert rm.spend_usd() == pytest.approx(4.0)

    def test_since_window_excludes_older_rows(self, ledger):
        now = time.time()
        _book(ledger, 5.0, ts=now - 3600)
        _book(ledger, 2.0, ts=now)
        assert rm.spend_usd(since=now - 60) == pytest.approx(2.0)

    def test_unreadable_ledger_is_none_not_zero(self, monkeypatch):
        """The distinction the whole design rests on. 0.0 would mean 'plenty of
        budget left' and quietly disable the ceiling; None means 'do not touch the
        gate', which is what every caller is required to honour."""
        class _Boom:
            def summary(self, **_kw):
                raise RuntimeError("db locked")

        monkeypatch.setattr(rm, "spend_usd", rm.spend_usd)  # keep the real fn
        import mast.billing.ledger as led_mod
        monkeypatch.setattr(led_mod, "get_ledger", lambda: _Boom())
        assert rm.spend_usd() is None

    def test_a_bad_rate_falls_back_instead_of_raising(self, ledger, monkeypatch):
        import mast.billing.pricing as pricing
        monkeypatch.setattr(pricing, "usd_to_cny_rate", lambda: 0.0)
        _book(ledger, 7.2, currency="CNY")
        # A zero rate would be a division by zero; the fallback keeps the gate
        # roughly right rather than taking the run down over billing.
        assert rm.spend_usd() == pytest.approx(1.0, rel=0.05)


class TestDailySpend:
    def test_counts_today_and_ignores_yesterday(self, ledger):
        """The cross-run dimension. A per-run ceiling cannot see run COUNT, and the
        wake scheduler exists to create more runs — ten woken runs at $8 are $80
        and every one is individually legal."""
        midnight = rm._local_midnight_ts()
        _book(ledger, 12.0, ts=midnight - 120)     # yesterday
        _book(ledger, 3.0, ts=midnight + 120)      # today
        assert rm.daily_spend_usd() == pytest.approx(3.0)


class TestRunMeter:
    def test_remaining_is_budget_minus_spend_since_start(self, ledger):
        started = time.time()
        _book(ledger, 4.0, ts=started - 3600)   # BEFORE the run — not ours
        _book(ledger, 1.5, ts=started + 1)      # during the run
        meter = rm.RunMeter(budget_usd=8.0, started_at=started)
        assert meter.spent_usd() == pytest.approx(1.5)
        assert meter.remaining_usd() == pytest.approx(6.5)

    def test_remaining_floors_at_zero(self, ledger):
        started = time.time()
        _book(ledger, 50.0, ts=started + 1)
        meter = rm.RunMeter(budget_usd=8.0, started_at=started)
        assert meter.remaining_usd() == 0.0

    def test_a_call_in_flight_at_run_start_is_not_free(self, ledger):
        """The window opens a hair before the run so a request already in flight
        when the run began is still charged to it."""
        started = time.time()
        _book(ledger, 2.0, ts=started - 0.2)
        meter = rm.RunMeter(budget_usd=8.0, started_at=started)
        assert meter.spent_usd() == pytest.approx(2.0)

    def test_remaining_is_none_when_spend_is_unknown(self, monkeypatch):
        monkeypatch.setattr(rm, "spend_usd", lambda **_kw: None)
        meter = rm.RunMeter(budget_usd=8.0)
        assert meter.remaining_usd() is None

    def test_as_dict_reports_the_basis_of_its_number(self, ledger):
        """The over-attribution caveat must travel WITH the number wherever it
        surfaces — a bare dollar figure reads as exact."""
        d = rm.RunMeter(budget_usd=8.0).as_dict()
        assert d["budget_usd"] == 8.0
        assert "basis" in d and "since run start" in d["basis"]


class TestMakeBudgetProbe:
    def test_zero_budget_yields_an_inert_probe(self, ledger):
        """0 means "no ceiling configured". Returning 0.0 would end every run on
        its first hop, which is what a `<= 0` gate does with a zero."""
        assert rm.make_budget_probe(0)() is None
        assert rm.make_budget_probe(None)() is None
        assert rm.make_budget_probe(-5)() is None

    def test_garbage_budget_is_inert_rather_than_fatal(self, ledger):
        assert rm.make_budget_probe("not-a-number")() is None

    def test_positive_budget_probes_real_spend(self, ledger):
        started = time.time()
        probe = rm.make_budget_probe(10.0, started_at=started)
        _book(ledger, 2.5, ts=started + 1)
        assert probe() == pytest.approx(7.5)

    def test_probe_shape_matches_what_the_graph_expects(self, ledger):
        """orchestrator.build(budget_probe=…) calls this with no arguments and
        must never see an exception."""
        probe = rm.make_budget_probe(1.0)
        assert callable(probe)
        probe()
