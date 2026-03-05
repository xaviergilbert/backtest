import datetime
import unittest

import pandas

import bktest
from bktest.account import Account
from bktest.backtest import _Pod
from bktest.data.source import DataFrameDataSource
from bktest.export import ExporterCollection
from bktest.export.quants import QuantStatsExporter
from bktest.export.model import Snapshot
from bktest.fee import ConstantFeeModel
from bktest.price_provider import PriceProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(prices_by_date):
    """Build a prices DataFrame from {date: {symbol: price}} dict."""
    rows = []
    for date, symbols in prices_by_date.items():
        for symbol, price in symbols.items():
            rows.append({"date": date, "symbol": symbol, "price": price})
    return pandas.DataFrame(rows)


def _make_pod(initial_cash, fixed_nav, quantity_in_decimal, prices_df, start, end,
              fee_model=None):
    """Create a _Pod backed by a DataFrameDataSource.

    Pass fee_model to test fee-related behaviour; defaults to zero fees.
    """
    data_source = DataFrameDataSource(prices_df)
    price_provider = PriceProvider(start, end, data_source, mapper=None, caching=False)
    account = Account(initial_cash=initial_cash, fee_model=fee_model) if fee_model \
        else Account(initial_cash=initial_cash)
    return _Pod(
        quantity_in_decimal=quantity_in_decimal,
        auto_close_others=True,
        price_provider=price_provider,
        account=account,
        exporters=ExporterCollection([]),
        fixed_nav=fixed_nav,
    )


def _make_exporter(fixed_nav=True):
    """Create a QuantStatsExporter with no file outputs and no benchmark."""
    return QuantStatsExporter(
        html_output_file=None,
        csv_output_file=None,
        benchmark_ticker=None,
        fixed_nav=fixed_nav,
    )


def _feed_snapshots(exporter, *specs):
    """Feed (date, equity, ordered[, total_fees]) tuples into the exporter."""
    for spec in specs:
        date, equity, ordered = spec[0], spec[1], spec[2]
        total_fees = spec[3] if len(spec) > 3 else 0.0
        exporter.on_snapshot(Snapshot(
            date=date, postponned=None, cash=0.0, equity=equity,
            holdings=[], ordered=ordered, total_fees=total_fees,
        ))


D0 = datetime.date(2024, 1, 1)
D1 = datetime.date(2024, 1, 2)
D2 = datetime.date(2024, 1, 3)


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------

class FixedNavTest(unittest.TestCase):

    def test_sizing_uses_initial_cash_not_current_equity(self):
        """fixed_nav=True pins position sizing to initial_cash so that profits
        do not grow future positions (no compounding).  After each rebalance the
        cash reset forces equity back to initial_cash."""
        prices_df = _make_price_df({D0: {"AAPL": 10.0}, D1: {"AAPL": 20.0}})
        pod = _make_pod(100_000, fixed_nav=True, quantity_in_decimal=True,
                        prices_df=prices_df, start=D0, end=D1)

        # First rebalance at $10: 50% of 100_000 → 5 000 shares
        pod.order(D0, [bktest.Order("AAPL", 0.5)])
        self.assertEqual(5_000, pod.account.find_holding("AAPL").quantity)
        self.assertAlmostEqual(100_000, pod.account.equity)

        # Price doubles → equity rises to 150 000 before the second rebalance
        pod.account.find_holding("AAPL").price = 20.0
        self.assertGreater(pod.account.equity, 100_000)

        # Second rebalance at $20: still 50% of 100_000 → 2 500 shares (not 3 750)
        # The excess cash is extracted; equity is reset to initial_cash.
        pod.order(D1, [bktest.Order("AAPL", 0.5)])
        self.assertEqual(2_500, pod.account.find_holding("AAPL").quantity)
        self.assertAlmostEqual(100_000, pod.account.equity)

    def test_compounding_uses_current_equity_when_fixed_nav_false(self):
        """Without fixed_nav the default behaviour compounds: a price gain grows
        the equity base and therefore grows positions at the next rebalance."""
        prices_df = _make_price_df({D0: {"AAPL": 10.0}, D1: {"AAPL": 20.0}})
        pod = _make_pod(100_000, fixed_nav=False, quantity_in_decimal=True,
                        prices_df=prices_df, start=D0, end=D1)

        pod.order(D0, [bktest.Order("AAPL", 0.5)])
        self.assertEqual(5_000, pod.account.find_holding("AAPL").quantity)

        pod.account.find_holding("AAPL").price = 20.0  # equity → 150 000

        # Second rebalance: 50% of 150 000 at $20 → 3 750 shares (compounded)
        pod.order(D1, [bktest.Order("AAPL", 0.5)])
        self.assertEqual(3_750, pod.account.find_holding("AAPL").quantity)

    def test_no_effect_in_share_mode(self):
        """fixed_nav has no effect when quantity_in_decimal=False because the
        caller provides absolute share counts, not percentages of equity."""
        prices_df = _make_price_df({D0: {"AAPL": 10.0}})
        pod_fixed   = _make_pod(100_000, fixed_nav=True,  quantity_in_decimal=False,
                                prices_df=prices_df, start=D0, end=D0)
        pod_default = _make_pod(100_000, fixed_nav=False, quantity_in_decimal=False,
                                prices_df=prices_df, start=D0, end=D0)

        pod_fixed.order(D0,   [bktest.Order("AAPL", 100, 10.0)])
        pod_default.order(D0, [bktest.Order("AAPL", 100, 10.0)])

        self.assertEqual(100, pod_fixed.account.find_holding("AAPL").quantity)
        self.assertEqual(
            pod_fixed.account.find_holding("AAPL").quantity,
            pod_default.account.find_holding("AAPL").quantity,
        )


# ---------------------------------------------------------------------------
# Long-short portfolio (primary use case)
# ---------------------------------------------------------------------------

class LongShortFixedNavTest(unittest.TestCase):
    """Validates fixed_nav with dollar-neutral long-short portfolios."""

    def test_dollar_neutral_equity_resets_after_price_move(self):
        """For a 100% long / 100% short portfolio, a gain on the long side
        raises equity above initial_cash.  The cash reset brings it back to
        initial_cash after the next rebalance, extracting the profit."""
        prices_df = _make_price_df({
            D0: {"AAPL": 10.0, "MSFT": 10.0},
            D1: {"AAPL": 12.0, "MSFT": 10.0},  # AAPL up 20 %, MSFT flat
        })
        pod = _make_pod(100_000, fixed_nav=True, quantity_in_decimal=True,
                        prices_df=prices_df, start=D0, end=D1)

        # Rebalance 1 at $10: 10 000 long AAPL, 10 000 short MSFT
        pod.order(D0, [bktest.Order("AAPL", 1.0), bktest.Order("MSFT", -1.0)])
        self.assertEqual( 10_000, pod.account.find_holding("AAPL").quantity)
        self.assertEqual(-10_000, pod.account.find_holding("MSFT").quantity)
        self.assertAlmostEqual(100_000, pod.account.equity)

        # AAPL rises: net value = 20 000 → equity > initial_cash
        pod.account.find_holding("AAPL").price = 12.0
        self.assertGreater(pod.account.equity, 100_000)

        # Rebalance 2: positions shrink to fit initial_cash; equity resets
        pod.order(D1, [bktest.Order("AAPL", 1.0), bktest.Order("MSFT", -1.0)])
        self.assertEqual( 8_333, pod.account.find_holding("AAPL").quantity)
        self.assertEqual(-10_000, pod.account.find_holding("MSFT").quantity)
        self.assertAlmostEqual(100_000, pod.account.equity)

    def test_leverage_4_sizing_and_equity_reset(self):
        """With leverage 4 (200% long / 200% short), sizing always references
        initial_cash even when both sides move in our favour (equity > initial_cash).
        Profits are extracted via the cash reset."""
        prices_df = _make_price_df({
            D0: {"AAPL": 10.0, "MSFT": 10.0},
            D1: {"AAPL": 11.0, "MSFT":  9.0},  # long gains, short gains
        })
        pod = _make_pod(100_000, fixed_nav=True, quantity_in_decimal=True,
                        prices_df=prices_df, start=D0, end=D1)

        # Rebalance 1: 200% of 100_000 → 20 000 shares each side
        pod.order(D0, [bktest.Order("AAPL", 2.0), bktest.Order("MSFT", -2.0)])
        self.assertEqual( 20_000, pod.account.find_holding("AAPL").quantity)
        self.assertEqual(-20_000, pod.account.find_holding("MSFT").quantity)
        self.assertAlmostEqual(100_000, pod.account.equity)

        pod.account.find_holding("AAPL").price = 11.0
        pod.account.find_holding("MSFT").price =  9.0
        self.assertGreater(pod.account.equity, 100_000)

        # Rebalance 2: sizes based on initial_cash at new prices, not inflated equity
        # AAPL: int(200_000 / 11) = 18 181   MSFT: int(-200_000 / 9) = -22 222
        pod.order(D1, [bktest.Order("AAPL", 2.0), bktest.Order("MSFT", -2.0)])
        self.assertEqual( 18_181, pod.account.find_holding("AAPL").quantity)
        self.assertEqual(-22_222, pod.account.find_holding("MSFT").quantity)
        self.assertAlmostEqual(100_000, pod.account.equity)


# ---------------------------------------------------------------------------
# Fee handling
# ---------------------------------------------------------------------------

class FeePreservationTest(unittest.TestCase):

    def test_equity_resets_to_initial_cash_regardless_of_fees(self):
        """The cash reset always restores equity to initial_cash even when fees
        are charged — fees are attributed to the period return, not permanently
        deducted from the NAV baseline."""
        prices_df = _make_price_df({D0: {"AAPL": 10.0}, D1: {"AAPL": 20.0}})
        pod = _make_pod(100_000, fixed_nav=True, quantity_in_decimal=True,
                        prices_df=prices_df, start=D0, end=D1,
                        fee_model=ConstantFeeModel(500.0))

        pod.order(D0, [bktest.Order("AAPL", 0.5)])
        self.assertAlmostEqual(100_000, pod.account.equity)

        pod.account.find_holding("AAPL").price = 20.0
        pod.order(D1, [bktest.Order("AAPL", 0.5)])
        self.assertAlmostEqual(100_000, pod.account.equity)

    def test_fees_reduce_period_return(self):
        """Fees paid during a rebalance are subtracted from pre_reset_equity so
        that period return = (market_gain − fees) / initial_cash, not market_gain
        / initial_cash."""
        exporter = _make_exporter()
        _feed_snapshots(exporter, (D0, 100_000.0, False))
        # Market grew to 110 000; fees = 2 000; equity reset to 100 000
        _feed_snapshots(exporter, (D1, 110_000.0, False), (D1, 100_000.0, True, 2_000.0))
        exporter.finalize()

        # (110 000 − 2 000) / 100 000 − 1 = 8 %  (not 10 %)
        self.assertAlmostEqual(0.08, exporter.returns.loc[pandas.Timestamp(D1)], places=5)


# ---------------------------------------------------------------------------
# QuantStatsExporter row format and snapshot protocol
# ---------------------------------------------------------------------------

class QuantStatsExporterTest(unittest.TestCase):

    def test_rebalance_row_stores_pre_and_post_reset(self):
        """Each row is a 3-tuple (date, pre_reset_equity, post_reset_equity).
        A non-rebalance row has pre == post.  On a rebalance day the non-ordered
        snapshot writes pre == post first; the ordered snapshot updates post_reset
        to initial_cash in place."""
        exporter = _make_exporter()
        _feed_snapshots(exporter, (D0, 105_000.0, False))
        self.assertEqual((D0, 105_000.0, 105_000.0), exporter.rows[-1])

        _feed_snapshots(exporter, (D1, 110_000.0, False))
        self.assertEqual((D1, 110_000.0, 110_000.0), exporter.rows[-1])

        _feed_snapshots(exporter, (D1, 100_000.0, True))
        self.assertEqual((D1, 110_000.0, 100_000.0), exporter.rows[-1])

    def test_ordered_snapshot_ignored_without_fixed_nav(self):
        """With fixed_nav=False ordered snapshots are silently ignored, preserving
        the original pre-order equity in the row (backward-compatible behaviour)."""
        exporter = _make_exporter(fixed_nav=False)
        _feed_snapshots(exporter, (D0, 150_000.0, False))
        _feed_snapshots(exporter, (D0, 100_000.0, True))
        self.assertEqual([(D0, 150_000.0, 150_000.0)], exporter.rows)

    def test_skip_day_snapshot_updates_correct_row(self):
        """When a rebalance is deferred to the next trading day (skip), snapshots
        are keyed by skip.date via the postponned field.  The non-ordered snapshot
        creates the row; the ordered snapshot finds it by date and updates post_reset."""
        skip_date = datetime.date(2024, 1, 1)
        prev_date = datetime.date(2023, 12, 29)
        exporter = _make_exporter()

        exporter.on_snapshot(Snapshot(
            date=prev_date, postponned=None, cash=0.0, equity=105_000.0,
            holdings=[], ordered=False,
        ))
        exporter.on_snapshot(Snapshot(
            date=skip_date, postponned=skip_date, cash=0.0, equity=112_000.0,
            holdings=[], ordered=False,
        ))
        self.assertEqual((skip_date, 112_000.0, 112_000.0), exporter.rows[-1])

        exporter.on_snapshot(Snapshot(
            date=skip_date, postponned=skip_date, cash=0.0, equity=100_000.0,
            holdings=[], ordered=True,
        ))
        self.assertEqual((skip_date, 112_000.0, 100_000.0), exporter.rows[-1])


# ---------------------------------------------------------------------------
# Return series
# ---------------------------------------------------------------------------

class ReturnCalculationTest(unittest.TestCase):

    def test_return_uses_pre_reset_equity(self):
        """On a rebalance day the return must be (pre_reset − prev_post_reset) /
        prev_post_reset.  Using post_reset would give 0 % every rebalance day
        since post_reset == initial_cash == prev post_reset."""
        exporter = _make_exporter()
        _feed_snapshots(exporter, (D0, 100_000.0, False))
        _feed_snapshots(exporter, (D1, 110_000.0, False), (D1, 100_000.0, True))
        _feed_snapshots(exporter, (D2, 101_000.0, False))
        exporter.finalize()

        ts = lambda d: pandas.Timestamp(d)
        self.assertAlmostEqual(0.10, exporter.returns.loc[ts(D1)], places=5)  # 10 %
        self.assertAlmostEqual(0.01, exporter.returns.loc[ts(D2)], places=5)  #  1 %

    def test_consecutive_rebalances_chain_post_reset_denominators(self):
        """When two consecutive days are both rebalance days, the second day's
        denominator must be the first day's post_reset (initial_cash), not its
        pre_reset.  shift(1) on post_reset_equity handles this automatically."""
        exporter = _make_exporter()
        _feed_snapshots(exporter, (D0, 100_000.0, False))
        _feed_snapshots(exporter, (D1, 105_000.0, False), (D1, 100_000.0, True))
        _feed_snapshots(exporter, (D2, 107_000.0, False), (D2, 100_000.0, True))
        exporter.finalize()

        ts = lambda d: pandas.Timestamp(d)
        self.assertAlmostEqual(0.05, exporter.returns.loc[ts(D1)], places=5)  # 5 %
        # Denominator is D1's post_reset (100k), NOT its pre_reset (105k)
        self.assertAlmostEqual(0.07, exporter.returns.loc[ts(D2)], places=5)  # 7 %

    def test_dataframe_exposes_pre_and_post_reset_columns(self):
        """After finalize(), dataframe['equity'] is post_reset (the NAV baseline),
        dataframe['pre_reset_equity'] is the true market value before extraction,
        and dataframe['daily_profit_pct'] is the net-of-extraction return."""
        exporter = _make_exporter()
        _feed_snapshots(exporter, (D0, 100_000.0, False))
        _feed_snapshots(exporter, (D1, 110_000.0, False), (D1, 100_000.0, True))
        exporter.finalize()

        self.assertAlmostEqual(100_000.0, exporter.dataframe.loc[D1, 'equity'])
        self.assertAlmostEqual(110_000.0, exporter.dataframe.loc[D1, 'pre_reset_equity'])
        self.assertAlmostEqual(0.10,      exporter.dataframe.loc[D1, 'daily_profit_pct'], places=5)


# ---------------------------------------------------------------------------
# configure() propagation
# ---------------------------------------------------------------------------

class ConfigurePropagationTest(unittest.TestCase):

    def test_configure_overrides_fixed_nav_on_children(self):
        """ExporterCollection.configure() is authoritative: it overrides whatever
        fixed_nav each child exporter was constructed with, in both directions."""
        exporter = QuantStatsExporter(html_output_file=None, csv_output_file=None,
                                      fixed_nav=False)
        collection = ExporterCollection([exporter])

        collection.configure(fixed_nav=True)
        self.assertTrue(exporter.fixed_nav)

        collection.configure(fixed_nav=False)
        self.assertFalse(exporter.fixed_nav)
