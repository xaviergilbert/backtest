"""End-to-end integration test for fixed_nav with leverage 4 (2x long, 2x short).

Simulates a full SimpleBacktester run with 3 weekly rebalancings on Monday close.
Weekly return period is Tuesday through the next Monday (included).

Verifies that:
- positions are sized from initial_cash (not drifting equity)
- equity resets to initial_cash after each rebalance
- daily returns reflect true market P&L net of fees
- old positions are auto-closed when the portfolio changes
"""
import datetime
import math
import unittest

import pandas

from bktest.backtest import SimpleBacktester
from bktest.data.source import DataFrameDataSource
from bktest.export.quants import QuantStatsExporter
from bktest.fee import ExpressionFeeModel
from bktest.order import DataFrameOrderProvider

INITIAL_CASH = 100_000
FEE_RATE = 0.001  # 10 bps

# 4 weeks of weekdays: Mon Jan 8 through Fri Feb 2
# Rebal on MON1, MON2, MON3. Prices through MON4 to capture week 3 returns.
MON1 = datetime.date(2024, 1, 8)
TUE1 = datetime.date(2024, 1, 9)
WED1 = datetime.date(2024, 1, 10)
THU1 = datetime.date(2024, 1, 11)
FRI1 = datetime.date(2024, 1, 12)
MON2 = datetime.date(2024, 1, 15)
TUE2 = datetime.date(2024, 1, 16)
WED2 = datetime.date(2024, 1, 17)
THU2 = datetime.date(2024, 1, 18)
FRI2 = datetime.date(2024, 1, 19)
MON3 = datetime.date(2024, 1, 22)
TUE3 = datetime.date(2024, 1, 23)
WED3 = datetime.date(2024, 1, 24)
THU3 = datetime.date(2024, 1, 25)
FRI3 = datetime.date(2024, 1, 26)
MON4 = datetime.date(2024, 1, 29)

# Daily prices for 6 stocks
PRICES = {
    MON1: {"AAPL": 50, "MSFT": 40, "GOOG": 100, "META": 80, "NVDA": 25, "AMZN": 60},
    TUE1: {"AAPL": 52, "MSFT": 38, "GOOG": 102, "META": 79, "NVDA": 26, "AMZN": 59},
    WED1: {"AAPL": 51, "MSFT": 39, "GOOG": 105, "META": 78, "NVDA": 27, "AMZN": 58},
    THU1: {"AAPL": 53, "MSFT": 37, "GOOG": 103, "META": 81, "NVDA": 28, "AMZN": 57},
    FRI1: {"AAPL": 54, "MSFT": 36, "GOOG": 108, "META": 82, "NVDA": 29, "AMZN": 56},
    MON2: {"AAPL": 55, "MSFT": 35, "GOOG": 110, "META": 83, "NVDA": 30, "AMZN": 55},
    TUE2: {"AAPL": 56, "MSFT": 34, "GOOG": 112, "META": 84, "NVDA": 31, "AMZN": 54},
    WED2: {"AAPL": 54, "MSFT": 36, "GOOG": 109, "META": 82, "NVDA": 29, "AMZN": 56},
    THU2: {"AAPL": 57, "MSFT": 33, "GOOG": 111, "META": 85, "NVDA": 32, "AMZN": 53},
    FRI2: {"AAPL": 58, "MSFT": 32, "GOOG": 113, "META": 86, "NVDA": 33, "AMZN": 52},
    MON3: {"AAPL": 59, "MSFT": 31, "GOOG": 115, "META": 87, "NVDA": 34, "AMZN": 51},
    TUE3: {"AAPL": 60, "MSFT": 30, "GOOG": 114, "META": 88, "NVDA": 35, "AMZN": 50},
    WED3: {"AAPL": 58, "MSFT": 32, "GOOG": 116, "META": 86, "NVDA": 33, "AMZN": 52},
    THU3: {"AAPL": 61, "MSFT": 29, "GOOG": 118, "META": 89, "NVDA": 36, "AMZN": 49},
    FRI3: {"AAPL": 62, "MSFT": 28, "GOOG": 120, "META": 90, "NVDA": 37, "AMZN": 48},
    MON4: {"AAPL": 63, "MSFT": 27, "GOOG": 122, "META": 91, "NVDA": 38, "AMZN": 47},
}

# 3 rebalancings on Monday close, each with a different portfolio
# Week 1 return: TUE1 → MON2 (rebal 2 happens at MON2 close)
# Week 2 return: TUE2 → MON3 (rebal 3 happens at MON3 close)
# Week 3 return: TUE3 → MON4
ORDERS = [
    # Rebal 1 (MON1): AAPL +2.0, MSFT -2.0
    {"date": MON1, "symbol": "AAPL", "quantity": 2.0},
    {"date": MON1, "symbol": "MSFT", "quantity": -2.0},
    # Rebal 2 (MON2): GOOG +0.8, NVDA +0.7, META +0.5, AMZN -1.2, AAPL -0.8
    {"date": MON2, "symbol": "GOOG", "quantity": 0.8},
    {"date": MON2, "symbol": "NVDA", "quantity": 0.7},
    {"date": MON2, "symbol": "META", "quantity": 0.5},
    {"date": MON2, "symbol": "AMZN", "quantity": -1.2},
    {"date": MON2, "symbol": "AAPL", "quantity": -0.8},
    # Rebal 3 (MON3): MSFT +1.0, NVDA +0.6, META +0.4, GOOG -1.5, AMZN -0.5
    {"date": MON3, "symbol": "MSFT", "quantity": 1.0},
    {"date": MON3, "symbol": "NVDA", "quantity": 0.6},
    {"date": MON3, "symbol": "META", "quantity": 0.4},
    {"date": MON3, "symbol": "GOOG", "quantity": -1.5},
    {"date": MON3, "symbol": "AMZN", "quantity": -0.5},
]


def _target(percent, price):
    return int(INITIAL_CASH * percent / price)


def _fee(delta, price):
    return abs(price * delta) * FEE_RATE


def _value(holdings, prices):
    """Compute portfolio value: sum(shares * price) for each symbol."""
    return sum(shares * prices[symbol] for symbol, shares in holdings.items())


class FixedNavIntegrationTest(unittest.TestCase):

    def setUp(self):
        prices_df = pandas.DataFrame([
            {"date": d, "symbol": s, "price": float(p)}
            for d, symbols in PRICES.items()
            for s, p in symbols.items()
        ])

        self.exporter = QuantStatsExporter(
            html_output_file=None,
            csv_output_file=None,
            benchmark_ticker=None,
        )

        self.bt = SimpleBacktester(
            start=MON1,
            end=MON4,
            order_provider=DataFrameOrderProvider(pandas.DataFrame(ORDERS)),
            initial_cash=INITIAL_CASH,
            quantity_in_decimal=True,
            data_source=DataFrameDataSource(prices_df),
            auto_close_others=True,
            exporters=[self.exporter],
            fee_model=ExpressionFeeModel(f"abs(price * quantity) * {FEE_RATE}"),
            caching=False,
            fixed_nav=True,
            allow_holidays=True,
        )

        self.bt.run()

    def test_full_scenario(self):
        df = self.exporter.dataframe
        returns = df["daily_profit_pct"]

        # ==================================================================
        # Rebal 1 (MON1 close): AAPL +2.0, MSFT -2.0
        # ==================================================================
        r1 = {
            "AAPL": _target(2.0, 50),    # int(200000/50) = 4000
            "MSFT": _target(-2.0, 40),    # int(-200000/40) = -5000
        }
        self.assertEqual(4000, r1["AAPL"])
        self.assertEqual(-5000, r1["MSFT"])

        val_r1 = _value(r1, PRICES[MON1])
        cash_r1 = INITIAL_CASH - val_r1

        # MON1: first day, NaN return
        self.assertTrue(math.isnan(returns.loc[MON1]))
        self.assertAlmostEqual(INITIAL_CASH, df.loc[MON1, "post_reset_equity"])

        # Week 1 daily returns: TUE1 through MON2 (included)
        prev_post = INITIAL_CASH
        for date in (TUE1, WED1, THU1, FRI1):
            equity = cash_r1 + _value(r1, PRICES[date])
            expected = equity / prev_post - 1
            self.assertAlmostEqual(expected, returns.loc[date], places=5,
                                   msg=f"return mismatch on {date}")
            prev_post = equity  # non-rebalance: pre == post

        # MON2 closes week 1: price update gives pre_reset, then rebal fires
        mon2_equity = cash_r1 + _value(r1, PRICES[MON2])

        # ==================================================================
        # Rebal 2 (MON2 close): 5 positions
        # ==================================================================
        r2 = {
            "GOOG": _target(0.8, 110),    # int(80000/110) = 727
            "NVDA": _target(0.7, 30),      # int(70000/30) = 2333
            "META": _target(0.5, 83),      # int(50000/83) = 602
            "AMZN": _target(-1.2, 55),     # int(-120000/55) = -2181
            "AAPL": _target(-0.8, 55),     # int(-80000/55) = -1454
        }

        # Fees: deltas from r1 to r2
        fees_r2 = (
            _fee(r2["AAPL"] - r1["AAPL"], 55) +   # AAPL: 4000 → -1454
            _fee(0 - r1["MSFT"], 35) +              # MSFT: -5000 → 0 (closed)
            _fee(r2["GOOG"], 110) +                  # GOOG: 0 → 727
            _fee(r2["NVDA"], 30) +                   # NVDA: 0 → 2333
            _fee(r2["META"], 83) +                   # META: 0 → 602
            _fee(r2["AMZN"], 55)                     # AMZN: 0 → -2181
        )

        # MON2 return includes week 1 P&L net of rebal fees
        expected_mon2 = (mon2_equity - fees_r2) / prev_post - 1
        self.assertAlmostEqual(expected_mon2, returns.loc[MON2], places=5)
        self.assertAlmostEqual(INITIAL_CASH, df.loc[MON2, "post_reset_equity"])

        val_r2 = _value(r2, PRICES[MON2])
        cash_r2 = INITIAL_CASH - val_r2

        # Week 2 daily returns: TUE2 through MON3 (included)
        prev_post = INITIAL_CASH
        for date in (TUE2, WED2, THU2, FRI2):
            equity = cash_r2 + _value(r2, PRICES[date])
            expected = equity / prev_post - 1
            self.assertAlmostEqual(expected, returns.loc[date], places=5,
                                   msg=f"return mismatch on {date}")
            prev_post = equity

        # MON3 closes week 2: price update gives pre_reset, then rebal fires
        mon3_equity = cash_r2 + _value(r2, PRICES[MON3])

        r3 = {
            "MSFT": _target(1.0, 31),      # int(100000/31) = 3225
            "NVDA": _target(0.6, 34),       # int(60000/34) = 1764
            "META": _target(0.4, 87),       # int(40000/87) = 459
            "GOOG": _target(-1.5, 115),     # int(-150000/115) = -1304
            "AMZN": _target(-0.5, 51),      # int(-50000/51) = -980
        }

        # Fees: deltas from r2 to r3
        fees_r3 = (
            _fee(r3["GOOG"] - r2["GOOG"], 115) +    # GOOG: 727 → -1304
            _fee(r3["NVDA"] - r2["NVDA"], 34) +      # NVDA: 2333 → 1764
            _fee(r3["META"] - r2["META"], 87) +       # META: 602 → 459
            _fee(r3["AMZN"] - r2["AMZN"], 51) +      # AMZN: -2181 → -980
            _fee(0 - r2["AAPL"], 59) +                # AAPL: -1454 → 0 (closed)
            _fee(r3["MSFT"], 31)                       # MSFT: 0 → 3225
        )

        # MON3 return includes week 2 P&L net of rebal fees
        expected_mon3 = (mon3_equity - fees_r3) / prev_post - 1
        self.assertAlmostEqual(expected_mon3, returns.loc[MON3], places=5)
        self.assertAlmostEqual(INITIAL_CASH, df.loc[MON3, "post_reset_equity"])

        val_r3 = _value(r3, PRICES[MON3])
        cash_r3 = INITIAL_CASH - val_r3

        # Week 3 daily returns: TUE3 through MON4
        prev_post = INITIAL_CASH
        for date in (TUE3, WED3, THU3, FRI3, MON4):
            equity = cash_r3 + _value(r3, PRICES[date])
            expected = equity / prev_post - 1
            self.assertAlmostEqual(expected, returns.loc[date], places=5,
                                   msg=f"return mismatch on {date}")
            prev_post = equity

        # ==================================================================
        # Final state: only r3 holdings remain
        # ==================================================================
        holdings = {h.symbol: h.quantity for h in self.bt.account.holdings}
        self.assertEqual(r3, holdings)


if __name__ == "__main__":
    unittest.main()
