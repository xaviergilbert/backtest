import abc
import os
import sys
import warnings

import pandas
import quantstats
import seaborn

from .base import Exporter
from .model import Snapshot


class QuantStatsExporter(Exporter):

    def __init__(
        self,
        html_output_file='report.html',
        csv_output_file='report.csv',
        benchmark_ticker="SPY",
        auto_delete=False,
        auto_override=False,
        fixed_nav=False,
    ):
        self.html_output_file = html_output_file
        self.csv_output_file = csv_output_file
        self.benchmark_ticker = benchmark_ticker
        self.auto_delete = auto_delete
        self.auto_override = auto_override
        self.fixed_nav = fixed_nav

        # Each row: (date, pre_reset_equity, post_reset_equity).
        # For non-rebalance days pre == post.  For fixed_nav rebalance days
        # pre_reset is the true market equity before cash extraction, and
        # post_reset is initial_cash (the denominator for the next day).
        self.rows = []

        warnings.filterwarnings(
            action='ignore',
            category=UserWarning,
            module=seaborn.__name__
        )

    def configure(self, fixed_nav: bool) -> None:
        self.fixed_nav = fixed_nav

    @abc.abstractmethod
    def initialize(self) -> None:
        if self.auto_override:
            return

        for file in [self.html_output_file, self.csv_output_file]:
            if file is None or not os.path.exists(file):
                continue

            can_delete = self.auto_delete
            if not can_delete:
                can_delete = input(
                    f"{file}: delete file? [y/N]").lower() == 'y'

            if can_delete:
                os.remove(file)

    @abc.abstractmethod
    def on_snapshot(self, snapshot: Snapshot) -> None:
        date = snapshot.date
        if snapshot.postponned is not None:
            date = snapshot.postponned

        if snapshot.ordered:
            if self.fixed_nav and self.rows and self.rows[-1][0] == date:
                # Rebalance day: subtract fees paid this period from pre_reset so
                # that the return = (market_gain - fees) / initial_cash.
                # post_reset is initial_cash (the denominator for the next period).
                date_, pre_reset, _ = self.rows[-1]
                self.rows[-1] = (date_, pre_reset - snapshot.total_fees, snapshot.equity)
            return

        # Non-ordered snapshot: pre_reset == post_reset (no cash extraction yet).
        self.rows.append((date, snapshot.equity, snapshot.equity))

    @abc.abstractmethod
    def finalize(self) -> None:
        df = pandas.DataFrame(
            self.rows,
            columns=["date", "pre_reset_equity", "post_reset_equity"]
        ).set_index("date")

        # Expose a single 'equity' column (post-reset) for backward compatibility.
        df["equity"] = df["post_reset_equity"]

        # Return on any day = (pre_reset - prev_post_reset) / prev_post_reset.
        # - Non-rebalance days: pre == post, so this is the standard formula.
        # - Rebalance days: pre_reset captures the true market move (net of fees);
        #   post_reset is always initial_cash (the cash-reset formula guarantees
        #   equity == initial_cash after every rebalance), becoming tomorrow's
        #   denominator and preventing compounding. On loss weeks capital is
        #   injected to restore the NAV — the negative return is still recorded
        #   correctly. Consecutive rebalance days are handled automatically
        #   because shift(1) always reads post_reset of the previous row.
        df["daily_profit_pct"] = (
            df["pre_reset_equity"] / df["post_reset_equity"].shift(1) - 1
        )

        self.dataframe = df

        if not len(self.dataframe):
            print(
                "[warning] cannot create tearsheet: dataframe is empty",
                file=sys.stderr
            )

            return

        history_df = self.dataframe.copy()

        history_df.reset_index(inplace=True)

        history_df['date'] = history_df['date'].astype(str)
        history_df['date'] = pandas.to_datetime(
            history_df['date'],
            format="%Y-%m-%d"
        )

        if self.benchmark_ticker:
            bench = quantstats.utils.download_returns(self.benchmark_ticker)

            bench = bench.reset_index()
            bench = bench.rename(columns={"Date": "date"})

            bench['date'] = pandas.to_datetime(
                bench['date'],
                format="%Y-%m-%d"
            ).dt.tz_localize(None)
            bench.rename({'Close': self.benchmark_ticker}, axis=1, inplace=True)

            merged = history_df.merge(bench, on='date', how='inner')

            merged.set_index('date', drop=True, inplace=True)

            returns = merged.daily_profit_pct
            benchmark = merged[self.benchmark_ticker]
        else:
            returns = history_df.set_index("date").daily_profit_pct
            benchmark = None

        if self.csv_output_file is not None:
            if self.auto_override or not os.path.exists(self.csv_output_file):
                returns.to_csv(self.csv_output_file)
            else:
                print(
                    f"[warning] {self.csv_output_file} already exists",
                    file=sys.stderr
                )

        self.returns = returns
        self.benchmark = benchmark

        if self.html_output_file is not None:
            if self.auto_override or not os.path.exists(self.html_output_file):
                quantstats.reports.html(
                    returns,
                    benchmark=benchmark,
                    output=self.html_output_file,
                    active_returns=False
                )
            else:
                print(
                    f"[warning] {self.html_output_file} already exists",
                    file=sys.stderr
                )
