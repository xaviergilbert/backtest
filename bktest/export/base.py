import datetime
import typing

from .model import Snapshot


class Exporter:

    def initialize(self) -> None:
        pass

    def on_skip(self, date: datetime.date, reason: str, ordered: bool) -> None:
        pass

    def on_snapshot(self, snapshot: Snapshot) -> None:
        pass

    def finalize(self) -> None:
        pass

    def configure(self, fixed_nav: bool) -> None:
        pass


class ExporterCollection:

    def __init__(
        self,
        elements: typing.List[Exporter] = [],
    ):
        self.elements = [] if elements is None else elements

    def fire_initialize(self):
        for exporter in self.elements:
            exporter.initialize()

    def fire_finalize(self):
        for exporter in self.elements:
            exporter.finalize()

    def fire_skip(
        self,
        date: datetime.date,
        reason: str,
        ordered: bool
    ):
        for exporter in self.elements:
            exporter.on_skip(date, reason, ordered)

    def configure(self, fixed_nav: bool) -> None:
        for exporter in self.elements:
            exporter.configure(fixed_nav)

    def fire_snapshot(
        self,
        date: datetime.date,
        account: "Account",
        result: "OrderResultCollection",
    ):
        cash = float(account.cash)
        equity = float(account.equity)
        holdings = account.holdings
        ordered = result is not None

        snapshot = Snapshot(
            date=date,
            cash=cash,
            equity=equity,
            holdings=holdings,
            ordered=ordered,
        )

        if ordered:
            snapshot.total_fees = result.total_fees
            snapshot.success_count = result.success_count
            snapshot.failed_count = result.failed_count

            snapshot.closed_count = result.closed_count
            snapshot.closed_total = result.closed_total

        for exporter in self.elements:
            exporter.on_snapshot(snapshot)
