"""Header bar widget for Orchestra TUI."""
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static


class HeaderBar(Horizontal):
    run_id: reactive[str] = reactive("N/A")
    elapsed: reactive[str] = reactive("00:00")
    cost: reactive[str] = reactive("$0.0000")
    mode: reactive[str] = reactive("--")

    t_active: reactive[bool] = reactive(True)
    i_active: reactive[bool] = reactive(True)
    o_active: reactive[bool] = reactive(True)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-group filters"):
            yield Static("[T]hinking", id="flt-t", classes="filter-badge")
            yield Static("[I]nput/Tool", id="flt-i", classes="filter-badge")
            yield Static("[O]utput", id="flt-o", classes="filter-badge")

        with Horizontal(classes="header-group meta"):
            yield Static("MODE:", classes="meta-label")
            yield Static(self.mode, id="meta-mode", classes="meta-value")
            yield Static("RUN:", classes="meta-label")
            yield Static(self.run_id, id="meta-run-id", classes="meta-value")
            yield Static("TIME:", classes="meta-label")
            yield Static(self.elapsed, id="meta-elapsed", classes="meta-value")
            yield Static("COST:", classes="meta-label")
            yield Static(self.cost, id="meta-cost", classes="meta-value")

    def _set_badge_active(self, badge_id: str, is_active: bool) -> None:
        badge = self.query_one(f"#{badge_id}", Static)
        if is_active:
            badge.remove_class("dimmed")
        else:
            badge.add_class("dimmed")

    def watch_t_active(self, val: bool) -> None:
        self._set_badge_active("flt-t", val)

    def watch_i_active(self, val: bool) -> None:
        self._set_badge_active("flt-i", val)

    def watch_o_active(self, val: bool) -> None:
        self._set_badge_active("flt-o", val)

    def watch_run_id(self, val: str) -> None:
        self.query_one("#meta-run-id", Static).update(val)

    def watch_elapsed(self, val: str) -> None:
        self.query_one("#meta-elapsed", Static).update(val)

    def watch_cost(self, val: str) -> None:
        self.query_one("#meta-cost", Static).update(val)

    def watch_mode(self, val: str) -> None:
        self.query_one("#meta-mode", Static).update(val)
