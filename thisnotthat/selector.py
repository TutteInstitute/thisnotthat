import pandas as pd
import panel as pn
from typing import *


class TimeSelectorWidget(pn.widgets.DatetimeRangeSlider):

    def __init__(self, series_time: pd.Series, step: int = 1000, **opts) -> None:
        self._time = series_time.copy().reset_index(drop=True)
        m = self._time.min()
        M = self._time.max()
        super().__init__(start=m, end=M, value=(m, M), step=step, **opts)
        self._plot = None
        self.link(None, callbacks={"value": self._on_change})

    def _on_change(self, _, event) -> None:
        if self._plot:
            self._plot.selected = self._time.loc[self._time.between(*event.new)].index.to_list()

    def link_to_plot(self, plot) -> None:
        self._plot = plot
