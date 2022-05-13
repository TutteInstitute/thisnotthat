import panel as pn
import param
import numpy as np
import pandas as pd
import numpy.typing as npt

from typing import *


class SearchPane(pn.reactive.Reactive):

    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def _query_style_change(self, event: param.parameterized.Event) -> None:
        if event.new == "Pandas query":
            self._saved_col_to_search = self.columns_to_search.value
            self.columns_to_search.value = []
            self.columns_to_search.disabled = True
        else:
            if hasattr(self, "_saved_col_to_search"):
                self.columns_to_search.value = self._saved_col_to_search
            self.columns_to_search.disabled = False

    def _run_query(self, event: param.parameterized.Event) -> None:
        self.warning_area.alert_type = "light"
        self.warning_area.object = ""
        if len(self.query_box.value) == 0:
            self.selected = []
        elif self.query_style_selector.value == "String search":
            try:
                indices = []
                for col in self.columns_to_search.value or self.data:
                    if hasattr(self.data[col], "str"):
                        new_indices = np.where(
                            self.data[col].str.contains(
                                self.query_box.value, regex=False
                            )
                        )[0].tolist()
                        indices.extend(new_indices)
                if len(indices) == 0:
                    self.warning_area.alert_type = "warning"
                    self.warning_area.object = (
                        f"No matches found for search string {self.query_box.value}!"
                    )
                self.selected = sorted(indices)
            except Exception as err:
                self.warning_area.alert_type = "danger"
                self.warning_area.object = str(err)
        elif self.query_style_selector.value == "Regex":
            try:
                indices = []
                for col in self.columns_to_search.value or self.data:
                    if hasattr(self.data[col], "str"):
                        new_indices = np.where(
                            self.data[col].str.contains(
                                self.query_box.value, regex=True
                            )
                        )[0].tolist()
                        indices.extend(new_indices)
                if len(indices) == 0:
                    self.warning_area.alert_type = "warning"
                    self.warning_area.object = f"No matches found for search with regex {self.query_box.value}!"
                self.selected = sorted(indices)
            except Exception as err:
                self.warning_area.alert_type = "danger"
                self.warning_area.object = str(err)
        elif self.query_style_selector.value == "Pandas query":
            try:
                self.selected = (
                    self.data.reset_index().query(self.query_box.value).index.tolist()
                )
                if len(self.selected) == 0:
                    self.warning_area.alert_type = "warning"
                    self.warning_area.object = f"No matches found for search with pandas query {self.query_box.value}!"
            except Exception as err:
                self.warning_area.alert_type = "danger"
                self.warning_area.object = str(err)

    def __init__(self, raw_dataframe: pd.DataFrame) -> None:
        super().__init__()
        self.data = raw_dataframe
        self.query_box = pn.widgets.TextAreaInput(
            name="Search query",
            placeholder="Enter search here ...",
            min_height=64,
            height=128,
        )
        self.query_style_selector = pn.widgets.RadioButtonGroup(
            name="Query type",
            options=["String search", "Regex", "Pandas query"],
            button_type="primary",
        )
        self.query_button = pn.widgets.Button(name="Search", button_type="success")
        self.query_button.on_click(self._run_query)
        self.columns_to_search = pn.widgets.MultiChoice(
            name="Columns to search (all if empty)", options=self.data.columns.tolist(),
        )
        self.query_style_selector.param.watch(self._query_style_change, "value")
        self.warning_area = pn.pane.Alert("", alert_type="light")
        self.pane = pn.WidgetBox(
            self.query_style_selector,
            self.query_box,
            self.query_button,
            self.columns_to_search,
            self.warning_area,
        )

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)
