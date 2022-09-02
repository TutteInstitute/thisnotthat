import panel as pn
import param
import numpy as np
import pandas as pd
import numpy.typing as npt

from typing import *


class SearchWidget(pn.reactive.Reactive):
    """A search pane that can be used to search for samples in a dataframe and select matching samples. If linked with
    a PlotPane this allows for search results to be selected in the plot for efficient visual representations of
    searches.

    The basic search pane provides three search modes: via string matching (potentially in a restricted set of
    columns in the dataframe), via regular expressions (again, potentially of selected columsn only) or by applying
    a query against the dataframe using the pandas ``query`` syntax.

    Parameters
    ----------
    raw_dataframe: DataFrame
        The dataframe to associate with data in a map representation in a PlotPane. The dataframe should have one row
        per sample in the map representation, and be in the same order as the data in the map representation.

    title: str (optional, default = "#### Search")
        A markdown title to be placed at the top of the pane.

    width: int or None (optional, default = None)
        The width of the pane, or, if ``None`` let the pane size itself.

    height: int or None (optional, default = None)
        The height of the pane, or, if ``None`` let the pane size itself.

    name: str (optional, default = "Label Editor")
        The panel name of the pane. See panel documentation for more details.
    """

    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def __init__(
        self,
        raw_dataframe: pd.DataFrame,
        *,
        title: str = "#### Search",
        width: Optional[int] = None,
        height: Optional[int] = None,
        name: str = "Search",
    ) -> None:
        super().__init__(name=name)
        if np.all(raw_dataframe.index.array == np.arange(len(raw_dataframe))):
            self.data = raw_dataframe
        else:
            self.data = raw_dataframe.reset_index()

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
        self.warning_area.visible = False
        self.pane = pn.WidgetBox(
            title,
            self.query_style_selector,
            self.query_box,
            self.query_button,
            self.columns_to_search,
            self.warning_area,
            width=width,
            height=height,
        )

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
        self.warning_area.visible = False
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
                    self.warning_area.visible = True
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
                    self.warning_area.visible = True
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
                    self.warning_area.visible = True
            except Exception as err:
                self.warning_area.alert_type = "danger"
                self.warning_area.object = str(err)
                self.warning_area.visible = True

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    def link_to_plot(self, plot):
        """Link this pane to a plot pane using a default set of params that can sensibly be linked.

        Parameters
        ----------
        plot: PlotPane
            The plot pane to link to.

        Returns
        -------
        link:
            The link object.
        """
        return self.link(plot, selected="selected", bidirectional=True)
