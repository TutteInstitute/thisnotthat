import panel as pn
import param
import pandas as pd
from io import BytesIO
import numpy as np
import numpy.typing as npt

from typing import *


class SimpleDataPane(pn.reactive.Reactive):
    """A dataframe viewer that can be linked to a PlotPane based on selections and labels. The pane is essentially a
    wrapper around the panel DataFrame pane to make selection from a PlotPane easy to reflect in the table, as well as
    providing built in support for working with labels. It also provides built in tooling for downloading a selected
    dataset for further triage. This particular dataview is relatively straightforward and doesn't provide a lot
    of interactivity. For a richer data view consider the ``DataPane`` class which uses the tabulator interface
    for viewing data tables.

    This pane is best used as an accompaniment to a PlotPane, allowing a user to easily go from selections of interesting
    regions in data map representation back to the source data, or other associated metadata.

    Parameters
    ----------
    raw_dataframe: DataFrame
        The dataframe to associate with data in a map representation in a PlotPane. The dataframe should have one row
        per sample in the map representation, and be in the same order as the data in the map representation.

    labels: Array of shape (n_samples,) or None (optional, default None)
        If there are class labels associated to the data, they can be passed in here; they will be appended as a new
        column to the dataframe, and exposed as a param of the pane -- so labels edited with the label editor will
        automatically get updated in the data view.

    max_rows: int (optional, default = 20)
        The maximum number of rows to display from the dataframe

    max_cols: int (optional, default = 20)
        The maximum number of columns to display from the dataframe

    width: int or None (optional, default = None)
        The width of the data pane.

    height: int or None (optional, default = None)
        The height of the data pane.

    sizing_mode: str (optional, default = "stretch_both")
        The panel sizing mode of the data table.

    name: str (optional, default = "Data Table")
        The panel name for the pane. See the panel documentation for more details.
    """

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def __init__(
        self,
        raw_dataframe: pd.DataFrame,
        *,
        labels: Optional[npt.ArrayLike] = None,
        max_rows: int = 20,
        max_cols: int = 20,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sizing_mode: str = "stretch_both",
        name: str = "Data Table",
    ):
        super().__init__(name=name)
        if np.all(raw_dataframe.index.array == np.arange(len(raw_dataframe))):
            self.data = raw_dataframe.copy()
        else:
            self.data = raw_dataframe.reset_index().rename(
                columns={"index": "original_index"}
            )
        self.data.index.name = "row_num"

        self._base_selection: List[int] = []

        self.table = pn.pane.DataFrame(
            self.data,
            max_rows=max_rows,
            max_cols=max_cols,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
        )
        self.file_download = pn.widgets.FileDownload(
            filename="data.csv", callback=self._get_csv, button_type="primary", sizing_mode="stretch_width"
        )
        self.pane = pn.Column(self.table, self.file_download, sizing_mode=sizing_mode)
        if labels is not None:
            self.labels = pd.Series(labels).copy()

    def _get_csv(self) -> BytesIO:
        return BytesIO(self.table.object.to_csv().encode())

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("selected", watch=True)
    def _update_selected(self) -> None:
        if len(self.selected) == 0:
            self.table.object = self.data
        else:
            self.table.object = self.data.iloc[self.selected]

    @param.depends("labels", watch=True)
    def _update_labels(self) -> None:
        self.data["label"] = self.labels.values
        if len(self.selected) > 0:
            self.table.object = self.data.iloc[self.selected]
        else:
            self.table.object = self.data

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
        return self.link(plot, labels="labels", selected="selected", bidirectional=True)

    @property
    def selected_dataframe(self):
        return self.table.object


class DataPane(pn.reactive.Reactive):
    """A dataframe viewer that can be linked to a PlotPane based on selections and labels. The pane is essentially a
    wrapper around the panel tabulator widget to make selection handling back and forth between the table and a
    PlotPane easy, as well as providing built in support for working with labels. It also provides built in
    tooling for downloading a selected dataset for further triage.

    This pane is best used as an accompaniment to a PlotPane, allowing a user to easily go from selections of interesting
    regions in data map representation back to the source data, or other associated metadata.

    Parameters
    ----------
    raw_dataframe: DataFrame
        The dataframe to associate with data in a map representation in a PlotPane. The dataframe should have one row
        per sample in the map representation, and be in the same order as the data in the map representation.

    labels: Array of shape (n_samples,) or None (optional, default None)
        If there are class labels associated to the data, they can be passed in here; they will be appended as a new
        column to the dataframe, and exposed as a param of the pane -- so labels edited with the label editor will
        automatically get updated in the data view.

    width: int (optional, default = 600)
        The width of the data pane.

    height: int (optional, default = 600)
        The height of the data pane.

    tabulator_configuration: dict (optional, default = {})
        A dictionary of tabulator configuration. See panel's tabulator pane documentation, or the tabulator
        documentation for further details.

    formatters: dict (optional, default = {})
        A dictionary of how tabulator should format data in each column. See panel's tabulator pane documentation,
        or the tabulator documentation for further details.

    header_align: str or dict (optional, default = "center")
        A string specifying alignment of headers, or a dictionary specifying alignment on a per-column basis.

    hidden_columns: list of str (optional, default = [])
        The column names of columns to be hidden/suppressed.

    layout: str (optional, default = "fit_data_table")
        Layout handling for the tabulator table. See panel's tabulator pane documentation,  or the tabulator
        documentation for further details.

    frozen_columns: list of str (optional, default = [])
        The column names of columns to keep frozen (always visible).

    page_size: int (optional, default = 20)
        The data table is broken up into pages; this is the number of rows to appear in each page.

    row_height: int (optional, default = 30)
        The height of rows in the data table.

    show_index: bool (optional, default = True)
        Show the dataframe row index in the table.

    sorters: list of dicts (optional, default = [])
        How to handle sorting in the table. See panel's tabulator pane documentation,  or the tabulator
        documentation for further details.

    theme: str (optional, default = "materialize")
        The tabulator theme to use for the table. See the tabulator documentation for more details.

    widths: dict (optional, default = {})
        A dictionary giving the widths of each column (keyed by column name).

    name: str (optional, default = "Data Table")
        The panel name for the pane. See the panel documentation for more details.
    """

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def __init__(
        self,
        raw_dataframe: pd.DataFrame,
        *,
        labels: Optional[npt.ArrayLike] = None,
        width: int = 600,
        height: int = 600,
        tabulator_configuration: dict = {},
        formatters: dict = {},
        header_align: Union[dict, str] = "center",
        hidden_columns: List[str] = [],
        layout: str = "fit_data_table",
        frozen_columns: List[str] = [],
        page_size: int = 20,
        row_height: int = 30,
        show_index: bool = True,
        sorters: List[Dict[str, str]] = [],
        theme: str = "materialize",
        widths: Dict[str, int] = {},
        name: str = "Data Table",
    ) -> None:
        super().__init__(name=name)
        if np.all(raw_dataframe.index.array == np.arange(len(raw_dataframe))):
            self.data = raw_dataframe.copy()
        else:
            self.data = raw_dataframe.reset_index().rename(
                columns={"index": "original_index"}
            )
        self.data.index.name = "row_num"

        self._base_selection: List[int] = []
        self.table = pn.widgets.Tabulator(
            self.data,
            pagination="remote",
            page_size=page_size,
            width=width,
            height=height,
            configuration=tabulator_configuration,
            formatters=formatters,
            header_align=header_align,
            hidden_columns=hidden_columns,
            layout=layout,
            frozen_columns=frozen_columns,
            row_height=row_height,
            show_index=show_index,
            sorters=sorters,
            theme=theme,
            widths=widths,
            selectable="checkbox",
            disabled=True,
        )
        self._table_watch = self.table.param.watch(
            self._update_table_selection, "selection"
        )
        self.file_download = pn.widgets.FileDownload(
            filename="data.csv", callback=self._get_csv, button_type="primary"
        )
        self.pane = pn.Column(self.table, self.file_download)
        if labels is not None:
            self.labels = pd.Series(labels).copy()

    def _get_csv(self) -> BytesIO:
        return BytesIO(self.table.value.to_csv().encode())

    def _update_table_selection(self, event: param.parameterized.Event) -> None:
        if len(event.old) == 0:
            self._base_selection = self.selected
        if len(event.new) > 0:
            self.selected = self.table.value.index[event.new].to_list()
        elif len(event.new) == 0 and len(event.old) > 0:
            self.selected = self._base_selection

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("selected", watch=True)
    def _update_selected(self) -> None:
        if len(self.table.selection) != len(self.selected):
            self.table.selection = []
            self.table.value = self.data.iloc[self.selected]
        if len(self.selected) == 0:
            self.table.value = self.data
        else:
            self.table.value = self.data.iloc[self.selected]

    @param.depends("labels", watch=True)
    def _update_labels(self) -> None:
        self.data["label"] = self.labels.values
        if len(self.selected) > 0:
            self.table.value = self.data.iloc[self.selected]
        else:
            self.table.value = self.data

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
        return self.link(plot, labels="labels", selected="selected", bidirectional=True)

    @property
    def selected_dataframe(self):
        return self.table.value
