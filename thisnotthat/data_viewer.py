import panel as pn
import param
import pandas as pd
from io import BytesIO
import numpy.typing as npt

from typing import *

class DataPane(pn.reactive.Reactive):

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def __init__(
            self,
            labels: npt.ArrayLike,
            raw_dataframe: pd.DataFrame,
            *,
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
    ) -> None:
        super().__init__()
        self.data = raw_dataframe.reset_index()
        self.data["label"] = labels
        self._base_selection = []
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
            self._data_pane_update_selected, "selection"
        )
        self.file_download = pn.widgets.FileDownload(
            filename="data.csv", callback=self._get_csv, button_type="primary"
        )
        self.pane = pn.Column(self.table, self.file_download)
        self.labels = pd.Series(labels)

    def _get_csv(self) -> BytesIO:
        return BytesIO(self.table.value.to_csv().encode())

    def _data_pane_update_selected(self, event: param.parameterized.Event) -> None:
        if len(event.old) == 0:
            self._base_selection = self.selected
        if len(event.new) > 0:
            self.selected = self.table.value.index[event.new].to_list()
        elif len(event.new) == 0 and len(event.old) > 0:
            self.selected = self._base_selection

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("selected", watch=True)
    def _data_pane_update_selection(self) -> None:
        if len(self.table.selection) != len(self.selected):
            self.table.selection = []
            self.table.value = self.data.iloc[self.selected]

    @param.depends("labels", watch=True)
    def _update_labels(self) -> None:
        self.data["label"] = self.labels
        if len(self.selected) > 0:
            self.table.value = self.data.iloc[self.selected]
        else:
            self.table.value = self.data
