import panel as pn
import param
import pandas as pd
from io import BytesIO


class DataPane(pn.reactive.Reactive):
    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def _get_csv(self):
        return BytesIO(self.table.value.to_csv().encode())

    def _data_pane_update_selected(self, event):
        if len(event.old) == 0:
            self._base_selection = self.selected
        if len(event.new) > 0:
            self.selected = self.table.value.index[event.new].to_list()
        elif len(event.new) == 0 and len(event.old) > 0:
            self.selected = self._base_selection

    def __init__(self, labels, raw_dataframe):
        super().__init__()
        self.data = raw_dataframe.copy()
        self.data["label"] = labels
        self._base_selection = []
        self.table = pn.widgets.Tabulator(
            self.data,
            pagination="remote",
            page_size=20,
            width=600,
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
        self.labels = labels

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("selected", watch=True)
    def _data_pane_update_selection(self):
        if len(self.table.selection) != len(self.selected):
            self.table.selection = []
            self.table.value = self.data.iloc[self.selected]

    @param.depends("labels", watch=True)
    def _update_labels(self):
        self.data["label"] = self.labels
        if len(self.selected) > 0:
            self.table.value = self.data.iloc[self.selected]
