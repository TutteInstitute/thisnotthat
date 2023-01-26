import panel as pn
import param
import numpy as np
import pandas as pd
import numpy.typing as npt
import io

from bokeh.models import ColumnDataSource
from .bokeh_plot import BokehPlotPane

from pynndescent import NNDescent
from PIL import Image

from typing import *


def SimpleSearchWidget(
    plot: BokehPlotPane,
    *,
    raw_dataframe: Optional[pd.DataFrame] = None,
    title: str = "##### Search",
    placeholder_text: str = "Enter search string ...",
    live_search: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    sizing_mode: str = "stretch_width",
    name: str = "Search",
):
    """Construct a simple search widget attached to a specific ``BokehPlotPane``. This allows for basic search in a very
    simple set-up. Notably the search is performed client-side in javascript, and so should work even without a
    python kernel backend.

    Parameters
    ----------
    plot: BokehPlotPane
        The particular plot pane the search should be attached to.

    raw_dataframe: dataframe or None (optional, default = None)
        A dataframe to run search over. If set to None then the search will be run over the dataframe associated to the
        plot. Each column of the dataframe will be searched with matches on any rows that contain the search string
        as a substring.

    title: str (optional, default = "#### Search")
        A title for the associated search widget in markdown format.

    placeholder_text: str (optional, default = "Enter search string ...")
        Text to place in the search input field when no search text is provided.

    live_search: bool (optional, default = True)
        If True then perform the search on every key-press; if False then perform search only when the enter key is
        pressed.

    width: int or None (optional, default = None)
        The width of the pane, or, if ``None`` let the pane size itself.

    height: int or None (optional, default = None)
        The height of the pane, or, if ``None`` let the pane size itself.

    sizing_mode: str (optional, default = "stretch_both")
        The panel sizing mode of the data table.

    name: str (optional, default = "Search")
        The panel name of the pane. See panel documentation for more details.

    Returns
    -------
    search_widget: pn.WidgetBox
        A search widget that is linked to the specified ``BokehPlotPane``.
    """
    if raw_dataframe is not None:
        search_datasource = ColumnDataSource(raw_dataframe)
    else:
        search_datasource = plot.data_source
    search_box = pn.widgets.TextInput(
        placeholder=placeholder_text, align=("start", "center"), sizing_mode=sizing_mode
    )
    result = pn.WidgetBox(
        pn.pane.Markdown(title, align=("end", "center")),
        search_box,
        horizontal=True,
        sizing_mode=sizing_mode,
        width=width,
        height=height,
        name=name,
    )
    if live_search:
        search_box.jscallback(
            value_input="""
var data = data_source.data;
var text_search = search_box.value_input;

// Loop over columns and values
// If there is no match for any column for a given row, change the alpha value
var string_match = false;
var selected_indices = [];
for (var i = 0; i < plot_source.data.x.length; i++) {
    string_match = false
    for (const column in data) {
        if (String(data[column][i]).includes(text_search) ) {
            string_match = true;
        }
    }
    if (string_match){
        selected_indices.push(i);
    }
}
plot_source.selected.indices = selected_indices;
plot_source.change.emit();
        """,
            args={
                "plot_source": plot.data_source,
                "data_source": search_datasource,
                "search_box": search_box,
            },
        )
    else:
        search_box.jscallback(
            value="""
var data = data_source.data;
var text_search = search_box.value;

// Loop over columns and values
// If there is no match for any column for a given row, change the alpha value
var string_match = false;
var selected_indices = [];
for (var i = 0; i < plot_source.data.x.length; i++) {
    string_match = false
    for (const column in data) {
        if (String(data[column][i]).includes(text_search) ) {
            string_match = true;
        }
    }
    if (string_match){
        selected_indices.push(i);
    }
}
plot_source.selected.indices = selected_indices;
plot_source.change.emit();
            """,
            args={
                "plot_source": plot.data_source,
                "data_source": search_datasource,
                "search_box": search_box,
            },
        )
    return result


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

    name: str (optional, default = "Search")
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
            name="Columns to search (all if empty)",
            options=self.data.columns.tolist(),
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


class VectorSearchWidget(pn.reactive.Reactive):
    """A search pane that can be used for searching for vector databases of content given a means to convert search
    queries into vector representations. The search uses pynndescent for fast vector searching. You must supply
    both vectors (one for each sample in the plot, in the same order), and a function or callable to convert the
    search input into vectors in the same space. Search input can either be text, or files. In file input is used
    then the content passed to the embedder is either text (in the case of text files), a numpy array
    (in the case of images), or a raw bytestring (in the case of other file types that aren't recognized). The embedder
    callable needs to be able to appropriately handle the content of these forms.

    Parameters
    ----------

    vectors_to_query: ArrayLike
        An array of vectors to be searched over.

    embedder: Callable
        A Callable or function that can take the search input and return a vector representing the input that exists
        in the same embedding space as ``vectors_to_query``.

    title: str (optional, default = "#### Search")
        A markdown title to be placed at the top of the pane.

    vector_metric: str (optional, default = "cosine")
        The metric to use for searching over the ``vectors_to_query``. Any metric supported by pynndescent is valid.

    input_type: str (optional, default = "text")
        Either "text" or "file" depending on how you wish to supply query data.

    placeholder_text: str (optional, default = "Enter keywords ...")
        Text to place in the search input field when no search text is provided.

    n_query_results: int (optional, default = 20)
        The default number of query results to return.

    max_query_results: int (optional, default = 100)
        The number of results returned can be set by a slides in the widget; this value determines the maximum value
        of that slider.

    pynnd_n_neighbors: int (optional, default = 60)
        The ``n_neighbors`` value to use for pynndescent; larger values result in more accurate searches with longer
        search times. The default value of 60 provides reasonable search times with good accuracy for cosine metrics.
        A smaller value can be used for Euclidean metrics. Other metrics may require some tuning.

    sizing_mode: str (optional, default = "stretch_both")
        The panel sizing mode of the search widget.

    width: int or None (optional, default = None)
        The width of the pane, or, if ``None`` let the pane size itself.

    height: int or None (optional, default = None)
        The height of the pane, or, if ``None`` let the pane size itself.

    name: str (optional, default = "Search")
        The panel name of the pane. See panel documentation for more details.
        """
    selected = param.List(default=[], doc="Indices of selected samples")

    def __init__(
        self,
        vectors_to_query: npt.ArrayLike,
        embedder: Callable,
        title: str = "#### Search",
        vector_metric: str = "cosine",
        input_type: Literal["text", "file"] = "text",
        placeholder_text: str = "Enter search string ...",
        n_query_results: int = 20,
        max_query_results: int = 100,
        pynnd_n_neighbors: int = 60,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sizing_mode: str = "stretch_width",
        name: str = "Search",
    ) -> None:
        super().__init__(name=name)

        self._search_index = NNDescent(vectors_to_query, metric=vector_metric, n_neighbors=pynnd_n_neighbors)
        self._search_index.prepare()
        self._embedder = embedder
        self._input_type = input_type

        self.search_button = pn.widgets.Button(name="Search", button_type="success")
        self.search_button.disabled = True
        self.search_button.on_click(self._run_query)

        if input_type == "text":
            self.search_box = pn.widgets.TextInput(
                placeholder=placeholder_text,
                align=("start", "center"),
                sizing_mode=sizing_mode,
            )
            self.search_box.param.watch(
                self._button_state, ["value_input"], onlychanged=True
            )
        elif input_type == "file":
            self.search_box = pn.widgets.FileInput(
                align=("start", "center"),
                sizing_mode=sizing_mode,
                multiple=False,
            )
            self.search_box.param.watch(
                self._button_state, ["value"], onlychanged=True
            )
        else:
            raise ValueError(f"Invalid input type {input_type}. Should be one of 'text' or 'file'")

        self.n_results_slider = pn.widgets.DiscreteSlider(
            name="Number of results", options=list(range(0, max_query_results, 10)), value=n_query_results
        )
        self.n_results_slider.param.watch(self._button_state, ["value"], onlychanged=True)

        self.pane = pn.WidgetBox(
            title,
            pn.Row(self.search_box, self.search_button),
            self.n_results_slider,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
        )


    def _button_state(self, *events) -> None:
        self.search_button.disabled = False

    def _run_query(self, event: param.parameterized.Event) -> None:
        if self._input_type == "text":
            query_vector = self._embedder([self.search_box.value])
        elif self._input_type == "file":
            if self.search_box.mime_type.startswith("image"):
                raw_img = io.BytesIO()
                self.search_box.save(raw_img)
                img = Image.open(raw_img)
                query_vector = self._embedder(np.asarray(img))
            elif self.search_box.mime_type.startswith("text"):
                query_vector = self._embedder(self.search_box.value.decode())
            else:
                query_vector = self._embedder(self.search_box.value)

        if query_vector.ndim == 1:
            result_indices, result_dists = self._search_index.query([query_vector], k=self.n_results_slider.value)
        else:
            result_indices, result_dists = self._search_index.query(query_vector, k=self.n_results_slider.value)

        self.selected = [int(x) for x in result_indices[0]]
        self.search_button.disabled = True


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
