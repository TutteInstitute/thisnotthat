from typing import Callable, Optional, Protocol, Sequence

from bokeh.layouts import LayoutDOM
import bokeh.plotting as bpl
import numpy as np
import pandas as pd
import panel as pn
import param
from sklearn.linear_model import LogisticRegression


class DataFrameSummarizer(Protocol):
    """
    See DataSummaryPane for details on setting up objects that follow this protocol.
    """
    def summarize(self, selected: Sequence[int]) -> pd.DataFrame: ...


DataFrameNoSelection = Callable[[], pd.DataFrame]


class DataSummaryPane(pn.reactive.Reactive):
    """
    A data pane that generates a summary of the selected points in the form of a
    data frame.

    summarizer
        Any object with a ``summarize`` method that will generate and return a data
        frame, given the list of indices of the selected points. This method will not
        be called when no point is selected, so one may assume the list of indices not
        to be empty. The ``thisnotthat.summary`` module contains a set of such
        summarizer classes.

    no_selection
        Parameter-less function that returns the data frame to display when nothing
        is selected.

    width
    height
    sizing_mode
        Geometry of the displayed frame. See Panel documentation.

    name
        Name of the pane.


    Example: (in a Jupyter notebook)

    import numpy as np
    import pandas as pd

    data = pd.DataFrame({
        "x": [8, 3, 9, -2, 4],
        "y": [7, 0, 11, -5, -3]
    })

    class CentroidSummarizer:

        def summarize(self, selected):
            return pd.DataFrame(
                data=[np.mean(data.iloc[selected])],
                columns=["x", "y"],
                index=["Centroid"]
            )

    plot = tnt.BokehPlotPane(data, show_legend=False)
    summary = tnt.DataSummaryPane(CentroidSummarizer())
    summary.link_to_plot(plot)
    display(pn.Row(plot, summary))

    Now play with the lasso tool to select points, and see the data pane show the
    centroid of the selection.
    """

    selected = param.List(default=[], doc="Indices of selected samples")

    def __init__(
        self,
        summarizer: DataFrameSummarizer,
        no_selection: DataFrameNoSelection = lambda: pd.DataFrame(
            columns=["Nothing to summarize"]
        ),
        width: Optional[int] = None,
        height: Optional[int] = None,
        sizing_mode: str = "stretch_both",
        name: str = "Summary"
    ) -> None:
        super().__init__(name=name)
        self.summarizer = summarizer
        self.no_selection = no_selection
        self._base_selection = []
        self.table = pn.pane.DataFrame(
            self.no_selection(),
            sizing_mode=sizing_mode,
            width=width,
            height=height
        )
        self.pane = pn.Column(self.table, sizing_mode=sizing_mode)

    def _get_model(self, *args, **kwargs):
        return self.pane._get_model(*args, **kwargs)

    @param.depends("selected", watch=True)
    def _update_selected(self) -> None:
        self.table.object = (
            self.summarizer.summarize(self.selected)
            if self.selected
            else self.no_selection()
        )

    def link_to_plot(self, plot):
        """
        Link this pane to the plot pane to summarize using a default set of params
        that can sensibly be linked.

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

    @property
    def summary_dataframe(self):
        """
        The latest summary generated.
        """
        return self.table.object


class SummarizerValueCounts:
    """
    Summarizer for the DataSummaryPane that compiles a summary as the value counts
    of a data series indexed by the selection.

    Parameters
    ----------

    data: pd.Series
        The data corresponding to the plot points.
    top_k: int
        The number of values to keep out of the value counts.
    """

    def __init__(self, data: pd.Series, top_k: int = 20):
        self.data = data
        self.top_k = top_k

    def summarize(self, selected: Sequence[int]) -> pd.DataFrame:
        """
        Generate the summary, given the indices of the selected points.
        """
        data_selected = self.data.iloc[selected]
        return data_selected.value_counts().to_frame().head(self.top_k)


class PlotSummarizer(Protocol):

    def summarize(self, selected: Sequence[int]) -> LayoutDOM: ...


def display_no_selection() -> LayoutDOM:
    fig = bpl.figure()
    fig.text([0], [0], ["Nothing to summarize."])
    return fig


PlotNoSelection = Callable[[], LayoutDOM]


class PlotSummaryPane(pn.reactive.Reactive):
    """
    A data pane that displays a summary of the selected points in the form of a
    plot. This display is regenerated every time the selection changes in the linked
    plot.

    summarizer
        Any object with a ``summarize`` method that will take the list of (indices of)
        selected points and produce (and return) a Bokeh displayable object that
        summarizes visually the selection. This method will not be called when no point
        is selected, so one may assume the list of indices not to be empty. The
        ``thisnotthat.summary`` module contains a set of such summarizer classes.

    no_selection
        Parameter-less function that returns a Bokeh displayable object that should come
        up when no point is selected.

    width
    height
    sizing_mode
        Geometry of the displayed figure. See Panel documentation.

    name
        Name of the pane.


    Example: (in a Jupyter notebook)

    from bokeh.plotting import figure
    import numpy as np
    import pandas as pd

    data = pd.DataFrame({
        "x": [8, 3, 9, -2, 4],
        "y": [7, 0, 11, -5, -3]
    })

    class MeansSummarizer:

        def summarize(self, selected):
            features = list(data.columns)
            fig = figure(y_range=features)
            fig.hbar(y=features, right=np.squeeze(np.mean(data.iloc[selected])))
            return fig

    plot = tnt.BokehPlotPane(data, show_legend=False)
    summary = tnt.DataSummaryPane(MeansSummarizer())
    summary.link_to_plot(plot)
    display(pn.Row(plot, summary))

    Now play with the lasso tool to select points, and see the data pane show a bar
    diagram of the mean features of the points.
    """

    selected = param.List(default=[], doc="Indices of selected samples")

    def __init__(
        self,
        summarizer: PlotSummarizer,
        no_selection: PlotNoSelection = display_no_selection,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sizing_mode: str = "stretch_both",
        name: str = "Summary"
    ) -> None:
        super().__init__(name=name)
        self.summarizer = summarizer
        self.no_selection = no_selection
        self._base_selection = []
        self._geometry_figure = {
            name: param
            for name, param in [("width", width), ("height", height)]
            if param
        }
        self._geometry_pane = {
            **self._geometry_figure,
            **{
                name: param
                for name, param in [("sizing_mode", sizing_mode)]
                if param
            }
        }
        self.summary_plot = pn.pane.Bokeh(
            self.no_selection(),
            sizing_mode=sizing_mode,
            width=width,
            height=height
        )
        self.pane = pn.Column(self.summary_plot, sizing_mode=sizing_mode)

    def _get_model(self, *args, **kwargs):
        return self.pane._get_model(*args, **kwargs)

    @param.depends("selected", watch=True)
    def _update_selected(self) -> None:
        # self.pane[0] = pn.pane.Bokeh(
        #     fig,
        #     **self._geometry_pane
        # )
        self.summary_plot.object = (
            self.summarizer.summarize(self.selected)
            if self.selected
            else self.no_selection()
        )

    def link_to_plot(self, plot):
        """
        Link this pane to the plot pane to summarize using a default set of params
        that can sensibly be linked.

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


class FeatureImportanceSummarizer:

    def __init__(
        self,
        data: pd.DataFrame,
        max_features: int = 15,
        tol_importance_relative: float = 0.01
    ) -> None:
        self.data = data.select_dtypes(np.number)  # Indexed 0 to length.
        self.max_features = max_features
        self.tol_importance_relative = tol_importance_relative

    def summarize(self, selected: Sequence[int]) -> LayoutDOM:
        classes = np.zeros((len(self.data),), dtype="int32")
        classes[selected] = 1
        classifier = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced"
        ).fit(
            self.data.to_numpy(),
            classes
        )
        assert classifier.coef_.shape[0] == 1 or classifier.coef_.ndim == 1
        importance = np.squeeze(classifier.coef_)
        index_importance = np.argsort(-np.abs(importance))[:self.max_features]
        importance_abs = np.abs(importance)[index_importance]
        importance_relative = importance_abs / np.max(importance_abs)
        importance_restricted = importance[
            np.where(importance_relative > self.tol_importance_relative)
        ]

        x = list(self.data.columns[index_importance[:len(importance_restricted)]])
        fig = bpl.figure(x_range=x)
        fig.vbar(
            x=x,
            top=importance[index_importance[:len(importance_restricted)]],
            width=0.8
        )
        return fig
