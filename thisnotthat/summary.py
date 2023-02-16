from typing import Callable, Optional, Protocol, Sequence

import pandas as pd
import panel as pn
import param


class DataFrameSummarizer(Protocol):
    """
    See SummaryDataPane for details on setting up objects that follow this protocol.
    """
    def summarize(self, selected: Sequence[int]) -> pd.DataFrame: ...


DataFrameNoSelection = Callable[[], pd.DataFrame]


class SummaryDataPane(pn.reactive.Reactive):
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
    summary = tnt.SummaryDataPane(CentroidSummarizer())
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
    Summarizer for the SummaryDataPane that compiles a summary as the value counts
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
