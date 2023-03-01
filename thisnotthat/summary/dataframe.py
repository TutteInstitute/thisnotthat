from typing import Callable, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
import panel as pn
import param
from pynndescent import NNDescent


class DataFrameSummarizer(Protocol):
    """
    See DataSummaryPane for details on setting up objects that follow this protocol.
    """

    def summarize(self, selected: Sequence[int]) -> pd.DataFrame:
        ...


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
        name: str = "Summary",
    ) -> None:
        super().__init__(name=name)
        self.summarizer = summarizer
        self.no_selection = no_selection
        self._base_selection = []
        self.table = pn.pane.DataFrame(
            self.no_selection(), sizing_mode=sizing_mode, width=width, height=height
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


class ValueCountsSummarizer:
    """
    Summarizer for the DataSummaryPane that compiles a summary as the value_counts
    of a data series indexed by the selection.

    Parameters
    ----------

    data: iterator
        An iterator of length equal to the number of points to be summarized with indices aligning with those points
        and values to be used to summarize your data via pandas.value_counts()
    top_k: int
        The number of values to keep out of the value counts.
    """

    def __init__(self, data: pd.Series, top_k: int = 20):
        self.data = pd.Series(data)
        self.top_k = top_k

    def summarize(self, selected: Sequence[int]) -> pd.DataFrame:
        """
        Generate the summary, given the indices of the selected points.
        """
        data_selected = self.data.iloc[selected]
        return data_selected.value_counts().to_frame().head(self.top_k)


class CountSelectedSummarizer:
    """
    A simple summarizer to return a dataframe with a the count of the number
    of points selected.  If a weight vector is supplied then it returns
    the weighted count of the selected points.

    Parameters
    ----------

    weights: list(numeric) (optional, default = None)
        A weight vector to multiply our selected points by before summing.
    """

    def __init__(self, weights=None):
        self.weights = weights

    def summarize(self, selected: Sequence[int]) -> pd.DataFrame:
        """
        Generate the summary, given the indices of the selected points.
        """
        if self.weights:
            count = np.sum(self.weights[selected])
        else:
            count = len(selected)

        return pd.DataFrame({"value": count}, index=["count"])


class JointLabelSummarizer:
    """
    Computes the nearest neighbours of the centroid of the selected points in
    a provided joint vector space.  Then returns a DataFrame of the n_neighbours
    nearest high space labels to that centroid weighted by their distance.

    It makes use pynndescents approximate nearest neighbours algorithm to make
    this calculation efficient.  The use of an approximate nearest neighbours
    algorithm may introduce some errors into this process.

    This is the same process used to summarize topics in top2vec <reference>

    Parameters
    ----------
    vector_space: np.array
        A numpy array corresponding to the joint vector representation of your
        data points.
    labels: list(strings)
        A list of the labels of your high space points that you'd like to use to
        summarize your selected points.
    label_space: np.array
        A numpy array corresponding to joint vector representation of your labels.
        This should be a joint vector space between your labels and your points.
    vector_metric: str (optional, default = "cosine")
        The metric to use for searching over the ``vectors_to_query``. Any metric supported by pynndescent is valid.
    n_neighbours: int (optional, default = 10)
        The number of nearest neighbour labels from our label space to display.
    """

    def __init__(
        self, vector_space, labels, label_space, vector_metric="cosine", n_neighbours=10
    ):
        self.vector_space = vector_space
        self.labels = np.array(labels)
        self.label_space = label_space
        self.vector_metric = vector_metric
        self.n_neighbours = n_neighbours
        self._search_index = NNDescent(
            self.label_space, metric=vector_metric, n_neighbors=2 * self.n_neighbours
        )
        self._search_index.prepare()

    def summarize(self, selected):
        """
        Generate the summary, given the indices of the selected points.
        """
        # Compute the centroid of the selected points.
        self._centroid = np.mean(self.vector_space[selected, :], axis=0)
        # Query against the points in the label space
        result_indices, result_dists = self._search_index.query(
            [self._centroid], k=self.n_neighbours
        )
        return pd.DataFrame(
            {"labels": self.labels[result_indices[0]], "distances": result_dists[0]}
        )
