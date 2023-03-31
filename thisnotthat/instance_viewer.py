import panel as pn
import param
import pandas as pd
import time

from typing import *


class InformationPane(pn.reactive.Reactive):
    """An information pane allowing you to provide more details in a well formatted way on a single specific
    selected instance of data. The goal is similar to the ``DataPane`` -- to allow a user to go from points in the map 
    representation of data in a plot, back to the source data. In this case, however, rather than providing a table
    of information on all the selected points, we are providing well formatted information on a single point. This is
    particularly relevant in a "click-to-select" model where clocking on a point can bring up further details in
    the InformationPane.
    
    This is handled by providing a dataframe of associated data, and a markdown template that can format a data instance 
    in a clean way.

    Parameters
    ----------
    raw_dataframe: DataFrame
        The dataframe to associate with data in a map representation in a PlotPane. The dataframe should have one row
        per sample in the map representation, and be in the same order as the data in the map representation.

    markdown_template: str
        A string in markdown providing formatting for a single row of data from the dataframe. Within the string
        a substitution of the row value of a column will be done where ``{column_name}`` appears in the string.

    width: int (optional, default = 200)
        The width of the pane.

    height: int (optional, default = 600)
        The height of the pane.

    placeholder_text: str (optional, default = "<center> ... nothing selected ...")
        Text to be displayed in the pane when no examples are selected.

    dedent: bool (optional, default = False)
        Whether to dedent the markdown text.

    disable_math: bool (optional, default = False)
        Whether to disable rendering of math via LaTeX in the markdown.

    extensions: List of str (optional, default = ["extra", "smarty", "codehilite"])
        Markdown rendering extensions to use. See the panel documentation for more details.

    style: dict (optional, default = {})
        Style information for makrdown rendering. See the panel documentation for more details.

    margin: List of int (optional, default = [5, 5])
        Margin padding around the markdown widget.

    name: str (optional, default = "Information")
        The panel name of the pane. See the panel documentation for more details.
    """

    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def __init__(
        self,
        raw_dataframe: pd.DataFrame,
        markdown_template: str,
        *,
        width: int = 200,
        height: int = 600,
        placeholder_text: str = "<center> ... nothing selected ...",
        dedent: bool = False,
        disable_math: bool = False,
        extensions: List[str] = ["extra", "smarty", "codehilite"],
        style: dict = {},
        margin: List[int] = [5, 5],
        throttle = 200,
        name: str = "Information",
    ):
        super().__init__(name=name)
        self.data = raw_dataframe
        self.markdown_template = markdown_template
        self.placeholder_text = placeholder_text
        self.throttle = throttle
        self._last_update = time.perf_counter() * 1000.0
        self.markdown = pn.pane.Markdown(
            self.placeholder_text,
            width=width - 20,
            height=height - 20,
            margin=margin,
            dedent=dedent,
            disable_math=disable_math,
            extensions=extensions,
            style=style,
        )
        self.pane = pn.Column(self.markdown, width=width, height=height, scroll=True)

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("selected", watch=True)
    def _info_pane_update_selection(self) -> None:
        if len(self.selected) == 0:
            self.markdown.object = self.placeholder_text
        else:
            if time.perf_counter() * 1000.0 - self._last_update > self.throttle:
                substitution_dict = {
                    col: self.data[col].iloc[self.selected[-1]] for col in self.data.columns
                }
                self.markdown.object = self.markdown_template.format(**substitution_dict)
                self._last_update = time.perf_counter() * 1000.0

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
        return plot.link(self, selected="selected", bidirectional=False)
