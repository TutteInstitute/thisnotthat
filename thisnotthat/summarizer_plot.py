from typing import Callable, Optional, Protocol, Sequence

import matplotlib.pyplot as plt
from bokeh.layouts import LayoutDOM
import bokeh.plotting as bpl
import numpy as np
import pandas as pd
import panel as pn
import param
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import make_column_transformer

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
    summary = tnt.PlotSummaryPane(MeansSummarizer())
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
    """
    Summarizer for the PlotSummaryPane that constructs a class balanced, L1 penalized,
    logistic regression between the selected points and the remaining data.
    Numeric variables have been centered and rescaled by their mean and variance to put them
    on similar scales in order to make the coefficients more comparable.
    Then it displays that feature importance in a bar plot.

    All of the standard caveates with using the coefficients of a linear model as a feature
    importance measure are included here.

    It might be worth reading the sklearn documentation on the
    Common pitfalls in the interpretation of coefficients of linear models
    (https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html)

    Parameters
    ----------

    data: pd.DataFrame
        A dataframe corresponding to the plot points.  The numeric features will
        be extracted from this dataframe for computing variable importance.
    max_features: int <default: 15>
        The maximum number of features to display the importance for.
    tol_importance_relative: float <default: 0.01>
        The minimum feature coefficient value in order to be considered important.
    one_hot_categorical_features: bool <default: True>
        Should the one hot encoding of the categorical features be included in our importance.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        max_features: int = 15,
        tol_importance_relative: float = 0.01,
        one_hot_categorical_features: bool = True,
    ) -> None:
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        if one_hot_categorical_features:
            preprocessor = make_column_transformer(
                (OneHotEncoder(drop="if_binary"), categorical_columns),
                (RobustScaler(), numeric_columns),
            )
        else:
            preprocessor = make_column_transformer(
                (StandardScaler(), numeric_columns),
            )
        self.data = preprocessor.fit_transform(data) # Indexed 0 to length.
        self.max_features = max_features
        self.tol_importance_relative = tol_importance_relative
        self._features = preprocessor.get_feature_names_out()

    def summarize(self, selected: Sequence[int]) -> LayoutDOM:
        classes = np.zeros((len(self.data),), dtype="int32")
        classes[selected] = 1
        classifier = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced"
        ).fit(
            self.data,
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

        selected_columns = self._features[index_importance[:len(importance_restricted)]]

        model_error = classifier.score(self.data, classes)
        fig = bpl.figure(
            y_range=selected_columns,
            title=f"Logistic Regression with a Mean Squared Error={model_error:.4}"
        )
        fig.hbar(
            y=selected_columns,
            right=importance[index_importance[:len(importance_restricted)]],
            height=0.8
        )
        plt.xlabel("Coefficient values corrected by the feature's std dev")
        return fig
