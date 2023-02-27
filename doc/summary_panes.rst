Summary Pane
============

Summary panes are interactive panes for helping a user gain an understanding of a selected set of points.  They are best
thought of as wrappers around the core of a summary function.  There are currently two types of Summary Panes supported:
`PlotSummaryPane` and `DataSummaryPane`.

These are differentiated by the type of Pane returned, either a plot or a pandas DataFrame.  Each of them constructed
by being passed a summarizer object.  Summarizer objects can be found under the namespaces: `summarizer_dataframe` and
`summarizer_plot`.

summarizer_dataframe
--------------------
* CountSelectedSummarizer
    * The simplest of all summarizers to simply return the number of points selected. Includes an optional weightparameter
* ValueCountsSummarizer
    * A summarizer that takes a categorical series and computes the pandas.value_counts of that series for the selected points.
* JointLabelSummarizer
    * A summarizer that takes a high dimensional joint embedding of your points and some labels and uses the centroid of your selected points and it's distance to your labels to compute a summary.

summarizer_plot
---------------
* FeatureImportanceSummarizer
    * A summarizer which computes an :math:`$l_1`-penalized logistic regression between the selected points and the remaining points and returns a bar plot of the top coefficient values.

Summarizers follow the following template example.  They are initialized with whatever information they require.
Then they have a `summarize` function which takes a `selected` variable.  The selected variable is a base zero
index into your data.  This is the basic variable that ties together all the Panes in ThisNotThat.  The summarize
function returns either a DataFrame or a plot and is passed to the corresponding SummaryPane.

.. code:: python3

    class CentroidSummarizer:

        def __init__(self, data):
            self.data = data

        def summarize(self, selected):
            indices = ['number selected', 'centroid']
            values = [len(selected), np.mean(self.data[selected,:], axis=0)]
            return pd.DataFrame({'values':values}, index=indices)

.. toctree::
   :maxdepth: 2
   :caption: Summary Panes:

    datasummarypane_value_counts_summarizer
    datasummarypane_custom_summarizer
    plotsummarypane_feature_importance
    DataSummaryPane_JointLabelSummary