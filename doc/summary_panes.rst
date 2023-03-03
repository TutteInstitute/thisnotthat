Summary Pane
============

Summary panes are interactive panes for helping a user gain an understanding of a selected set of points.  They are best
thought of as wrappers around the core of a summary function.  There are currently two types of Summary Panes supported:
`PlotSummaryPane` and `DataSummaryPane`.

These are differentiated by the type of Pane returned, either a plot or a pandas DataFrame.  Each of them constructed
by being passed a summarizer object.  Summarizer objects can be found under the namespaces: `summary` in the submodules
`dataframe` and `plot`.

summary.dataframe
--------------------
* CountSelectedSummarizer
    * The simplest of all summarizers to simply return the number of points selected. Includes an optional weightparameter
* ValueCountsSummarizer
    * A summarizer that takes a categorical series and computes the pandas.value_counts of that series for the selected points.
* JointLabelSummarizer
    * A summarizer that takes a high dimensional joint embedding of your points and some labels and uses the centroid of your selected points and it's distance to your labels to compute a summary.

.. toctree::
   :maxdepth: 1
   :caption: summarizer dataframe examples:

   datasummarypane_value_counts_summarizer
   DataSummaryPane_JointLabelSummary


summary.plot
---------------
* FeatureImportanceSummarizer
    * A summarizer which computes an :math:`$l_1`-penalized logistic regression between the selected points and the remaining points and returns a bar plot of the top coefficient values.
* JointWordCloudSummarizer
    * A summarizer that takes a high dimensional joint embedding of your points and some labels and uses the centroid of your selected points and it's distance to your labels to compute a summary, then returns a wordcloud for visualization.

.. toctree::
   :maxdepth: 1
   :caption: summarizer plot examples:

   plotsummarypane_feature_importance
   PlotSummaryPane_JointWordCloutSummary

Custom Summarizers
------------------

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
   :maxdepth: 1
   :caption: custom summarizer examples:

   datasummarypane_custom_summarizer

.. _Panel library: https://panel.holoviz.org/
.. _numpy: https://numpy.org/
.. _pandas: https://pandas.pydata.org/
.. _umap-learn: https://umap-learn.readthedocs.io/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _numba: https://numba.pydata.org/
.. _pynndescent: https://pynndescent.readthedocs.io/en/latest/
.. _matplotlib: https://matplotlib.org/
.. _hdbscan: https://hdbscan.readthedocs.io/
.. _apricot-select: https://apricot-select.readthedocs.io/
.. _networkx: https://networkx.org/