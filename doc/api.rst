TNT API Guide
=============

TNT Provides a number of different Pane classes that can be combined and
linked in various ways. The core Panes are the Plot Panes, with various
associated other panes.

Plot Panes
----------

.. autoclass:: thisnotthat.bokeh_plot.BokehPlotPane
   :members:

.. autoclass:: thisnotthat.deck_plot.DeckglPlotPane
   :members:

Data Panes
----------

.. autoclass:: thisnotthat.instance_viewer.InformationPane
   :members:

.. autoclass:: thisnotthat.data_viewer.DataPane
   :members:

Search and Edit Panes
---------------------

.. autoclass:: thisnotthat.search.SearchPane
   :members:

.. autoclass:: thisnotthat.label_editor.LabelEditorPane
   :members:

.. autoclass:: thisnotthat.plot_controls.PlotControlPane
   :members:


Finally TNT provides tools for annotating plots with cluster labels.
There are various methods for achieving this.

Cluster Labelling Methods
-------------------------

.. automodule:: thisnotthat.map_cluster_labelling
   :members:
