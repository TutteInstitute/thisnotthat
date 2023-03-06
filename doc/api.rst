TNT API Guide
=============

TNT Provides a number of different Pane and Widget classes that can be combined and
linked in various ways. The core Panes are the Plot Panes, with various
associated other panes and widgets.

.. currentmodule:: thisnotthat

Plot Panes
----------

.. autoclass:: BokehPlotPane
   :members: add_cluster_labels

.. autoclass:: DeckglPlotPane
   :members:

Data Panes
----------

.. autoclass:: InformationPane
   :members: link_to_plot

.. autoclass:: DataPane
   :members: link_to_plot

.. autoclass:: SimpleDataPane
   :members: link_to_plot

Summary Panes
-------------

.. autoclass:: PlotSummaryPane
   :members: link_to_plot

.. autoclass:: DataSummaryPane
   :members: link_to_plot


Search and Edit Widgets
-----------------------

.. autoclass:: SearchWidget
   :members: link_to_plot

.. autofunction:: SimpleSearchWidget
   :annotation:

.. autoclass:: LabelEditorWidget
   :members: link_to_plot

.. autoclass:: PlotControlWidget
   :members: link_to_plot


Finally TNT provides tools for annotating plots with cluster labels.
There are various methods for achieving this.

Cluster Labelling Methods
-------------------------

.. automodule:: thisnotthat.map_cluster_labelling
   :members:


