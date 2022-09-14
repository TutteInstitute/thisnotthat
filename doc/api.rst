TNT API Guide
=============

TNT Provides a number of different Pane classes that can be combined and
linked in various ways. The core Panes are the Plot Panes, with various
associated other panes.

.. currentmodule:: {{ thisnotthat }}

Plot Panes
----------

.. autoclass:: BokehPlotPane

   .. automethod:: __init__

   :members:

.. autoclass:: DeckglPlotPane
   :members:

Data Panes
----------

.. autoclass:: InformationPane
   :members:

.. autoclass:: DataPane
   :members:

Search and Edit Widgets
-----------------------

.. autoclass:: SearchWidget
   :members:

.. autoclass:: LabelEditorWidget
   :members:

.. autoclass:: PlotControlWidget
   :members:


Finally TNT provides tools for annotating plots with cluster labels.
There are various methods for achieving this.

Cluster Labelling Methods
-------------------------

.. automodule:: map_cluster_labelling
   :members:
