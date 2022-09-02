Basic Concepts
==============

This Not That (TNT) is built on top of the `Panel library`_, and ultimately familiarity with
Panel and its concepts will be necessary to get the most out of TNT. On the other hand
you can get a lot done without having to know much beyond some core pieces. This guide will
outline the core concepts it will be beneficial to be aware of, and the basics of how to use
them. If you want to dig deeper the excellent `Panel documentation`_ is highly recommended.

To get the most out of TNT there are four main concepts that will matter. The first is the
concept of widgets and panes; the second is interactive params from the `Param library`_;
the third is linking params between widgets and panes; and finally managing to put together
a layout of multiple panes and widgets in a panel display. Let's discuss these different
concepts one by one, borrowing from the `Panel documentation`_ where necessary.

Widgets and Panes
-----------------

According to Panel a ``Pane`` is a renderable view that is reactive to parameter changes, and a
``Widget`` is a control component that allows users to provide input to an app or dashboard. Within
TNT this distinction gets a little blurrier, since many Panes and interactive and allow users to
provide input, but the core idea remains the same: a Pane provides a view; a Widget provides controls
to interact with views.

Since TNT is designed to make working with data maps easy the primary type
of Pane is a PlotPane that can provide a scatterplot view of a datamap,
and various ways to enrich and interact with the data through that view. Other Panes provide other
ways to view data, including an instance viewer and a tabular data viewer.

TNT provides several widgets useful for interacting with the plots, including a search widget
a plot control widget for altering plotting parameters, and a label editor for tagging and
labelling data via interactive selections in the plot.

All of these Panes and Widgets are, in turn, built up from standard Panel Panes and Widgets.
The goal of TNT is to provide a standard library of pre-built and easy to use Panes and Widgets
for building data map based applications.

Params
------

Linking
-------

Panel Layouts
-------------


.. _Panel library: https://panel.holoviz.org/
.. _Panel documentation: https://panel.holoviz.org/user_guide/Overview.html
.. _Param library: https://param.holoviz.org/