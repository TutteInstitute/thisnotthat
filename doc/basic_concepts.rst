Basic Concepts
==============

This Not That (TNT) is built on top of the `Panel library`_, and ultimately familiarity with
Panel and its concepts will be necessary to get the most out of TNT. On the other hand
you can get a lot done without having to know much beyond some core pieces. This guide will
outline the core concepts it will be beneficial to be aware of, and the basics of how to use
them. If you want to dig deeper the excellent `Panel documentation`_ is highly recommended.

To get the most out of TNT there are four main concepts that will matter. The first is the
concept of Widgets and Panes; the second is interactive Params from the `Param library`_;
the third is linking params between Widgets and Panes; and finally managing to put together
a layout of multiple Panes and Widgets in a Panel display. Let's discuss these different
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

Pane and Widgets from Panel are themselves built atop Params from the `Param library`_. A ``Param``
provides  a way to encapsulate a parameter along with all the things that depend on that parameter
in a simple interface. In practice this means that Panes and Widgets have attributes that are Params,
and changes to those attributes are automatically propagated anything that depends upon those Params,
including changes to interactive views, and other Panes or Widgets that have declared a dependency on
that Param.

This is a little easier to think about in a practical example. All TNT PlotPanes have an attribute
``selected`` that is a Param. If you access that attribute in Python it will list the indices of the
points currently selected in the plot -- changing the selection in the plot will change the value of
the attribute. Conversely if you set the value of the ``selected`` attribute in Python that change will
propagate through to the plot. In a sense Params are "live" interactive values that can be updated
by other elements, and that can push updates to dependencies if they are changed.

There is obviously a lot more to Params than just this, but this should be enough to get started
working with TNT. It is definitely worth reading the `Param documentation`_ to learn more.

Linking
-------

If Panes and Widgets are visual elements of a data map app, and Params are their exposed
interactive values, linking is the glue that binds the Params together. The `Panel library`_
provides a rich range of ways to link params together, with varying levels of fine-grained
control, described in detail in their `linking docs`_. To make good use of the TNT we only
really need to worry about the simplest version: the ``link`` method.

Panel's Pane and Widget objects have a method ``link`` that allows the user to specify Params
that should be linked between the objects. The link can be made bi-directional (so changes
from either side get propagated across). Since much of TNT is built around the use of
a PlotPane with a scatterplot of a data map, TNT Panes and Widgets go one step further
and *also* have a convenience method ``link_to_plot`` which links the relevant Params
of the Pane or widget in question the the Params of a specified PlotPane.

Once Params have been linked the given Panes and Widgets will pass information back and forth
via the Python kernel, so interactions in one Pane or Widget will effect the other, and
(if bidirectional) vice versa. Thus after constructing the Panes and widgets you want and
linking them together, you have the basis for an interactive app.

Panel Layouts
-------------

Having all the pieces of a data map app, and having them appropriately linked together, the
last step is to actually arrange them in an app pr interface. For this there are Panel
Layouts -- these are Panel classes that allow you to arrange Panes and Widgets into a
cohesive whole. They are, for the most part, very simple to use. The most basic are
`Row`_, `Column`_ and `GridSpec`_ which pack Widgets and Panes into a row, column, or
grid arrangement. You can, of course, nest these so you might have:

.. code:: python3

    pn.Row(
        plot_pane,
        pn.Column(
            search_widget,
            plot_control_widget,
        )
    )

or any other manner of arrangement. More advanced `layout options`_ are also available
in Panel, and you can even make use of `templates`_ for more polished apps.

.. _Panel library: https://panel.holoviz.org/
.. _Panel documentation: https://panel.holoviz.org/user_guide/Overview.html
.. _Param library: https://param.holoviz.org/
.. _Param documentation: https://param.holoviz.org/getting_started.html
.. _linking docs: https://panel.holoviz.org/user_guide/Links.html
.. _layout options: https://panel.holoviz.org/reference/index.html#layouts
.. _templates: https://panel.holoviz.org/user_guide/Templates.html
.. _Row: https://panel.holoviz.org/reference/layouts/Row.html
.. _Column: https://panel.holoviz.org/reference/layouts/Column.html
.. _GridSpec: https://panel.holoviz.org/reference/layouts/GridSpec.html