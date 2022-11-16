Building Panels
===============

This guide provides a brief walkthrough of the steps involved in building a simple
data map exploration app with This Not That (TNT). This is not a comprehensive
guide (see the details of the different components and the `Panel documentation`_
for that), but it should be enough to get you started putting together your own
data map Panels and apps. For working examples see the Tutorials.

Choosing Components
-------------------

The first step in putting together a Panel is to decide what components it should have.
TNT provides a number of components (documented in their own pages), and different
collections of them are suitable for different kinds of tasks.

First and foremost you will want a PlotPane of some kind to display the data map. TNT
supports a number of plotting backends which, in turn, provide different ranges of
functionality and performance. Choosing the right PlotPane type for your task will
go a long way toward making your Panel effective. Currently TNT supports the following
PlotPane types:

* :py:class:`~thisnotthat.BokehPlotPane`
* :py:class:`~thisnotthat.DeckglPlotPane`

and we hope that more can be added in the future. The API documentation for each PlotPane
type will provide the most details on the capabilities of the PlotPanes, but here is a useful
rule of thumb guide:

* BokehPlotPane
   Supports the greatest range of interactive features and has a good range of options
   for customising the visual styling of the plots. This is a good choice if you want
   cluster labelling, easy lasso-tool selection, and a wide variety of interactive
   tool options.
* DeckglPlotPane
   Has the best interactive performance, especially for larger datasets, and supports a
   reasonable range of options for customising the visual styling of plots. The selection
   tools are more spartan (with a basic brush-like selection), and cluster labelling is
   not yet supported. This is a good choice for very large datasets, and where smooth
   zooming and panning for exploration is more important than selection and other interactions.

With the data map plot component itself taken care of, the next concern is what other components
to use to supplement your data exploration. Currently the major supported extra components are:

* :py:class:`~thisnotthat.DataPane`
* :py:class:`~thisnotthat.InformationPane`
* :py:class:`~thisnotthat.SearchWidget`
* :py:class:`~thisnotthat.LabelEditorWidget`
* :py:class:`~thisnotthat.PlotControlWidget`

Rather than going through these components individually, let's look at a few different
kinds of potential use cases for data maps, and discuss which components will be useful
in those kinds of cases.

#. **Just looking at a data map**

   In this simplest case a PlotPane, potentially making use of colour, marker size and hover text
   may be sufficient for your needs. If you have a lot of options for what to colour by, how
   to set the marker size, or what to use as hover text, then a ``PlotControlWidget`` could
   be useful to make it easy to swap between different options quickly, and contrast them.

#. **Exploring text-based or image-based datasets**

   An ``InformationPane`` to make it easier to display formatted text or image of a selected item
   is probably going to be very useful. For text-based datatsets the ``SearchWidget`` may also come
   in handy. If you have a lot of extra information to go along with the text or images then
   a ``PlotContorlPane`` might make some sense so you can adjust plot attributes based on
   those other factors.

#. **Exploring data that has an associated dataframe**

   This is a great case for the ``DataPane``, allowing a user to view the associated dataframe and
   connect it to selections in the plot -- making it easy to pivot back and forth between dataframe
   and map representations. The ``SearchWidget`` may also be valuable here, particularly in the
   *Pandas query* mode, allowing complex dataframe queries to be represented in the plot. If the
   associated dataframe is not too wide then a ``PlotControlWiget`` using dataframe data may also be
   a useful addition.

#. **Bulk labelling, or tagging for triage**

   Sometimes you want to be able to do some initial bulk labelling of a dataset, or tag regions of
   interest for later triage, or simply annotate a dataset with extra information for future users.
   Data maps provide a powerful way to do this, allowing quick selection of clusters, or interesting
   phenomena. To enable this in TNT you will want the ``LabelEditorWidget``. In this use case it is
   also often helpful to have the ``SearchWidget``. If you have associated dataframe data then
   the ``DataPane`` and ``PlotControlWidget`` may be useful extra additions.

Naturally there are more use cases, but this should be enough to give you a sense of when different
components are likely to be useful, and which components will combine well together. So, with your use
case in mind, select the set of components that will work the best for what you have in mind.

Linking and Layout
------------------

Once you have components chosen and created the next step is to link together the relevant
Params. Since all TNT Panes and Widgets derive from the `Panel library`_ they all support
the ``link`` method from Panel. In general the easiest way to link the components together
is to instead use the ``link_to_plot`` method, and link any non-PlotPane to your PlotPane.
This is the easiest approach for two reasons: first, you don't need to know which Params
from any given component can and should be linked with the PlotPane, the method takes care
of that for you; second, because it creates bidirectional links this ensures the PlotPane
acts as the central hub for changes made to Params in any of the other components, interlinking
everything correctly. On the other hand, if you have specific linking needs, or want to link
to other Panel components, you can use the ``link`` method to specify everything more explicitly.

Having bound everything together into a linked interactive package, the last required step
is to provide a layout for the components. For this we defer to the `Panel library`_ which
provides a wealth of `layout options`_. We recommend the `Row`_, `Column`_ or `Gridspec`_ as
good options to keep things simple. For more complex layouts involving many components
`Tabs`_,  `Card`_ or `Accordion`_ can be very useful. Lastly Panel `Templates`_ provide
good simple templates for an entire app or dashboard if that is what is desired.

When putting together a layout of TNT components some suggestions include:

* The ``SearchWidget``, ``PlotControlWidget`` and ``LabelEditorWidget`` often work well on the
   right hand side of a PlotPane. If you have all three using ``Tabs`` or ``Accordion`` to pack
   them together can be useful.
* The ``DataPane`` is often best placed below the PlotPane, since it is often quite wide.
* Most TNT Panes and Widgets support specifying a size with ``width`` and ``height`` and it
   can be beneficial to help ensure the various components line up nicely.
* The ``BokehPlotPane`` supports using a legend which, by default, is outside the plot to the
   right. If you are planning to add other components beside the PlotPane it can be best to
   either disable the legend (``show_legend=False``), or specify a location for the legend
   other than ``"outside"`` (e.g. ``legend_location="top_right"``).

Extending with Panel
--------------------

TNT provides the basic building blocks for data map based Panels, and you can go a long way
with just TNT components. On the other hand the `Panel library`_ has a very rich `gallery
of panes and widgets`_ that can offer all manner of interactive functionality. Since TNT
panes and widgets are really just custom Panel panes and widgets you can mix and match
as needed, building whatever other interactive functionality you want for your Panel out
of the components from the `Panel library`_. Perhaps you want to build your own custom search
tools; perhaps there are custom interactions specific to your use case or domain -- you can
build and add whatever you need.

Deploying an App
----------------


.. _Panel documentation: https://panel.holoviz.org/user_guide/Overview.html
.. _Panel library: https://panel.holoviz.org/
.. _gallery of panes and widgets: https://panel.holoviz.org/reference/index.html
.. _layout options: https://panel.holoviz.org/reference/index.html#layouts
.. _Row: https://panel.holoviz.org/reference/layouts/Row.html
.. _Column: https://panel.holoviz.org/reference/layouts/GridSpec.html
.. _Gridspec: https://panel.holoviz.org/reference/layouts/GridSpec.html
.. _Tabs: https://panel.holoviz.org/reference/layouts/Tabs.html
.. _Card: https://panel.holoviz.org/reference/layouts/Card.html
.. _Accordion: https://panel.holoviz.org/reference/layouts/Accordion.html
.. _Templates: https://panel.holoviz.org/reference/index.html#templates