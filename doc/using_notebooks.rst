Using TNT in a Notebook
=======================

This Not That (TNT) is designed to be able to be used interactively in a notebook
to make data exploration fast an easy. The goal of this quick-start guide is
to quickly get you up and running with basic TNT tools in a notebook as
quickly as possible. To ensure that this all works seamlessly please be sure
to follow the `Panel installation instructions`_ to ensure you have panel and
it's extensions installed and working with your desired jupyter environment, be
it JupyterLab or classic notebooks.

Basic Plots
-----------

The most basic thing that TNT can provide is interactive plot of data maps. Let
us suppose that you have some data ``data_map`` as a numpy array produced by UMAP
(or another manifold learning approach such as t-SNE, TriMAP, PyMDE or PacMAP) and
you want to visualize it with TNT.

First you will need both TNT and Panel imported.

.. code:: python3

    import thisnotthat as tnt
    import panel as pn

Next you will need to instantiate the panel extension (which ensures more complex
interactions with the plot will work):

.. code:: python3

    pn.extension()

To create a plot "pane" you can pass the :py:class:`~thisnotthat.BokehPlotPane` the data map
array.

.. code:: python3

    map_plot = tnt.BokehPlotPane(data_map)

And to display the plot you can either access its :py:attr:`~thisnotthat.BokehPlotPane.pane` attribute
or place it within a panel layout, such as a ``Row`` or ``Tabs``:

.. code:: python3

    pn.Row(map_plot)

This will now display a plot inline in the notebook that supports zoom, pan, and selection. In this form
the indices of the currently selected points can be access as ``map_plot.selected``, with later evaluations
of that attribute updating according to what is currently selected in the plot. You can also set the
value of the ``selected`` attribute in another cell to select points directly -- e.g.
``map_plot.selected = [1, 75, 124]``.

Colour, Marker Size, Hover Text
-------------------------------

As basic plot is probably not as interesting as it could be. It would be good to overlay some extra
information in the plot itself, such as the colour and size of points, and hover-text that can be shown
as a tooltip when hovering over points in the plot. This is all supported in TNT for all of the ``PlotPane``
types. To adjust these you can use keyword arguments to the ``PlotPane``. For example suppose we have
categorical data ``label_vector``, continuous data ``numeric_vector`` and string data ``text_vector``
such that there is an entry in each list or vector for each row in ``map_data``. Then we can use:

.. code:: python3

    map_plot = tnt.BokehPlotPane(
        map_data,
        labels=label_vector,
        marker_size=numeric_vector,
        hover_text=text_vector,
    )

The ``PlotPane``s also support changing these interactively via the param attributes
* ``labels``;
* ``marker_size``;
* ``hover_text``.

Since these are `Params`_, setting them from another cell will automatically update the
plot itself. See the API documentation for other param attributes that can be used.

TNT as a Bulk Labeller
----------------------

A relatively common use case for TNT is as a quick and dirty bulk-labelling approach. Labelling this
way will not usually result in perfect labels, but it can get a lot of broad brush labelling done
very quickly that can then be fine tuned with more careful labelling after the fact. Alternatively
it can simply be used to tag clusters of points for later triage or more detailed inspection or
labelling (or as irrelevant, or to be discarded). The key to getting this done is the
:py:class:`~thisnotthat.LabelEditorWidget` which supports editing an initial labelling, and
adding new labels based on the current selection in the plot.

To create a :py:class:`~thisnotthat.LabelEditorWidget` you need to pass it an initial set of labels --
a list of vector of strings, one for each point in the map. This can simply be a sequence of strings
``"unlabelled"`` or similar if you wish. You can even use the ``labels`` attribute of the ``PlotPane``
if you wish. Thus assuming we have a plot pane as above we could use

.. code:: python3

    label_editor = tnt.LabelEditorWidget(map_plot.labels)

You can then display the plot and the label editor inline in the notebook with

.. code:: python3

    pn.Row(map_plot, label_editor)

Unfortunately this will note (yet) let you edit the plot labelling via the label editor -- we need
to link up the param attributes from the plot and the label editor. The easiest way to do this is
to use the :py:method:`~thisnotthat.LabelEditorWidget.link_to_plot` method:

.. code:: python3

    label_editor.link_to_plot(map_plot)

Now selecting points in the plot will enable the "New Label" button in the label editor, and
selecting new colors (via the color swatch) or renaming labels (by editor the label name in the
label editor) will introduce corresponding changes in the plot.

Once you are done with your labelling you can extract the label information either as a
label vector from the label editor as ``label_editor.labels``, or extract the full dataframe
of information from the plot as ``map_plot.dataframe``, which will contain the current label
information.

Search and Data Views
---------------------

To make exploration easier, and provide richer access to underlying data, it can be beneficial to have
views of the source data that are linked to the plot, and the ability to search the data and see the
results show up on the plot. TNT provides utilities to do this.

Suppose we have some source data which we can format in a dataframe called ``source_data``. We can have a table view
of the dataframe that is linked, via selections, to the plot. To add a table view under the plot
we might use:

.. code:: python3

    data_view = tnt.DataPane(source_data)
    data_view.link_to_plot(map_plot)
    pn.Column(map_plot, data_view)

If, instead we want to have a richer representation of individual selected points (perhaps your
source data contains complex text for example), you can an information panel which can use a
markdown template and fields from the dataframe. For example we might use something like:

.. code:: python3

    info_view = tnt.InformationPane(
        source_data,
        """# {title_text}

    {body_text}
        """
    )
    info_view.link_to_plot(map_plot)
    pn.Row(map_plot, info_view)

where ``title_text`` and ``body_text`` are column names in the ``source_data`` dataframe. The
markdown can, of course, be more complicated, and format any number of fields from the dataframe.

Finally it can be very useful to be able to search for data from the ``source_data`` representation
and see that search reflected in the data map. For that we have the :py:class:`~thisnotthat.SearchWidget`
which enables this. A simple example might look like:

.. code:: python3

    search_pane = tnt.SearchWidget(source_data)
    search_pane.link_to_plot(map_plot)
    pn.Row(map_plot, search_pane)


.. _Panel installation instructions: https://panel.holoviz.org/getting_started/index.html#jupyterlab-and-classic-notebook
.. _Params: