Deploying TNT Based Web Apps
============================

Because TNT is built on top of, and works with, the `Panel library`_ it is relatively
easy to deploy TNT based web apps allowing other users to use TNT to explore a
given data map. For more detailed discussion of using panel to deploy apps please
see the panel documentation on `deploying panel apps`_ and on `server deployment`_.
The goal of this guide is to get you up an running with a basic app quickly.

Using Notebooks
---------------

As discussed in :doc:`using_notebooks` it is easy to use TNT interactively within a notebook.
This can be a useful way of prototyping what you would like to build. Once you have a final
layout of panes that you would like to export you can add a cell that calls the ``servable``
method on the layout. For example we might have:

.. code:: python3

    map_plot = tnt.BokehPlotPane(
        map_data,
        labels=label_vector,
        marker_size=numeric_vector,
        hover_text=text_vector,
    )
    data_view = tnt.DataPane(source_data)
    data_view.link_to_plot(map_plot)
    search_pane = tnt.SearchWidget(source_data)
    search_pane.link_to_plot(map_plot)
    app = pn.Column(pn.Row(map_plot, search_pane), data_view)
    app.servable()

This will provide an interactive version of ``app`` appearing inline in the notebook,
but also marks the output of that cell as what should be deployed if the notebook is
run via the ``panel serve`` command line utility. Thus if we have saved this notebook as
``tnt_app_demo.ipynb`` we can run

.. code:: bash

    panel serve tnt_app_demo.ipynb

on the command line of the server we wish to deploy from, and as simply as that we
have a web application running on the server.

Using Scripts
-------------

It is also possible to collect more complex operations into a script that can then
be deployed. This works similarly to the notebook option: adding ``.servable()`` to
one or more layouts of panes in the script will make them discoverable for ``panel serve``.
Thus assuming we have ``servable`` content that gets run by the script ``tnt_app_script.py``
we can run

.. code:: bash

    panel serve tnt_app_script.py

on the command line of the server we wish to deploy from, and as simply as that we
have a web application running on the server.


.. _Panel library: https://panel.holoviz.org/
.. _deploying panel apps: https://panel.holoviz.org/user_guide/Deploy_and_Export.html
.. _server deployment: https://panel.holoviz.org/user_guide/Server_Deployment.html
