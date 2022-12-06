.. -*- mode: rst -*-

.. image:: doc/tnt_logo.png
  :width: 600
  :alt: TNT Logo

===================
This Not That (TNT)
===================

This Not That is a suite of tools for working with, exploring, and interacting with "data maps". A data map is a two
(or three) dimensional representation of higher dimensional vector space data, usually produced by UMAP, t_SNE, or
another manifold learning technique. The goal of This Not That is to make it quick and easy to visualize, enrich and
interact with data maps. This not that also aims to make it easy to build and deploy simple web-apps on top of data
maps to let other users gain access to rich interactive data maps as a means to explore data sets.

---------------------------
This Not That for Labelling
---------------------------

This Not That makes it easy to quickly build an interactive data labeller directly in your notebook, allowing
you to quickly tag data for further analysis, extract interesting samples, or do an initial round of bulk
labelling.

-----------------------------
This Not That for Exploration
-----------------------------

This Not That makes it easy to get an interactive plot of a data map, but also allows you to quickly connect
other tools to it, including powerful search utilities, a rich data instance viewer, data tables tied to
in-plot selections, and tools for quickly adjusting plot attributes from lists of extra variables.

This Not That also provides a variety of automated solutions for adding textual annotation layers to data maps,
providing easier navigation and exploration.

-------------------------------
This Not That for Data Map Apps
-------------------------------

Because This Not That is built on top of the `Panel library`_ it is trivial to deploy TNT based solutions as
interactive web applications.

----------
Installing
----------

This Not That is built on top of Panel and Bokeh, so you will need these installed. In all you will need:

* panel
* bokeh
* numpy >= 1.22
* pandas
* scikit-learn
* matplotlib
* umap-learn
* hdbscan
* glasbey
* cmocean
* vectorizers

We also highly recommend installing:

* networkx
* apricot-select

Currently you can pip install directly from this repository:

.. code:: bash

    pip install git+https://github.com/TutteInstitute/thisnotthat

------------
Contributing
------------

Contributions are more than welcome! There are lots of opportunities
for potential projects, so please get in touch if you would like to
help out. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute. We are also keen to hear user stories and suggestions for new Pane's to add to
to our catalog. We also welcome contributions to get other plot libraries integrated and
working with TNT, including plotly, VTK/PyVista, and others, so if you have expertise with these please consider
contributing.

To contribute please `fork the project <https://github.com/TutteInstitute/thisnotthat/issues#fork-destination-box>`_ make your changes and
submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.

-------
License
-------

The This Not That package is 2-clause BSD licensed.


.. _Panel library: https://panel.holoviz.org/



