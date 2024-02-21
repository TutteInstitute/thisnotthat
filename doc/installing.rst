Installation
============

Requirements
------------

This Not That (TNT) is built on top of (and as an extension of) the `Panel library`_. Thus
at a minimum you will need to have panel installed to make use of TNT. Panel itself is built
atop Bokeh, and Bokeh is the best default choice for plotting data maps within TNT. To make
effective use of TNT you will also need `numpy`_ and `pandas`_ installed. Given that you will
be exploring data maps you will likely want to install `umap-learn`_ anyway, which will bring in
numpy, `scikit-learn`_, `numba`_ and `pynndescent`_ (which will all be useful with TNT anyway). For
color handling `matplotlib`_ is needed, and finally for cluster label annotations `hdbscan`_ is required.

In summary you will need:

* panel
  + bokeh
  + param
* umap-learn
  + numpy >= 1.22
  + scikit-learn
  + numba
  + pynndescent
* pandas
* matplotlib
* hdbscan
* cmocean
* colorcet
* glasbey
* cmocean
* vectorizers

Recommended Extras
------------------

While the requirements are enough to get you working with TNT, there are some optional extras
that can prove to be very beneficial. If you are interested in text annotations of clusters in
data maps then both `apricot-select`_ for submodular selection, and `networkx`_ for graph handling
can be very beneficial.

* apricot-select
* networkx

Installing
------------------
To install the package from PyPI:

.. code:: bash

    pip install thisnotthat

To install the package from source:

.. code:: bash

    pip install git+https://github.com/TutteInstitute/thisnotthat

We hope to have a version of TNT on conda-forge sometime soon.

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
