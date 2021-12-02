# thisnotthat -- 2D data explorer and label editor

Consider a dataset composed of a sequence of _records_, each consisting in a loose set of named _fields_. A great process for auditing and understanding such a dataset is to figure out the correlations between records: which are similar and why, as opposed to which are dissimilar, and why. While somebody with a monk's patience and tireless eyeballs can work such correlations out of a spreadsheet view of the data, there exists a more interesting approach:

1. Embed the records in a vector space.
1. Reduce the dimension of the record vectors to 2 and visualize them.
    - Bonus: use hover-like tooling to keep a description of the records handy during visualization.

Some of the natural record correlations will take the form of _clusters_ in the plot display, which can be visually appreciated. A small upgrade to this process involves exploring the similarities quantitatively, using a clustering algorithm. The exploration process then goes:

1. Embed the records in a vector space.
1. _Maybe_ reduce the dimension a little, just enough to compress the data a bit?
1. Run the clustering algorithm to discover the groups of similar records, according to a chosen _distance function_. These group identifiers become the groups' respective _labels_.
1. Reduce the dimension of the record vectors to 2 and visualize them, colored by label.
    - Bonus: use hover-like tooling to keep a description of the records handy during visualization.

In both cases, the plotting of the dataset grants insights into the correlations and the reasons behind them. These insights can drive the scientist towards matching or breaking assumptions or prior knowledge regarding phenomena the data is supposed to describe. Thus, a given cluster can correspond to one of these phenomena. Two or more clusters may describe the same phenomenon, even if the algorithm has marked them as distinct. Alternatively, a correlation between some records might be spurious with respect to a phenomenon of interest, so one may want to split a cluster into two or more. All these cases involve _editing labels_, a process by which one exchanges knowledge between information present in data and prior knowledge one possesses.

`thisnotthat` provides a data viewer and label editor as an ipywidget.

## [Look it up in this Example](Example.ipynb)

This notebook also works as a poor man's user manual.

## Installing

```
pip install git+https://github.com/
```
This is alpha-quality software. Shout at [Benoit Hamelin](https://github.com/hamelin/) to report [issues](https://github.com/TutteInstitute/thisnotthat/issues). Loudly because he's growing old.
