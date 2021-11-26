from contextlib import contextmanager
from typing import *

import bqplot as bq
import ipywidgets as wg
import numpy as np
import traitlets as tr


T = TypeVar("T")


def uniques_sorted(sortables: Iterable[T]) -> List[T]:
    return sorted(list(set(sortables)))


class Labeler(wg.HBox):
    labels = tr.List()
    names = tr.Dict()

    def __init__(
        self,
        data: np.ndarray,
        labels: Sequence[Hashable],
        notes: Sequence[str] = [],
        names: Mapping[Hashable, Any] = {},
    ) -> None:
        super().__init__(children=[], layout=wg.Layout(width="800px", height="800px", resize="both"))
        assert data.shape[0] == len(labels)
        assert data.shape[1] == 2
        self.data = data
        self.labels = [*labels]
        self.notes = [*notes]

        self.names = {**names}
        index_unknown = 0
        for label in uniques_sorted(self.labels):
            if label not in self.names:
                index_unknown += 1
                self.names[label] = f"Cluster {index_unknown}"

        self.refresh()

    def refresh(self) -> None:
        scale_x = bq.LinearScale()
        scale_y = bq.LinearScale()
        scatters = []
        for label, color in zip(uniques_sorted(self.labels), COLORS):
            i_labeled = [i for i, l_ in enumerate(self.labels) if l_ == label]
            data_labeled = self.data[i_labeled]
            if self.notes:
                raise NotImplementedError("Tooltips")
            scatters.append(
                bq.Scatter(
                    x=data_labeled[:, 0],
                    y=data_labeled[:, 1],
                    colors=[color],
                    scales={"x": scale_x, "y": scale_y},
                    names=[self.names[self.labels[i]] for i in i_labeled],
                    display_names=False
                )
            )

        self.figure = bq.Figure(
            marks=scatters,
            axes=[
                bq.Axis(scale=scale_x),
                bq.Axis(scale=scale_y, orientation="vertical")
            ],
            layout=wg.Layout(width="95%", height="95%"),
            fig_margin={"top": 15, "bottom": 15, "left": 15, "right": 15},
            min_aspect_ratio=1.0,
            max_aspect_ratio=1.0
        )
        self.children = [self.figure]


    @contextmanager
    def editing(self) -> Iterable["Labeler"]:
        yield self
        self.refresh()


COLORS = """\
steelblue
chartreuse
darkgreen
slategray
magenta
midnightblue
salmon
azure
darkgoldenrod
lightcyan
silver
thistle
lightpink
red
darkcyan
orchid
springgreen
lawngreen
palevioletred
mediumturquoise
mediumblue
darkmagenta
wheat
sandybrown
blueviolet
burlywood
lightgray
darkred
maroon
floralwhite
deeppink
mediumvioletred
darkorchid
darkslateblue
pink
forestgreen
oldlace
darksalmon
slateblue
bisque
darkgray
lightsalmon
goldenrod
limegreen
seashell
chocolate
darkturquoise
mediumslateblue
cyan
tomato
paleturquoise
olivedrab
peru
lightslategray
papayawhip
rosybrown
lightgoldenrodyellow
saddlebrown
greenyellow
yellow
purple
beige
snow
powderblue
aqua
lime
palegoldenrod
firebrick
cadetblue
tan
ivory
dimgray
hotpink
deepskyblue
mediumpurple
coral
blanchedalmond
honeydew
turquoise
green
lightskyblue
skyblue
indianred
olive
orangered
seagreen
darkolivegreen
darkorange
plum
slategray
darkviolet
lemonchiffon
white
violet
fuchsia
darkseagreen
sienna
peachpuff
navy
lavender
palegreen
lightyellow
gray
mediumaquamarine
whitesmoke
mistyrose
indigo
cornsilk
darkblue
ghostwhite
mediumorchid
teal
antiquewhite
gold
linen
aquamarine
orange
moccasin
dodgerblue
aliceblue
darkslategray
crimson
darkkhaki
lightblue
brown
navajowhite
mediumseagreen
yellowgreen
khaki
mediumspringgreen
cornflowerblue
lightsteelblue
gainsboro
lightgreen
blue
lightcoral
lavenderblush
lightseagreen
royalblue
mintcream\
""".split("\n")