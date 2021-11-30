from contextlib import contextmanager
import itertools as it
from typing import *

import bqplot as bq
import bqplot.interacts as bqi
import ipywidgets as wg
import numpy as np
import traitlets as tr


T = TypeVar("T")


def uniques_sorted(sortables: Iterable[T]) -> List[T]:
    return sorted(list(set(sortables)))


class Labeler(wg.GridBox):
    labels = tr.List()
    names = tr.Dict()

    def __init__(
        self,
        data: np.ndarray,
        labels: Sequence[Hashable],
        notes: Sequence[str] = [],
        names: Mapping[Hashable, Any] = {},
        colors: Mapping[Hashable, str] = {}
    ) -> None:
        super().__init__(
            children=[],
            layout=wg.Layout(
                width="800px",
                height="600px",
                grid_template_rows="90% 10%",
                grid_template_columns="75% 25%",
                grid_template_areas="""
                    "plot legend"
                    "toolbar toolbar"
                """,
                align_content="stretch",
                justify_content="space-between"
            )
        )
        assert data.shape[0] == len(labels)
        assert data.shape[1] == 2
        self.data = data
        self.labels = [*labels]
        self.notes = [*notes]

        self.names = {**names}
        self.colors = {**colors}
        index_unknown = 0
        for label, color in zip(uniques_sorted(self.labels), it.cycle(COLORS)):
            if label not in self.names:
                index_unknown += 1
                self.names[label] = f"Cluster {index_unknown}"
            if label not in self.colors:
                self.colors[label] = color

        self.plot = bq.Figure(
            layout=wg.Layout(grid_area="plot", height="98%"),
            fig_margin={"top": 15, "bottom": 15, "left": 15, "right": 15},
            min_aspect_ratio=1.0,
            max_aspect_ratio=1.0
        )
        self._pz = bqi.PanZoom()
        self._legend = wg.VBox(children=[], layout=wg.Layout(grid_area="legend"))
        self._toolbar = self._make_toolbar()
        self.children = [self.plot, self._legend, self._toolbar]
        self.reset()

    def _make_toolbar(self) -> wg.DOMWidget:
        self._button_reset = wg.Button(
            description="Reset",
            icon="home",
            layout=wg.Layout(width="8em")
        )
        self._button_reset.on_click(self.reset)
        self._toggle_tools = wg.ToggleButtons(
            options={"Explore ": None, "Lasso ": None},
            icons=["hand-point-up", "circle-notch"],
            index=0,
            style=wg.ToggleButtonsStyle(button_width="6em")
        )
        self._button_split = wg.Button(
            description="Split",
            icon="cut",
            disabled=True,
            layout=wg.Layout(width="8em")
        )
        self._dropdown_merge = wg.Dropdown(
            description="Merge to",
            options=[],
            disabled=True,
            layout=wg.Layout(width="auto"),
            style={"description_width": "5em"}
        )

        tr.link((self._toggle_tools, "value"), (self.plot, "interaction"))
        return wg.HBox(
            children=[
                self._button_reset,
                self._toggle_tools,
                self._button_split,
                self._dropdown_merge
            ],
            layout=wg.Layout(width="100%", grid_area="toolbar")
        )

    @property
    def labels_named(self) -> Sequence[str]:
        return [self.names[i] for i in self.labels]

    @contextmanager
    def editing(self) -> Iterable["Labeler"]:
        yield self
        self.refresh()

    def reset(self, *_) -> None:
        self._reset_plot()
        self._reset_legend()
        self._reset_toolbar()

    def _reset_plot(self) -> None:
        scale_x, scale_y = bq.LinearScale(), bq.LinearScale()
        axis_x = bq.Axis(scale=scale_x)
        axis_y = bq.Axis(scale=scale_y, orientation="vertical")
        self.plot.axes = [axis_x, axis_y]

        scatters = []
        for label in uniques_sorted(self.labels):
            i_labeled = [i for i, l_ in enumerate(self.labels) if l_ == label]
            data_labeled = self.data[i_labeled]
            if self.notes:
                raise NotImplementedError("Tooltips")
            scatters.append(
                bq.Scatter(
                    name=label,
                    x=data_labeled[:, 0],
                    y=data_labeled[:, 1],
                    colors=[self.colors[label]],
                    scales={"x": scale_x, "y": scale_y},
                    display_names=False
                )
            )
        self.plot.marks = scatters

        self._pz = bqi.PanZoom(scales={"x": [scale_x], "y": [scale_y]})

    def _reset_legend(self) -> None:
        items_legend = []
        for i, label in enumerate(uniques_sorted(self.labels)):
            picker = wg.ColorPicker(
                concise=True,
                value=self.colors[label],
                layout=wg.Layout(width="2em")
            )
            picker.observe(self._update_color(i, label), "value")
            textbox = wg.Text(
                value=self.names[label],
                continuous_update=False,
                layout=wg.Layout(width="10em")
            )
            textbox.observe(self._update_name(label), "value")
            items_legend.append(
                wg.HBox(
                    children=[picker, textbox],
                    layout=wg.Layout(min_height="2.2em")
                )
            )
        self._legend.children = items_legend

    def _reset_toolbar(self) -> None:
        self._toggle_tools.options = {
            "Explore ": self._pz,
            "Lasso ": bqi.LassoSelector()
        }

    def _update_name(self, label: Hashable) -> None:
        def _update(change: Dict):
            name_new = change["new"]
            label_merge = None
            for label_current, name_current in self.names.items():
                if label_current != label and name_current == name_new:
                    label_merge = label_current
                    break

            self.names[label] = change['new']
            if label_merge is not None:
                del self.names[label_merge]
                self.colors[label] = self.colors[label_merge]
                del self.colors[label_merge]
                for i in range(len(self.labels)):
                    if self.labels[i] == label_merge:
                        self.labels[i] = label
                self.refresh()

        return _update

    def _update_color(self, i: int, label: Hashable) -> None:
        def _update(change: Dict):
            self.colors[label] = change['new']
            self.plot.marks[i].colors = [change['new']]
        return _update


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