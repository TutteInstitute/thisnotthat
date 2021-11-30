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


class Counter:

    def __init__(self) -> None:
        self.n = 0

    def __int__(self) -> int:
        self.n += 1
        return self.n

    def __repr__(self) -> str:
        return f"Counter({self.n})"

    def __str__(self) -> int:
        return str(int(self))


class Labeler(wg.GridBox):

    def __init__(
        self,
        data: np.ndarray,
        labels: Sequence[int],
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
        label_to_int: Mapping[Hashable, int] = {
            ll: i
            for i, ll in enumerate(uniques_sorted(labels))
        }
        self.labels = np.array([label_to_int[ll] for ll in labels])
        self._label_next: int = max(list(label_to_int.values())) + 1
        self.notes = [*notes]

        self.names: List[str] = []
        index_unknown = Counter()
        colors_: List[str] = [c for _, c in zip(range(self._label_next), COLORS)]
        for label, i in label_to_int.items():
            self.names.append(names.get(label, "") or f"Cluster {index_unknown}")
            if label in colors:
                colors_[i] = colors[label]

        self.plot, pan_zoom, lasso = self._make_elements_plot(colors_)
        self._pz = bqi.PanZoom()
        self._legend = self._make_legend()
        self._toolbar = self._make_toolbar(pan_zoom, lasso)
        self.children = [self.plot, self._legend, self._toolbar]

    def _make_elements_plot(
        self,
        colors: List[str]
    ) -> Tuple[bq.Figure, bqi.Interaction, bqi.Interaction]:
        scale_x, scale_y = bq.LinearScale(), bq.LinearScale()
        scale_colors = bq.ColorScale(
            colors=colors
        )
        axis_x = bq.Axis(scale=scale_x)
        axis_y = bq.Axis(scale=scale_y, orientation="vertical")

        scatter = bq.Scatter(
            x=self.data[:, 0],
            y=self.data[:, 1],
            color=self.labels,
            scales={"x": scale_x, "y": scale_y, "color": scale_colors},
            display_names=False
        )

        return (
            bq.Figure(
                axes=[axis_x, axis_y],
                marks=[scatter],
                layout=wg.Layout(grid_area="plot", height="98%"),
                fig_margin={"top": 15, "bottom": 15, "left": 15, "right": 15},
                min_aspect_ratio=1.0,
                max_aspect_ratio=1.0
            ),
            bqi.PanZoom(scales={"x": [scale_x], "y": [scale_y]}),
            bqi.LassoSelector(marks=[scatter])
        )

    def _items_legend(self) -> List[wg.DOMWidget]:
        items_legend = []
        for label, color in enumerate(self.colors):
            picker = wg.ColorPicker(
                concise=True,
                value=color,
                layout=wg.Layout(width="2em")
            )
            picker.observe(self._update_color(label), "value")
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
        return items_legend

    def _make_legend(self) -> wg.DOMWidget:
        return wg.VBox(
            children=self._items_legend(),
            layout=wg.Layout(grid_area="legend")
        )

    def _make_toolbar(
        self,
        pan_zoom: bqi.Interaction,
        lasso: bqi.Interaction
    ) -> wg.DOMWidget:
        self._button_reset = wg.Button(
            description="Reset",
            icon="home",
            layout=wg.Layout(width="8em")
        )
        self._button_reset.on_click(self.reset)
        self._toggle_tools = wg.ToggleButtons(
            options={"Pan/Zoom ": pan_zoom, "Pick ": None, "Lasso ": lasso},
            icons=["arrows", "crosshairs", "circle-notch"],
            index=0,
            style=wg.ToggleButtonsStyle(button_width="6em")
        )
        self._button_split = wg.Button(
            description="Split",
            icon="flag",
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
        self._set_options_merge()
        self._scatter.observe(set_disabled(self._dropdown_merge), names="selected")

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
    def _scatter(self) -> bq.Scatter:
        return self.plot.marks[0]

    @property
    def _scale_color(self) -> bq.ColorScale:
        return self._scatter.scales["color"]

    @property
    def colors(self) -> List[str]:
        return self._scale_color.colors

    @colors.setter
    def colors(self, colors: List[str]) -> None:
        self._scale_color.colors = colors

    @property
    def labels_named(self) -> Sequence[str]:
        return [self.names[i] for i in self.labels]

    def reset(self, *_) -> None:
        self._reset_plot()
        self._reset_legend()
        self._reset_toolbar()

    def merge_to(self, label_from: int, label_to: int) -> None:
        del self.names[label_from]
        colors = [*self.colors]
        del colors[label_from]
        self.colors = colors

        for i in range(len(self.labels)):
            if self.labels[i] == label_from:
                self.labels[i] = label_to
        self.plot.marks[0].color = []
        self.plot.marks[0].color = self.labels

        self._legend.children = self._items_legend()
        self._set_options_merge()

    def _set_options_merge(self) -> None:
        self._dropdown_merge.options = self.names

    def _update_name(self, label: int) -> None:
        def _update(change: Dict):
            name_new = change["new"]
            self.names[label] = name_new
            for label_current, name_current in enumerate(self.names):
                if label_current != label and name_current == name_new:
                    self.merge_to(label, label_current)
                    break

            self._set_options_merge()

        return _update

    def _update_color(self, label: int) -> None:
        def _update(change: Dict):
            colors = [*self.plot.marks[0].scales["color"].colors]
            colors[label] = change["new"]
            self.plot.marks[0].scales["color"].colors = colors
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