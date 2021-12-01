import itertools as it
from typing import *

import bqplot as bq
import bqplot.interacts as bqi
import ipywidgets as wg
import numpy as np
import traitlets as tr


__all__ = ["Labeler"]


Table = Mapping[Hashable, Sequence[float]]
Label = Hashable
Labels = Sequence[Hashable]


def uniques_sorted(sortables: Iterable[Hashable]) -> List[str]:
    return sorted(list(set(sortables)), key=str)


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
        data: Table,
        labels: Labels,
        hover: Sequence = [],
        colors: Mapping[Label, str] = {},
        width: str = "800px",
        height: str = "600px"
    ) -> None:
        super().__init__(
            children=[],
            layout=wg.Layout(
                width=width,
                height=height,
                grid_template_rows="90% 10%",
                grid_template_columns="73% 27%",
                grid_template_areas="""
                    "plot legend"
                    "toolbar toolbar"
                """,
                align_content="stretch",
                justify_content="space-between"
            )
        )

        data_ = np.array(data)
        self._x, self._y = data[:, 0], data[:, 1]

        assert len(self._x) == len(labels)
        labels_fixed = [
            "Unknown" if ll is np.nan or ll is None else ll
            for ll in labels
        ]
        labels_unique = uniques_sorted(labels_fixed)
        self.num_labels = len(labels_unique)
        label_to_int = {ll: i for i, ll in enumerate(labels_unique)}
        self.labels = np.array([label_to_int[ll] for ll in labels_fixed])
        self.names = [str(ll) for ll in labels_unique]

        colors_: List[str] = [c for _, c in zip(range(len(self.names)), COLORS)]
        for i, label in enumerate(labels_unique):
            if label in colors:
                colors_[i] = colors[label]

        self.plot, pan_zoom, lasso = self._make_elements_plot(colors_, hover)
        self._pz = bqi.PanZoom()
        self._legend = self._make_legend()
        self._toolbar = self._make_toolbar(pan_zoom, lasso)
        self.children = [self.plot, self._legend, self._toolbar]

    def _make_elements_plot(
        self,
        colors: List[str],
        hover: Sequence
    ) -> Tuple[bq.Figure, bqi.Interaction, bqi.Interaction]:
        scale_x, scale_y = bq.LinearScale(), bq.LinearScale()
        scale_colors = bq.ColorScale(
            max=len(colors) - 1,
            colors=colors
        )
        axis_x = bq.Axis(scale=scale_x)
        axis_y = bq.Axis(scale=scale_y, orientation="vertical")

        def mark_selection(change: Dict) -> None:
            scatter = change["owner"]
            if len(self.selected) > 0:
                scatter.unselected_style = {"opacity": 0.1}
            else:
                scatter.unselected_style = {}

        scatter = bq.Scatter(
            x=self._x,
            y=self._y,
            color=self.labels,
            scales={"x": scale_x, "y": scale_y, "color": scale_colors},
            display_names=False
        )
        scatter.observe(mark_selection, names="selected")

        def on_click(sc: bq.Scatter, event: Dict) -> None:
            index = event["data"]["index"]
            if index in self.selected:
                self.deselect([index])
            else:
                self.select([index])

        scatter.on_element_click(on_click)

        if len(hover) > 0:
            assert len(hover) == len(self._x)
            scatter.names = [str(h) for h in hover]
            scatter.tooltip = bq.Tooltip(fields=["name"], show_labels=False)

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

    @property
    def _labels_unique(self) -> Iterator[int]:
        return uniques_sorted(self.labels)

    def _items_legend(self) -> List[wg.DOMWidget]:
        items_legend = []
        for label in self._labels_unique:
            picker = wg.ColorPicker(
                concise=True,
                value=self.colors[label],
                layout=wg.Layout(width="2em")
            )
            picker.observe(self._update_color(label), "value")
            selector = wg.Button(
                icon="check-square",
                layout=wg.Layout(width="3em")
            )
            selector.on_click(self._select_cluster(label))
            textbox = wg.Text(
                value=self.names[label],
                continuous_update=False,
                layout=wg.Layout(width="10em")
            )
            textbox.observe(self._update_name(label), "value")
            items_legend.append(
                wg.HBox(
                    children=[picker, selector, textbox],
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
            options=[("Pick ", None), ("Pan/Zoom ", pan_zoom), ("Lasso ", lasso)],
            icons=["hand-point-up", "arrows", "circle-notch"],
            index=0,
            style=wg.ToggleButtonsStyle(button_width="6em")
        )

        def set_disabled(widget: wg.Widget) -> Callable[[Dict], None]:
            def _set(change: Dict):
                widget.disabled = (len(self.selected) == 0)
            return _set

        self._button_split = wg.Button(
            description="Split",
            icon="flag",
            disabled=True,
            layout=wg.Layout(width="8em")
        )
        self._scatter.observe(set_disabled(self._button_split), names="selected")
        self._dropdown_merge = wg.Dropdown(
            description="Merge to",
            options=[],
            disabled=True,
            layout=wg.Layout(width="auto"),
            style={"description_width": "5em"}
        )
        self._set_options_merge()
        self._dropdown_merge.observe(self._merge_to, names="value")
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

    def select(self, indexes: Sequence[int]) -> None:
        self.selected = np.hstack([self.selected, np.array(indexes)])

    def deselect(self, indexes: Sequence[int]) -> None:
        self.selected = np.array([i for i in self.selected if i not in indexes])

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
    def selected(self) -> np.ndarray:
        if self._scatter.selected is None:
            return np.array([])
        return self._scatter.selected

    @selected.setter
    def selected(self, selected: Optional[np.ndarray]) -> None:
        self._scatter.selected = selected

    @property
    def labels_named(self) -> Sequence[str]:
        return [self.names[i] for i in self.labels]

    def reset(self, *_) -> None:
        if len(self.selected) > 0:
            self.selected = None
            interaction = self.plot.interaction
            self.plot.interaction = None
            self.plot.interaction = interaction
        else:
            for dim in ["x", "y"]:
                self._scatter.scales[dim].min = None
                self._scatter.scales[dim].max = None

    def _set_options_merge(self) -> None:
        self._dropdown_merge.options = [("(choose...)", -1)] + [
            (self.names[label], label) for label in self._labels_unique
        ]

    def _update_name(self, label: int) -> Callable[[Dict], None]:
        def _update(change: Dict):
            self.names[label] = change["new"]

        return _update

    def _update_color(self, label: int) -> Callable[[Dict], None]:
        def _update(change: Dict):
            colors = [*self.plot.marks[0].scales["color"].colors]
            colors[label] = change["new"]
            self.plot.marks[0].scales["color"].colors = colors
        return _update

    def _select_cluster(self, label: int) -> Callable[[Dict], None]:
        def _click(*_) -> None:
            indexes_cluster = np.array(
                [i for i, ll in enumerate(self.labels) if ll == label]
            )
            if np.all(np.isin(indexes_cluster, self.selected)):
                self.selected = np.setdiff1d(self.selected, indexes_cluster)
            else:
                self.selected = np.union1d(self.selected, indexes_cluster)
        return _click

    def _merge_to(self, change: Dict) -> None:
        label_target = change["new"]
        if label_target >= 0:
            num_labels = len(uniques_sorted(self.labels))
            assert label_target < len(self.names)
            for i in self.selected.astype(int):
                self.labels[i] = label_target

            self._scatter.color = None
            self._scatter.color = self.labels
            self._scale_color.max = self.num_labels - 1
            self._legend.children = self._items_legend()
            self._set_options_merge()
            self._dropdown_merge.value = -1


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
