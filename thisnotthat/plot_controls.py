import panel as pn
import param
import pandas as pd
import numpy as np
import numpy.typing as npt
import bokeh.palettes
import bisect

from typing import *


class PlotControlPane(pn.reactive.Reactive):

    color_by_vector = param.Series(doc="Color by")
    color_by_palette = param.List([], item_type=str, doc="Color by palette")
    marker_size = param.List([], item_type=float, doc="Marker size")
    hover_text = param.List([], item_type=str, doc="Hover text")

    def __init__(self, raw_dataframe: pd.DataFrame, *, name="Plot Controls"):
        super().__init__(name=name)
        self.dataframe = raw_dataframe

        self.palette_selector = pn.widgets.Select(
            name="Color Palette",
            groups={
                "Default": ["Default palette"],
                "ColorBrewer palettes": list(bokeh.palettes.brewer.keys()),
                "D3 palettes": list(bokeh.palettes.d3.keys()),
                "Smooth palettes": [
                    "Viridis",
                    "Cividis",
                    "Greys",
                    "Inferno",
                    "Magma",
                    "Plasma",
                    "Turbo",
                ],
            },
        )
        self.palette_selector.param.watch(
            self._options_changed, "value", onlychanged=True
        )
        self.color_by_column = pn.widgets.Select(
            name="Color by column", options=["Default"] + list(self.dataframe.columns),
        )
        self.color_by_column.param.watch(
            self._options_changed, "value", onlychanged=True
        )
        self.hover_text_column = pn.widgets.Select(
            name="Hover text column",
            options=["Default"] + list(self.dataframe.columns),
        )
        self.hover_text_column.param.watch(
            self._options_changed, "value", onlychanged=True
        )
        self.marker_size_column = pn.widgets.Select(
            name="Marker size column",
            options=["Default"]
            + list(self.dataframe.select_dtypes(include="number").columns),
        )
        self.marker_size_column.param.watch(
            self._options_changed, "value", onlychanged=True
        )
        self.apply_changes = pn.widgets.Button(
            name="Apply Changes", button_type="default",
        )
        self.apply_changes.on_click(self._apply_changes)
        self.pane = pn.WidgetBox(
            self.palette_selector,
            self.color_by_column,
            self.marker_size_column,
            self.hover_text_column,
            self.apply_changes,
        )

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    def _options_changed(self, event) -> None:
        self.apply_changes.button_type = "success"

    def _change_palette(self):
        if pd.api.types.is_numeric_dtype(self.color_by_vector):
            # Continuous scale required
            if (
                self.palette_selector.value
                in self.palette_selector.groups["Smooth palettes"]
            ):
                palette_name = self.palette_selector.value + "256"
                self.color_by_palette = list(getattr(bokeh.palettes, palette_name))
            elif (
                self.palette_selector.value
                in self.palette_selector.groups["ColorBrewer palettes"]
            ):
                palette_dict = bokeh.palettes.brewer[self.palette_selector.value]
                max_palette_size = max(palette_dict.keys())
                palette = palette_dict[max_palette_size]
                self.color_by_palette = list(palette)
            elif (
                self.palette_selector.value
                in self.palette_selector.groups["D3 palettes"]
            ):
                palette_dict = bokeh.palettes.d3[self.palette_selector.value]
                max_palette_size = max(palette_dict.keys())
                palette = palette_dict[max_palette_size]
                self.color_by_palette = list(palette)
            else:
                raise ValueError("Palette option not in a valid palette group")
        else:
            # Discrete scale required
            n_colors_required = self.dataframe[self.color_by_column.value].nunique()
            if n_colors_required > 256:
                n_colors_required = 256

            if (
                self.palette_selector.value
                in self.palette_selector.groups["Smooth palettes"]
            ):
                palette_name = self.palette_selector.value + "256"
                raw_palette = getattr(bokeh.palettes, palette_name)
                palette = bokeh.palettes.linear_palette(raw_palette, n_colors_required)
                self.color_by_palette = list(palette)
            elif (
                self.palette_selector.value
                in self.palette_selector.groups["ColorBrewer palettes"]
            ):
                palette_dict = bokeh.palettes.brewer[self.palette_selector.value]
                palette_sizes = sorted(list(palette_dict.keys()))

                if n_colors_required <= max(palette_sizes):
                    best_size_index = bisect.bisect_left(
                        palette_sizes, n_colors_required
                    )
                    palette = palette_dict[palette_sizes[best_size_index]]
                else:
                    max_size = max(palette_sizes)
                    n_copies = (n_colors_required // max_size) + 1
                    palette = palette_dict[max_size] * n_copies

                self.color_by_palette = list(palette)
            elif (
                self.palette_selector.value
                in self.palette_selector.groups["D3 palettes"]
            ):
                palette_dict = bokeh.palettes.d3[self.palette_selector.value]
                palette_sizes = sorted(list(palette_dict.keys()))

                if n_colors_required <= max(palette_sizes):
                    best_size_index = bisect.bisect_left(
                        palette_sizes, n_colors_required
                    )
                    palette = palette_dict[palette_sizes[best_size_index]]
                else:
                    max_size = max(palette_sizes)
                    n_copies = (n_colors_required // max_size) + 1
                    palette = palette_dict[max_size] * n_copies

                self.color_by_palette = list(palette)
            else:
                raise ValueError("Palette option not in a valid palette group")

    def _apply_changes(self, event) -> None:
        if self.color_by_column.value == "Default":
            self.color_by_vector = pd.Series([])
        else:
            self.color_by_vector = self.dataframe[self.color_by_column.value]

        if self.palette_selector.value == "Default palette":
            self.color_by_palette = []
        else:
            self._change_palette()

        if self.hover_text_column.value == "Default":
            self.hover_text = []
        else:
            self.hover_text = (
                self.dataframe[self.hover_text_column.value].map(str).to_list()
            )

        if self.marker_size_column.value == "Default":
            self.marker_size = []
        else:
            self.marker_size = self.dataframe[self.marker_size_column.value].to_list()

        self.apply_changes.button_type = "default"
