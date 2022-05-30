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
        self.palette_selector.param.watch(self._palette_change, "value")
        self.color_by_column = pn.widgets.Select(
            name="Color by column", options=["Default"] + list(self.dataframe.columns),
        )
        self.color_by_column.param.watch(self._color_by_change, "value")
        self.hover_text_column = pn.widgets.Select(
            name="Hover text column",
            options=["Default"] + list(self.dataframe.columns),
        )
        self.hover_text_column.param.watch(self._hover_text_change, "value")
        self.marker_size_column = pn.widgets.Select(
            name="Marker size column",
            options=["Default"]
            + list(self.dataframe.select_dtypes(include="number").columns),
        )
        self.marker_size_column.param.watch(self._marker_size_change, "value")
        # self.apply_button = pn.widgets.Button(
        #     name="Apply", button_type="success"
        # )
        self.pane = pn.WidgetBox(
            self.palette_selector,
            self.color_by_column,
            self.marker_size_column,
            self.hover_text_column,
            # self.apply_button,
        )

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    def _palette_change(self, event) -> None:
        if (
            self.palette_selector.value == "Default palette"
            or self.color_by_column.value == "Default"
        ):
            self.color_by_vector = pd.Series([])
            self.color_by_palette = []
            return

        if pd.api.types.is_numeric_dtype(self.dataframe[self.color_by_column.value]):
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
                best_size_index = bisect.bisect_left(palette_sizes, n_colors_required)
                palette = palette_dict[palette_sizes[best_size_index]]
                self.color_by_palette = list(palette)
            elif (
                self.palette_selector.value
                in self.palette_selector.groups["D3 palettes"]
            ):
                palette_dict = bokeh.palettes.d3[self.palette_selector.value]
                palette_sizes = sorted(list(palette_dict.keys()))
                best_size = bisect.bisect_left(palette_sizes, n_colors_required)
                palette = palette_dict[best_size]
                self.color_by_palette = list(palette)
            else:
                raise ValueError("Palette option not in a valid palette group")

    def _color_by_change(self, event) -> None:
        if (
            self.palette_selector.value == "Default palette"
            or self.color_by_column.value == "Default"
        ):
            self.color_by_vector = pd.Series([])
            self.color_by_palette = []
            return

        self._palette_change(None)
        self.color_by_vector = self.dataframe[self.color_by_column.value]

    def _hover_text_change(self, event) -> None:
        if self.hover_text_column.value == "Default":
            self.hover_text = []
        else:
            self.hover_text = self.dataframe[self.hover_text_column.value].to_list()

    def _marker_size_change(self, event) -> None:
        if self.marker_size_column.value == "Default":
            self.marker_size = []
        else:
            self.marker_size = self.dataframe[self.marker_size_column.value].to_list()
