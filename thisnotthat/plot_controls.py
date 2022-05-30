import panel as pn
import param
import pandas as pd
import numpy as np
import numpy.typing as npt
import bokeh.palettes

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
                    "Gray",
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
        if pd.api.types.is_numeric_dtype(self.dataframe[self.color_by_column.value]):
            # Continuous scale required
            pass
        else:
            # Discrete scale required
            pass

    def _color_by_change(self, event) -> None:
        pass

    def _hover_text_change(self, event) -> None:
        pass

    def _marker_size_change(self, event) -> None:
        pass
