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
        self.dataframe = raw_dataframe

        self.palette_selector = pn.widgets.Select(
            name="Color Palette",
            options=bokeh.palettes.__palettes__,
        )
        self.column_selected = pn.widgets.Select(
            name="Color by column",
            options=self.dataframe.columns
        )
        self.apply_button = pn.widgets.Button(
            name="Apply", button_type="success"
        )
        self.pane = pn.WidgetBox(
            self.palette_selector,
            self.column_selected,
            self.apply_button,
        )

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)
