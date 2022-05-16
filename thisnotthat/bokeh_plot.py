import panel as pn
import param
import bokeh.plotting
import bokeh.models
import bokeh.transform
import bokeh.palettes
import numpy as np
import numpy.typing as npt
import pandas as pd

from .utils import _palette_index

from typing import *


class BokehPlotPane(pn.reactive.Reactive):
    labels = param.Series(doc="Labels")
    color_palette = param.List([], item_type=str, doc="Color palette")
    color_factors = param.List([], item_type=str, doc="Color palette")
    selected = param.List([], doc="Indices of selected samples")

    def _update_selected(self, attr, old, new) -> None:
        self.selected = self.data_source.selected.indices

    def __init__(
        self,
        data: npt.ArrayLike,
        labels: Iterable[str],
        annotation: Optional[Iterable[str]] = None,
        size: Optional[Iterable[float]] = None,
        *,
        label_color_mapping: Optional[Dict[str, str]] = None,
        palette: Sequence[str] = bokeh.palettes.Turbo256,
        width: int = 600,
        height: int = 600,
        fill_alpha=0.75,
        line_color="white",
        line_width=0.25,
        hover_fill_color="red",
        hover_line_color="black",
        hover_line_width=2,
        selection_fill_alpha=1.0,
        nonselection_fill_alpha=0.1,
        nonselection_fill_color="gray",
    ):
        super().__init__()
        self.data_source = bokeh.models.ColumnDataSource(
            {
                "x": np.asarray(data).T[0],
                "y": np.asarray(data).T[1],
                "label": labels,
                "annotation": annotation if annotation is not None else labels,
                "size": size if size is not None else np.full(data.shape[0], 0.1)
            }
        )
        self.data_source.selected.on_change("indices", self._update_selected)
        if label_color_mapping is not None:
            factors = []
            colors = []
            for label, color in label_color_mapping.items():
                factors.append(label)
                colors.append(color)
            self._factor_cmap = bokeh.transform.factor_cmap(
                "label", palette=colors, factors=factors
            )
            self.color_mapping = self._factor_cmap["transform"]
            self.color_mapping.palette = self.color_mapping.palette + [
                palette[x] for x in _palette_index(len(palette))
            ]
        else:
            self._factor_cmap = bokeh.transform.factor_cmap(
                "label", palette=palette, factors=list(set(labels))
            )
            self.color_mapping = self._factor_cmap["transform"]
            self.color_mapping.palette = [
                self.color_mapping.palette[x] for x in _palette_index(256)
            ]

        self.plot = bokeh.plotting.figure(
            width=width,
            height=height,
            output_backend="webgl",
            border_fill_color="whitesmoke",
        )
        points = self.plot.circle(
            source=self.data_source,
            radius="size",
            fill_color=self._factor_cmap,
            fill_alpha=fill_alpha,
            line_color=line_color,
            line_width=line_width,
            hover_fill_color=hover_fill_color,
            hover_line_color=hover_line_color,
            hover_line_width=hover_line_width,
            selection_fill_alpha=selection_fill_alpha,
            nonselection_fill_alpha=nonselection_fill_alpha,
            nonselection_fill_color=nonselection_fill_color,
            legend_field="label",
        )
        self.plot.add_tools(
            bokeh.models.HoverTool(
                tooltips=[("text", "@annotation")], renderers=[points]
            )
        )
        self.plot.add_tools(bokeh.models.LassoSelectTool())
        self.plot.xgrid.grid_line_color = None
        self.plot.ygrid.grid_line_color = None
        self.plot.xaxis.bounds = (0, 0)
        self.plot.yaxis.bounds = (0, 0)
        self.pane = pn.pane.Bokeh(self.plot)

        self.labels = pd.Series(labels)
        self.color_palette = list(self.color_mapping.palette)
        self.color_factors = list(self.color_mapping.factors)

    # Reactive requires this to make the model auto-display as requires
    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("color_palette", watch=True)
    def _update_palette(self) -> None:
        self.color_mapping.palette = self.color_palette
        pn.io.push_notebook(self.pane)

    @param.depends("color_factors", watch=True)
    def _update_factors(self) -> None:
        self.color_mapping.factors = self.color_factors
        pn.io.push_notebook(self.pane)

    @param.depends("labels", watch=True)
    def _update_labels(self) -> None:
        self.data_source.data["label"] = self.labels  # self.dataframe["label"]
        # We auto-update the factors from elsewhere (? from legend yes, but not from table edits)
        #         self.factors = list(self.color_mapping.factors) + [
        #             x for x in self.labels.unique() if x not in self.color_mapping.factors
        #         ]
        pn.io.push_notebook(self.pane)

    @param.depends("selected", watch=True)
    def _update_selection(self) -> None:
        self.data_source.selected.indices = self.selected
        pn.io.push_notebook(self.pane)
