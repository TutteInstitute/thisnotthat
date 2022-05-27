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


class BokehPlotPane(pn.viewable.Viewer, pn.reactive.Reactive):
    labels = param.Series(doc="Labels")
    label_color_palette = param.List([], item_type=str, doc="Color palette")
    label_color_factors = param.List([], item_type=str, doc="Color palette")
    selected = param.List([], doc="Indices of selected samples")
    color_by_vector = param.Series(doc="Color by")
    color_by_palette = param.List(
        list(bokeh.palettes.Viridis256), item_type=str, doc="Color by palette"
    )
    marker_size = param.List([], item_type=float, doc="Marker size")
    hover_text = param.List([], item_type=str, doc="Hover text")

    def _update_selected(self, attr, old, new) -> None:
        self.selected = self.data_source.selected.indices

    def __init__(
        self,
        data: npt.ArrayLike,
        labels: Iterable[str],
        hover_text: Optional[Iterable[str]] = None,
        size: Optional[Iterable[float]] = None,
        *,
        label_color_mapping: Optional[Dict[str, str]] = None,
        palette: Sequence[str] = bokeh.palettes.Turbo256,
        width: int = 600,
        height: int = 600,
        max_point_size: Optional[float] = None,
        min_point_size: Optional[float] = None,
        fill_alpha: float = 0.75,
        line_color: str = "white",
        line_width: float = 0.25,
        hover_fill_color: str = "red",
        hover_line_color: str = "black",
        hover_line_width: float = 2,
        selection_fill_alpha: float = 1.0,
        nonselection_fill_alpha: float = 0.1,
        nonselection_fill_color: str = "gray",
        background_fill_color: str = "#ffffff",
        border_fill_color: str = "whitesmoke",
        toolbar_location: str = "above",
        title: Optional[str] = None,
        title_location: str = "above",
        show_legend: bool = True,
        legend_location: str = "outside",
        name: str = "Plot",
    ):
        super().__init__(name=name)
        self.data_source = bokeh.models.ColumnDataSource(
            {
                "x": np.asarray(data).T[0],
                "y": np.asarray(data).T[1],
                "label": labels,
                "hover_text": hover_text if hover_text is not None else labels,
                "size": size if size is not None else np.full(data.shape[0], 0.1),
                "apparent_size": size
                if size is not None
                else np.full(data.shape[0], 0.1),
            }
        )
        self.data_source.selected.on_change("indices", self._update_selected)
        if label_color_mapping is not None:
            factors = []
            colors = []
            for label, color in label_color_mapping.items():
                factors.append(label)
                colors.append(color)
            self._label_colormap = bokeh.transform.factor_cmap(
                "label", palette=colors, factors=factors
            )
            self.color_mapping = self._label_colormap["transform"]
            self.color_mapping.palette = self.color_mapping.palette + [
                palette[x] for x in _palette_index(len(palette))
            ]
        else:
            self._label_colormap = bokeh.transform.factor_cmap(
                "label", palette=palette, factors=list(set(labels))
            )
            self.color_mapping = self._label_colormap["transform"]
            self.color_mapping.palette = [
                self.color_mapping.palette[x] for x in _palette_index(256)
            ]

        self.plot = bokeh.plotting.figure(
            width=width,
            height=height,
            output_backend="webgl",
            background_fill_color=background_fill_color,
            border_fill_color=border_fill_color,
            toolbar_location=toolbar_location,
            tools="pan,wheel_zoom,lasso_select,save,reset,help",
            title=title,
            title_location=title_location,
        )
        if show_legend:
            if legend_location == "outside":
                self._legend = bokeh.models.Legend(location="center", label_width=150)
                self.plot.add_layout(self._legend, "right")

            self._color_by_legend_source = bokeh.models.ColumnDataSource(
                {
                    "x": np.zeros(16),
                    "y": np.zeros(16),
                    "color_by": np.linspace(0, 1, 16),
                }
            )
            self._color_by_renderer = self.plot.circle(
                source=self._color_by_legend_source, visible=False,
            )

            self.points = self.plot.circle(
                source=self.data_source,
                radius="apparent_size",
                fill_color=self._label_colormap,
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

            if legend_location != "outside":
                self.plot.legend.location = legend_location
        else:
            self.points = self.plot.circle(
                source=self.data_source,
                radius="apparent_size",
                fill_color=self._label_colormap,
                fill_alpha=fill_alpha,
                line_color=line_color,
                line_width=line_width,
                hover_fill_color=hover_fill_color,
                hover_line_color=hover_line_color,
                hover_line_width=hover_line_width,
                selection_fill_alpha=selection_fill_alpha,
                nonselection_fill_alpha=nonselection_fill_alpha,
                nonselection_fill_color=nonselection_fill_color,
            )

        if max_point_size is not None or min_point_size is not None:
            if max_point_size is None:
                max_point_size = 1.0e6
            elif min_point_size is None:
                min_point_size = 0.0
            circle_resize_callback = bokeh.models.callbacks.CustomJS(
                args=dict(source=self.data_source),
                code="""
            const scale = cb_obj.end - cb_obj.start;
            const size = source.data["size"];
            const apparent_size = source.data["apparent_size"];
            for (var i = 0; i < size.length; i++) {
                if ((size[i] / scale) > %f) {
                    apparent_size[i] = %f * scale;
                } else if ((size[i] / scale) < %f) {
                    apparent_size[i] = %f * scale;
                } else {
                    apparent_size[i] = size[i];
                }
            }
            source.change.emit();
            """
                % (max_point_size, max_point_size, min_point_size, min_point_size),
            )
            self.plot.x_range.js_on_change("start", circle_resize_callback)

        self.plot.add_tools(
            bokeh.models.HoverTool(
                tooltips=[("text", "@hover_text")], renderers=[self.points]
            )
        )
        self.plot.xgrid.grid_line_color = None
        self.plot.ygrid.grid_line_color = None
        self.plot.xaxis.bounds = (0, 0)
        self.plot.yaxis.bounds = (0, 0)
        self.pane = pn.pane.Bokeh(self.plot)

        self.show_legend = show_legend
        self.color_by_vector = pd.Series([], dtype=object)
        self.labels = pd.Series(labels)
        self.label_color_palette = list(self.color_mapping.palette)
        self.label_color_factors = list(self.color_mapping.factors)

    # Reactive requires this to make the model auto-display as requires
    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("label_color_palette", watch=True)
    def _update_palette(self) -> None:
        self.color_mapping.palette = self.label_color_palette
        pn.io.push_notebook(self.pane)

    @param.depends("label_color_factors", watch=True)
    def _update_factors(self) -> None:
        self.color_mapping.factors = self.label_color_factors
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

    @param.depends("color_by_vector", watch=True)
    def _update_color_by_vectors(self) -> None:
        if len(self.color_by_vector) == 0:
            self.points.glyph.fill_color = self._label_colormap
            self.plot.legend.items = [
                bokeh.models.LegendItem(
                    label={"field": "label"}, renderers=[self.points]
                )
            ]
        elif pd.api.types.is_numeric_dtype(self.color_by_vector):
            self.data_source.data["color_by"] = self.color_by_vector
            colormap = bokeh.transform.linear_cmap(
                "color_by",
                self.color_by_palette,
                self.color_by_vector.min(),
                self.color_by_vector.max(),
            )
            self.points.glyph.fill_color = colormap
            self._color_by_renderer.glyph.color = colormap
            self._color_by_legend_source.data["color_by"] = [
                np.round(x, decimals=2)
                for x in np.linspace(
                    self.color_by_vector.min(), self.color_by_vector.max(), 16,
                )
            ]
            self.plot.legend.items = [
                bokeh.models.LegendItem(
                    label={"field": "color_by"}, renderers=[self._color_by_renderer]
                )
            ]
        else:
            self.data_source.data["color_by"] = self.color_by_vector
            colormap = bokeh.transform.factor_cmap(
                "color_by", self.color_by_palette, list(self.color_by_vector.unique())
            )
            self.points.glyph.fill_color = colormap
            self.plot.legend.visible = self.show_legend

        pn.io.push_notebook(self.pane)

    @param.depends("color_by_palette", watch=True)
    def _update_color_by_palette(self) -> None:
        if len(self.color_by_palette) == 0:
            self.points.glyph.fill_color = self._label_colormap
            self.plot.legend.items = [
                bokeh.models.LegendItem(
                    label={"field": "label"}, renderers=[self.points]
                )
            ]

        elif pd.api.types.is_numeric_dtype(self.color_by_vector):
            colormap = bokeh.transform.linear_cmap(
                "color_by",
                self.color_by_palette,
                self.color_by_vector.min(),
                self.color_by_vector.max(),
            )
            self.points.glyph.fill_color = colormap
            self._color_by_renderer.glyph.color = colormap
            self._color_by_legend_source.data["color_by"] = [
                np.round(x, decimals=2)
                for x in np.linspace(
                    self.color_by_vector.min(), self.color_by_vector.max(), 16,
                )
            ]
            self.plot.legend.items = [
                bokeh.models.LegendItem(
                    label={"field": "color_by"}, renderers=[self._color_by_renderer]
                )
            ]
        else:
            colormap = bokeh.transform.factor_cmap(
                "color_by", self.color_by_palette, list(self.color_by_vector.unique())
            )
            self.points.glyph.fill_color = colormap
            self.plot.legend.items = [
                bokeh.models.LegendItem(
                    label={"field": "color_by"}, renderers=[self.points]
                )
            ]

        pn.io.push_notebook(self.pane)
