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


def add_text_layer(
    plot_figure,
    text_dataframe,
    text_size,
    layer_type="middle",
    *,
    angle=0,
    text_color="#444444",
    text_font={"value": "helvetica"},
    text_font_style="normal",
    text_line_height=0.9,
    text_alpha=1.0,
):
    label_data_source = bokeh.models.ColumnDataSource(text_dataframe)
    labels = bokeh.models.Text(
        text_font_size=str(text_size) + "pt",
        text_baseline="bottom",
        text_align="center",
        angle=angle,
        text_color=text_color,
        text_font=text_font,
        text_font_style=text_font_style,
        text_line_height=text_line_height,
        text_alpha=text_alpha,
    )
    text_resize_callback = bokeh.models.callbacks.CustomJS(
        args=dict(labels=labels),
        code="""
    const scale = cb_obj.end - cb_obj.start;
    const text_size = (%f / scale);
    if (text_size > 48 && %s) {
        var alpha = (64 - text_size) / 16.0;

    } else if (text_size < 14 && %s) {
        var alpha = (text_size - 2.0) / 12.0;

    } else {
        var alpha = 1.0;
    }
    if (alpha > 0) {
        labels.text_alpha = alpha;
    } else {
        labels.text_alpha = 0.0;
    }
    labels.text_font_size = text_size + "pt";
    labels.change.emit();
    """
        % (
            text_size,
            "false" if layer_type == "bottom" else "true",
            "false" if layer_type == "top" else "true",
        ),
    )
    plot_figure.add_glyph(label_data_source, labels)
    plot_figure.x_range.js_on_change("start", text_resize_callback)


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
        labels: Optional[Iterable[str]] = None,
        hover_text: Optional[Iterable[str]] = None,
        marker_size: Optional[Iterable[float]] = None,
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
        if labels is None:
            labels = ["unlabelled"] * len(data)

        self.data_source = bokeh.models.ColumnDataSource(
            {
                "x": np.asarray(data).T[0],
                "y": np.asarray(data).T[1],
                "label": labels,
                "hover_text": hover_text if hover_text is not None else labels,
                "size": marker_size
                if marker_size is not None
                else np.full(data.shape[0], 0.1),
                "apparent_size": marker_size
                if marker_size is not None
                else np.full(data.shape[0], 0.1),
            }
        )
        self.data_source.selected.on_change("indices", self._update_selected)

        self._base_marker_size = pd.Series(
            marker_size if marker_size is not None else np.full(data.shape[0], 0.1)
        )
        self._base_hover_text = hover_text if hover_text is not None else labels
        self._base_hover_is_labels = hover_text is None
        self._base_palette = list(palette)

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
            else:
                self._legend = bokeh.models.Legend(
                    location=legend_location, label_width=150
                )
                self.plot.add_layout(self._legend, "right")

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

            self._color_by_legend_source = bokeh.models.ColumnDataSource(
                {"x": np.zeros(8), "y": np.zeros(8), "color_by": np.linspace(0, 1, 8),}
            )
            self._color_by_renderer = self.plot.square(
                source=self._color_by_legend_source, line_width=0, visible=False,
            )
            self._color_by_legend = bokeh.models.Legend(
                items=[
                    bokeh.models.LegendItem(
                        label={"field": "color_by"}, renderers=[self._color_by_renderer]
                    )
                ],
                location=legend_location if legend_location != "outside" else "center",
                label_width=150,
            )
            self.plot.add_layout(
                self._color_by_legend,
                "right" if legend_location == "outside" else "center",
            )
            self._color_by_legend.visible = False

            # if legend_location != "outside":
            #     self.plot.legend.location = legend_location
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

        self.max_point_size = max_point_size
        self.min_point_size = min_point_size
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
        self.labels = pd.Series(labels).copy()  # .reset_index(drop=True)
        self.label_color_palette = list(self.color_mapping.palette)
        self.label_color_factors = list(self.color_mapping.factors)
        self.color_by_palette = list(self.color_mapping.palette)

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
        if self._base_hover_is_labels:
            if len(self.hover_text) == 0:
                self.data_source.data["hover_text"] = self.labels
            self._base_hover_text = self.labels
        # We auto-update the factors from elsewhere (? from legend yes, but not from table edits)
        #         self.factors = list(self.color_mapping.factors) + [
        #             x for x in self.labels.unique() if x not in self.color_mapping.factors
        #         ]
        pn.io.push_notebook(self.pane)

    @param.depends("selected", watch=True)
    def _update_selection(self) -> None:
        self.data_source.selected.indices = self.selected
        pn.io.push_notebook(self.pane)

    @param.depends("marker_size", watch=True)
    def _update_marker_size(self):

        if (
            self.max_point_size is None and self.min_point_size is None
        ) or self.plot.x_range.start is None:

            def _map_apparent_size(x):
                return x

        else:
            if self.max_point_size is None:
                max_point_size = 1.0e6
            else:
                max_point_size = self.max_point_size

            if self.min_point_size is None:
                min_point_size = 0.0
            else:
                min_point_size = self.min_point_size

            scale = self.plot.x_range.end - self.plot.x_range.start

            def _map_apparent_size(x):
                if (x / scale) > max_point_size:
                    return max_point_size * scale
                elif (x / scale) < min_point_size:
                    return min_point_size * scale
                else:
                    return x

        if len(self.marker_size) == 0:
            self.data_source.data["size"] = self._base_marker_size
            self.data_source.data["apparent_size"] = self._base_marker_size.map(
                _map_apparent_size
            )
        elif len(self.marker_size) == 1:
            size_vector = pd.Series(
                np.full(len(self.data_source.data["size"]), self.marker_size[0])
            )
            self.data_source.data["size"] = size_vector
            self.data_source.data["apparent_size"] = size_vector.map(_map_apparent_size)
        else:
            rescaled_size = pd.Series(self.marker_size)
            rescaled_size = (rescaled_size - rescaled_size.mean()) / rescaled_size.std()
            rescaled_size = 0.05 * (rescaled_size - rescaled_size.min() + 1)
            self.data_source.data["size"] = rescaled_size
            self.data_source.data["apparent_size"] = rescaled_size.map(
                _map_apparent_size
            )

        pn.io.push_notebook(self.pane)

    @param.depends("hover_text", watch=True)
    def _update_hover_text(self):
        if len(self.hover_text) == 0:
            self.data_source.data["hover_text"] = self._base_hover_text
        else:
            self.data_source.data["hover_text"] = [str(x) for x in self.hover_text]

        pn.io.push_notebook(self.pane)

    @param.depends("color_by_palette", "color_by_vector", watch=True)
    def _update_color_by(self) -> None:
        if len(self.color_by_palette) == 0:
            print("choosing base palette")
            palette = self._base_palette
        else:
            print("choosing palette: ", str(self.color_by_palette))
            palette = self.color_by_palette

        if len(self.color_by_vector) == 0:
            print("using default color-mapping")
            self.points.glyph.fill_color = self._label_colormap
            if self.show_legend:
                if self._legend.items[0].label["field"] != "label":
                    self._legend.items[0].label["field"] = "label"
                self._legend.items[0].renderers = [self.points]

        elif pd.api.types.is_numeric_dtype(self.color_by_vector):
            print("color_by_vector is numeric")
            self.data_source.data["color_by"] = self.color_by_vector
            colormap = bokeh.transform.linear_cmap(
                "color_by",
                palette,
                self.color_by_vector.min(),
                self.color_by_vector.max(),
            )
            self.points.glyph.fill_color = colormap
            if self.show_legend:
                self._color_by_legend_source.data["color_by"] = [
                    np.round(x, decimals=2)
                    for x in np.linspace(
                        self.color_by_vector.max(), self.color_by_vector.min(), 8,
                    )
                ]
                print(
                    f"setting  renderer for legend to {self._color_by_renderer} instead of {self.points}"
                )
                self._color_by_renderer.glyph.fill_color = colormap
                self._legend.items[0].label["field"] = "color_by"
                self._legend.items[0].renderers = [self._color_by_renderer]
        else:
            print("treating color_by_vector as categorical")
            self.data_source.data["color_by"] = self.color_by_vector
            colormap = bokeh.transform.factor_cmap(
                "color_by", palette, list(self.color_by_vector.unique())
            )
            self.points.glyph.fill_color = colormap
            if self.show_legend:
                self._legend.items[0].label["field"] = "color_by"
                self._legend.items[0].renderers = [self.points]

        pn.io.push_notebook(self.pane)

    # def _update_color_by_palette(self) -> None:
    #     if len(self.color_by_vector) == 0:
    #         # # HACK: Not sure why this is needed, but things don't update without it?
    #         # self.data_source.data["color_by"] = ["nil"] * len(
    #         #     self.data_source.data["label"]
    #         # )
    #         # colormap = bokeh.transform.factor_cmap(
    #         #     "color_by", self.color_by_palette, ["nil"],
    #         # )
    #         # self.points.glyph.fill_color = colormap
    #         # # END HACK
    #         self.points.glyph.fill_color = self._label_colormap
    #         if self.show_legend:
    #             if self.plot.legend.items[0].label["field"] != "label":
    #                 self.plot.legend.items[0].label["field"] = "label"
    #             self.plot.legend.items[0].renderers = [self.points]
    #
    #     elif pd.api.types.is_numeric_dtype(self.color_by_vector):
    #         colormap = bokeh.transform.linear_cmap(
    #             "color_by",
    #             self.color_by_palette,
    #             self.color_by_vector.min(),
    #             self.color_by_vector.max(),
    #         )
    #         self.points.glyph.fill_color = colormap
    #         if self.show_legend:
    #             self._color_by_legend_source.data["color_by"] = [
    #                 np.round(x, decimals=2)
    #                 for x in np.linspace(
    #                     self.color_by_vector.max(), self.color_by_vector.min(), 16,
    #                 )
    #             ]
    #             self._color_by_renderer.glyph.fill_color = colormap
    #             self.plot.legend.items[0].label["field"] = "color_by"
    #             self.plot.legend.items[0].renderers = [self._color_by_renderer]
    #     else:
    #         colormap = bokeh.transform.factor_cmap(
    #             "color_by", self.color_by_palette, list(self.color_by_vector.unique())
    #         )
    #         self.points.glyph.fill_color = colormap
    #         if self.show_legend:
    #             self.plot.legend.items[0].label["field"] = "color_by"
    #             self.plot.legend.items[0].renderers = [self.points]
    #
    #     pn.io.push_notebook(self.pane)

    def add_cluster_labels(
        self,
        cluster_labelling,
        *,
        angle=0,
        text_size_scale=12,
        text_color="#444444",
        text_font={"value": "helvetica"},
        text_font_style="normal",
        text_line_height=0.9,
        text_alpha=1.0,
    ):
        for i, (label_locations, label_strings) in enumerate(
            zip(cluster_labelling.location_layers, cluster_labelling.labels_for_display)
        ):
            cluster_label_layer = pd.DataFrame(
                {
                    "x": label_locations.T[0],
                    "y": label_locations.T[1],
                    "text": label_strings,
                }
            )

            if i == 0:
                layer_type = "bottom"
            elif i == len(cluster_labelling.location_layers) - 1:
                layer_type = "top"
            else:
                layer_type = "middle"

            add_text_layer(
                self.plot,
                cluster_label_layer,
                text_size_scale * 2 ** i,
                layer_type=layer_type,
                angle=angle,
                text_color=text_color,
                text_font=text_font,
                text_font_style=text_font_style,
                text_line_height=text_line_height,
                text_alpha=text_alpha,
            )

    @property
    def dataframe(self):
        result = pd.DataFrame(self.data_source.data)
        if "color_by" in result:
            result = result.drop(columns=["apparent_size", "color_by"])
        else:
            result = result.drop(columns=["apparent_size"])
        return result
