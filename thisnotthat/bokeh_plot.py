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


class TextLayers(Protocol):
    location_layers: List[npt.NDArray]
    labels: List[List[List[str]]]
    labels_for_display: List[List[str]]


def add_text_layer(
    plot_figure: bokeh.plotting.Figure,
    text_dataframe: pd.DataFrame,
    text_size: float,
    layer_type: str = "middle",
    *,
    angle: float = 0,
    text_color: str = "#444444",
    text_font: Dict[str, str] = {"value": "helvetica"},
    text_font_style: str = "normal",
    text_line_height: float = 0.9,
    text_alpha: float = 1.0,
    max_text_size: float = 64.0,
    min_text_size: float = 2.0,
    text_transition_width: float = 16.0,
) -> None:
    """Add a textual label layer to a bokeh plot figure.

    Parameters
    ----------
    plot_figure: Figure
        The bokeh plot figure to add textual labels to

    text_dataframe: DatFrame
        A dataframe containing columns for "x", "y", and "text" giving the locations and text content of labels.

    text_size: float
        The size of text in pt to use for this label layer

    layer_type: str (optional, default = "middle")
        The "kind" of layer -- should be one of:
            * "bottom"
            * "middle"
            * "top"
        This controls how transparency works -- "bottom" layers never disappear upon zooming in, and "top" layers
        never disappear on zooming out.

    angle: float (optional, default = 0.0)
        The angle of rotation of the text in this layer.

    text_color: str (optional, default = "#444444")
        The colour of the text in this layer.

    text_font: dict (optional, default = {"value":"helvetica"}
        Text font information as passed to bokeh's ``Text`` marker type.

    text_font_style: str (optional, default = "normal")
        The font style as passed to bokeh; options include "bold", "italic" and others.

    text_line_height: float (optional, default = 0.9)
        The line height of text. Decreasing this will compact lines closer together (potentially resulting in overlap.

    text_alpha: float (optional, default = 1.0)
        The default alpha level of the text in this layer.

    max_text_size: float (optional, default = 64.0)
        The maximum apparent size of text to use before transitioning to another layer.

    min_text_size: float (optional, default = 2.0)
        The minimum apparent size of text to use before transitioning to another layer.

    text_transition_width: float (optional, default = 16.0)
        The range of apparent point sizes over which to perform a transparency based fade when transitioning to
        another layer.
    """
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
    upper_transition_val = max_text_size - text_transition_width
    lower_transition_val = min_text_size + text_transition_width
    text_resize_callback = bokeh.models.callbacks.CustomJS(
        args=dict(labels=labels),
        code="""
    const scale = cb_obj.end - cb_obj.start;
    const text_size = (%f / scale);
    if (text_size > %f && %s) {
        var alpha = (%f - text_size) / %f;

    } else if (text_size < %f && %s) {
        var alpha = (text_size - %f) / %f;

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
            upper_transition_val,
            "false" if layer_type == "bottom" else "true",
            max_text_size,
            text_transition_width,
            lower_transition_val,
            "false" if layer_type == "top" else "true",
            min_text_size,
            text_transition_width,
        ),
    )
    plot_figure.add_glyph(label_data_source, labels)
    plot_figure.x_range.js_on_change("start", text_resize_callback)


class BokehPlotPane(pn.viewable.Viewer, pn.reactive.Reactive):
    """A Scatterplot pane for visualizing a map representation of data using Bokeh as the plot backend. The bokeh 
    backend provides a lot of versatility and interactivity options, including lasso-selection, in plot legends
    that update, and more. This plot makes use of bokeh's webgl backend, and can scale well into the tens of
    thousands of points, but performance can be slower for larger datasets at which point the dek.gl backend may 
    become preferable.

    Parameters
    ----------
    data: Array of shape (n_samples, 2)
        The data of the map representation to be plotted, formatted as a numpy array of shape (n_samples, 2).
        
    labels: Iterable of str of length n_samples or None (optional, default = None)
        If text class labels for data points is available it can be passed here, and used for colouring points,
        as well as potentially interacting with label editor tools. If ``None`` the data will be treated as "unlabelled"
        for the purpose of class labelling, and can then be added to via the label editor.
        
    hover_text: Iterable of str of length n_samples or None (optional, default = None)
        If a text item for each sample to use when hovering over points is available it can be passed here. If 
        ``None`` then the ``labels`` will be used for any hover text.
        
    marker_size: float, Iterable of float of length n_samples, or None (optional, default = None)
        Individual markers can be sized by passing a vector of sizes. If a single float value is passed then all
        markers will be sized at the given float value. If ``None`` a default marker size of 0.1 will be used.
        
    label_color_mapping: dict or None (optional, default = None)
        If ``labels`` are provided this color_mapping will be used -- it should be a dictionary keyed by the class
        labels, with values giving hexstring colour specifications for the desired colour of the class label. If
        ``None`` a colormapping will be generated automatically based on the labels and palette.

    palette: Sequence of str (optional, default = Turbo256 palette)
        The palette to use for colouring markers. It should be a sequence of hexstring colour specifications. Note
        that for class labelling the order of colours is modified, so smooth colour palettes can be used. Even if
        a ``label_color_mapping`` is provided, this palette will be used for generating new colours if new labels
        are created using the label editor.

    width: int (optional, default = 600)
        Width of the plot figure. Note that this includes space for the legend if ``show_legend`` is ``True``,
        and includes space for the legend outside the plot if that is selected.

    height: int (optional, default = 600)
        Height of the plot figure.

    max_point_size: float or None (optional, default = None)
        The largest apparent size of points in the scatterplot; further zooming will leave points fixed at this size.
        This is useful when you want points clearly visible when zoomed out, but want to avoid excessive overlaps of
        points when zooming in.

    min_point_size: float or None (optional, default = None)
        The smallest apparent size of points in the scatterplot; further zooming out will leave points fixed at this
        size. This can be useful if you want points to remain visible when zoomed out where they otherwise would not
        because of scaling issues.

    marker_scale_factor: float or None (optional, default = None)
        If setting marker size via the ``marker_size`` param or via the plot control pane this can control the overall
        scale or size of the markers. Set this to what you want the "average" marker size to be.

    fill_alpha: float (optional, default = 0.75)
        The alpha value to use for the fill on points in the scatterplot.

    line_color: str (optional, default = "#FFFFFF")
        The line colour to use for the outline of points in the scatterplot.

    line_width: float (optional, default = 0.25)
        The line width to use for the outline of points in the scatterplot.

    hover_fill_color: str (optional, default = "#FF0000")
        The colour of points being hovered over in the scatterplot -- use this for highlighting points with hover.

    hover_line_color: str (optional, default = "#000000")
        The line colour of points being hovered over in the scatterplot.

    hover_line_width: float (optional, default = 2)
        The line width of points being hovered over in the scatterplot.

    tooltip_template: str (optional, default = "<div>@hover_text</div>"
        The template used to define the hover tooltip in the scatterplot. See custom tooltips in the bokeh documentation
        for more details on this.

    selection_fill_alpha: float (optional, default = 1.0)
        The alpha value to use for selected points in the scatterplot.

    nonselection_fill_alpha: float (optional, default = 0.1)
        The alpha value to use for points that are not in an active selection in the scatterplot.

    nonselection_fill_color: str (optional, default = "gray")
        The colour to use to fill points that are not in an active selection in the scatterplot.

    background_fill_color: str (optional, default = "#FFFFFF")
        The colour to use for the background of the scatterplot.

    border_fill_color: str (optional, default = "whitesmoke")
        The colour to use for the background of the non-plot region of the bokeh figure.

    toolbar_location: str (optional, default = "above")
        The toolbar location in the bokeh figure. See the bokeh documentation for more details on options.

    tools: str (optional, default = "pan,wheel_zoom,lasso_select,save,reset,help")
        The tools to place in the toolbar in the bokeh figure. See the bokeh documentation for more details on options.

    title: str or None (optional, default = None)
        The title, if any, to put on the figure.

    title_location: str (optional, default = "above")
        The location of the title, if any. See the bokeh documentation for more details on options.

    show_legend: bool (optional, default = True)
        Whether to show a bokeh legend associated to the plot.

    legend_location: str (optional, default = "outside")
        Where to locate the legend, if a legend is being shown. The standard bokeh legend location options are
        available, and also the option "outside", to locate the legend to the right of the plot.

    name: str (optional, default = "Plot")
        The panel name of the plot pane. See panel documentation for more details.

    Attributes
    ----------

    plot: Figure
        The actual bokeh figure -- custom adjustments to the plot can be made via this attribute.

    pane: Pane
        The panel "Pane" object containing the plot. You can use this attribute to see the plot
        directly in ajupyter notebook.

    dataframe: DataFrame
        The dataframe associated with this plot, including label information that may have been
        edited via the label editor.
    """

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
        marker_scale_factor: Optional[float] = None,
        fill_alpha: float = 0.75,
        line_color: str = "white",
        line_width: float = 0.25,
        hover_fill_color: str = "red",
        hover_line_color: str = "black",
        hover_line_width: float = 2,
        tooltip_template: str = """<div>@hover_text</div>""",
        selection_fill_alpha: float = 1.0,
        nonselection_fill_alpha: float = 0.1,
        nonselection_fill_color: str = "gray",
        background_fill_color: str = "#ffffff",
        border_fill_color: str = "whitesmoke",
        toolbar_location: str = "above",
        tools="pan,wheel_zoom,lasso_select,save,reset,help",
        title: Optional[str] = None,
        title_location: str = "above",
        show_legend: bool = True,
        legend_location: str = "outside",
        name: str = "Plot",
    ):
        super().__init__(name=name)
        if labels is None:
            labels = ["unlabelled"] * len(data)
        if type(marker_size) in (int, float):
            marker_size = np.full(len(data), marker_size, dtype=np.float64)

        self.data_source = bokeh.models.ColumnDataSource(
            {
                "x": np.asarray(data).T[0],
                "y": np.asarray(data).T[1],
                "label": labels,
                "hover_text": hover_text if hover_text is not None else labels,
                "size": marker_size
                if marker_size is not None
                else np.full(len(data), 0.1),
                "apparent_size": marker_size
                if marker_size is not None
                else np.full(len(data), 0.1),
                "color_by": np.zeros(len(data), dtype=np.int8),
            }
        )
        self.data_source.selected.on_change("indices", self._update_selected)

        self._base_marker_size = pd.Series(
            marker_size if marker_size is not None else np.full(len(data), 0.1)
        )
        if marker_scale_factor is None:
            self._base_marker_scale = np.mean(self._base_marker_size)
        else:
            self._base_marker_scale = marker_scale_factor

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
                self.color_mapping.palette[x] for x in _palette_index(len(palette))
            ]

        self.plot = bokeh.plotting.figure(
            width=width,
            height=height,
            output_backend="webgl",
            background_fill_color=background_fill_color,
            border_fill_color=border_fill_color,
            toolbar_location=toolbar_location,
            tools=tools,
            title=title,
            title_location=title_location,
        )
        self.plot.toolbar.active_scroll = self.plot.select_one(
            bokeh.models.WheelZoomTool
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
        )
        if show_legend:
            self._legend = bokeh.models.Legend(
                items=[
                    bokeh.models.LegendItem(
                        label={"field": "label"}, renderers=[self.points]
                    )
                ],
                location=legend_location if legend_location != "outside" else "center",
                label_width=150,
            )
            self.plot.add_layout(
                self._legend, "right" if legend_location == "outside" else "center",
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
            bokeh.models.HoverTool(tooltips=tooltip_template, renderers=[self.points])
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
            rescaled_size = self._base_marker_scale * (
                rescaled_size - rescaled_size.min() + 1
            )
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
            palette = self._base_palette
        else:
            palette = self.color_by_palette

        if len(self.color_by_vector) == 0:
            self.points.glyph.fill_color = self._label_colormap
            if self.show_legend:
                if self._legend.items[0].label["field"] != "label":
                    self._legend.items[0].label["field"] = "label"
                self._legend.visible = True
                self._color_by_legend.visible = False

        elif pd.api.types.is_numeric_dtype(self.color_by_vector):
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
                self._color_by_renderer.glyph.fill_color = colormap
                self._legend.visible = False
                self._color_by_legend.visible = True
        else:
            self.data_source.data["color_by"] = self.color_by_vector
            colormap = bokeh.transform.factor_cmap(
                "color_by", palette, list(self.color_by_vector.unique())
            )
            self.points.glyph.fill_color = colormap
            if self.show_legend:
                self._legend.items[0].label["field"] = "color_by"
                self._legend.visible = True
                self._color_by_legend.visible = False

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
        cluster_labelling: TextLayers,
        *,
        angle: float = 0,
        text_size_scale: int = 12,
        text_layer_scale_factor: float = 2.0,
        text_color: str = "#444444",
        text_font: Dict[str, str] = {"value": "helvetica"},
        text_font_style: str = "normal",
        text_line_height: float = 0.9,
        text_alpha: float = 1.0,
        max_text_size: float = 64.0,
        min_text_size: float = 2.0,
        text_transition_width: float = 16.0,
    ):
        """Given a cluster labelling generated via one of the methods in the ``map_cluster_labelling``
        module, add the labelling information to this plot in a manner that allows lower level, more detailed,
        labels to be revealed upon zooming in, with transitions between the layers, and appropriate sizing
        of labels at different levels.

        Parameters
        ----------
        cluster_labelling: TextLayers
            A cluster labelling in multiple layers produced by a method from ``map_cluster_labelling``.

        angle: float (optional, default = 0.0)
            The angle of rotation of the text in this layer.

        text_size_scale: float (optional, default = 12)
            A base scale for the text.

        text_layer_scale_factor: flaot (optional, default = 2.0)
            The multiplier to scale text sizes by when going up a layer.

        text_color: str (optional, default = "#444444")
            The colour of the text in this layer.

        text_font: dict (optional, default = {"value":"helvetica"}
            Text font information as passed to bokeh's ``Text`` marker type.

        text_font_style: str (optional, default = "normal")
            The font style as passed to bokeh; options include "bold", "italic" and others.

        text_line_height: float (optional, default = 0.9)
            The line height of text. Decreasing this will compact lines closer together (potentially resulting in overlap.

        text_alpha: float (optional, default = 1.0)
            The default alpha level of the text in this layer.

        max_text_size: float (optional, default = 64.0)
            The maximum apparent size of text to use before transitioning to another layer.

        min_text_size: float (optional, default = 2.0)
            The minimum apparent size of text to use before transitioning to another layer.

        text_transition_width: float (optional, default = 16.0)
            The range of apparent point sizes over which to perform a transparency based fade when transitioning to
            another layer.
        """
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
                text_size_scale * text_layer_scale_factor ** i,
                layer_type=layer_type,
                angle=angle,
                text_color=text_color,
                text_font=text_font,
                text_font_style=text_font_style,
                text_line_height=text_line_height,
                text_alpha=text_alpha,
                max_text_size=max_text_size,
                min_text_size=min_text_size,
                text_transition_width=text_transition_width,
            )

    @property
    def dataframe(self):
        result = pd.DataFrame(self.data_source.data)
        if "color_by" in result:
            result = result.drop(columns=["apparent_size", "color_by"])
        else:
            result = result.drop(columns=["apparent_size"])
        return result
