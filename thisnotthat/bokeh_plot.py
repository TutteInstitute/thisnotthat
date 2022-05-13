import panel as pn
import param
import bokeh.plotting
import bokeh.models
import bokeh.transform
import bokeh.palettes

# Provide indexing into a list that jumps around a lot
# ideal for selecting varied colors from a large linear color palette
def _palette_index(size):
    step = size
    current = size - 1
    used = set([])
    for i in range(size):
        used.add(current)
        yield current

        while current in used:
            current += step
            if current >= size:
                step //= 2
                current = 0
            if step == 0:
                break

    return


class BokehPlotPane(pn.viewable.Viewer, pn.reactive.Reactive):
    labels = param.Series(doc="Labels")
    color_palette = param.List([], item_type=str, doc="Color palette")
    color_factors = param.List([], item_type=str, doc="Color palette")
    selected = param.List([], doc="Indices of selected samples")

    def _update_selected(self, attr, old, new):
        self.selected = self.data_source.selected.indices

    def __init__(self, data, labels, annotation):
        super().__init__()
        self.data_source = bokeh.models.ColumnDataSource(
            {"x": data.T[0], "y": data.T[1], "label": labels, "annotation": annotation}
        )
        self.data_source.selected.on_change("indices", self._update_selected)
        self._factor_cmap = bokeh.transform.factor_cmap(
            "label", palette=bokeh.palettes.Turbo256, factors=list(set(labels))
        )
        self.color_mapping = self._factor_cmap["transform"]
        self.color_mapping.palette = [
            self.color_mapping.palette[x] for x in _palette_index(256)
        ]

        self.plot = bokeh.plotting.figure(
            width=600,
            height=600,
            output_backend="webgl",
            border_fill_color="whitesmoke",
        )
        points = self.plot.circle(
            source=self.data_source,
            radius=0.1,
            color=self._factor_cmap,
            alpha=0.75,
            line_color="white",
            line_width=0.25,
            hover_fill_color="red",
            hover_line_color="black",
            hover_line_width=2,
            selection_fill_alpha=1.0,
            nonselection_fill_alpha=0.1,
            nonselection_fill_color="gray",
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
        #         self.plot.legend.click_policy="mute"
        self.pane = pn.pane.Bokeh(self.plot)

        self.labels = self.dataframe.label
        self.color_palette = list(self.color_mapping.palette)
        self.color_factors = list(self.color_mapping.factors)

    # Reactive requires this to make the model auto-display as requires
    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("color_palette", watch=True)
    def _update_palette(self):
        self.color_mapping.palette = self.color_palette
        pn.io.push_notebook(self.pane)

    @param.depends("color_factors", watch=True)
    def _update_factors(self):
        self.color_mapping.factors = self.color_factors
        pn.io.push_notebook(self.pane)

    @param.depends("labels", watch=True)
    def _update_labels(self):
        self.data_source.data["label"] = self.labels  # self.dataframe["label"]
        # We auto-update the factors from elsewhere (? from legend yes, but not from table edits)
        #         self.factors = list(self.color_mapping.factors) + [
        #             x for x in self.labels.unique() if x not in self.color_mapping.factors
        #         ]
        pn.io.push_notebook(self.pane)

    @param.depends("selected", watch=True)
    def _update_selection(self):
        self.data_source.selected.indices = self.selected
        pn.io.push_notebook(self.pane)
