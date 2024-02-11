import functools
import panel as pn
import param
import pandas as pd
import numpy as np
import numpy.typing as npt
from glasbey import extend_palette

from .palettes import get_palette
from .utils import _palette_index

from typing import *

from bokeh.events import Reset


class LegendWidget(pn.reactive.Reactive):
    """An interactive legend with editable label names, and a colour picker for legend item colours. By linking this
    pane to a PlotPane with regard to ``labels``, ``label_color_palette`` and ``label_color_factors`` you can edit
    the names of class labels, and alter the colours used in the palette for plotting class labels, while having those
    changes reflected in the labels themselves. This allows for easy label editing via this legend pane.

    The legend can also be made "selectable" with buttons for selecting items from the legend, with the ``selected``
    param reflecting that (thus, if ``selected`` is linked to a PlotPane the legend selection will be displayed
    in the plot)

    Parameters
    ----------
    labels: Array of shape (n_samples,)
        The class labels vector giving the class label of each sample.

    factors: List of str or None (optional, default = [])
        The different factors (distinct class names) to recognise in the legend and display. If ``None`` the factors
        will be derived from the ``labels`` array.

    palette: str or Sequence of str or None (optional, default = None)
        The color palette to use; the order of colours will be matched to the order of factors to create an initial
        mapping from factors to colours. You may use named palette from ``tnt.palettes.all_palettes``, or a sequence
        of hexstring colour specifications. If ``None`` a default palette will be used instead. Note that having more
        colours than factors can be beneficial as new class labels, if created, will take colours from the remaining
        unused colours in the palette.

    palette_length: int or None (optional, default = None)
        The length of the palette to use if palette is a named palette; this is particularly relevant for continuous
        palettes. If ``None`` then the length will be determined from labels, or if no labels are provided a default
        length of 256 will be used.

    palette_shuffle: bool (optional, default = False)
        Whether to shuffle the palette. If using a continuous palette for categorical labels this should be set
        to ``True`` to try to provide as much distinguishability between colours as possible.

    selectable: bool (optional, default = False)
        Whether to add selection toggle buttons to each legend item, allowing for selections to be made based
        on class labels in the legend.

    color_picker_width: int (optional, default = 50)
        The width of the colour picker button used for selecting colours in the editable legend.

    color_picker_height: int (optional, default = 50)
        The height of the colour picker button used for selecting colours in the editable legend.

    color_picker_margin: List of int (optional, default = [1, 5])
        The margin of the colour picker button used for selecting colours in the editable legend.

    label_height: int (optional, default = 50)
        The height of the editable legend class name textboxes used for in the editable legend.

    label_width: int (optional, default = 225)
        The width of the editable legend class name textboxes used for in the editable legend.

    label_max_width: int (optional, default = 225)
        The maximum allowable  of the editable legend class name textboxes used for in the editable legend.

    label_min_width: int (optional, default = 125)
        The minimum allowable width  of the editable legend class name textboxes used for in the editable legend.

    label_margin: List of int (optional, default = [0, 0]
        The margin of the editable legend class name textboxes used for in the editable legend.

    name: str (optional, default = "Editable Legend")
        The panel name of the pane. See the panel documentation for more details.
    """

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    label_color_palette = param.List([], item_type=str, doc="Color palette")
    label_color_factors = param.List([], item_type=str, doc="Color palette")
    selected = param.List([], item_type=int, doc="Indices of selected samples")

    def __init__(
        self,
        labels: npt.ArrayLike,
        *,
        factors: Optional[List[str]] = None,
        palette: Optional[Union[str, Sequence[str]]] = None,
        palette_length: Optional[int] = None,
        palette_shuffle: bool = False,
        selectable: bool = False,
        color_picker_width: int = 50,
        color_picker_height: int = 50,
        color_picker_margin: tuple[int] = (1, 5),
        label_height: int = 50,
        label_width: int = 225,
        label_max_width: int = 225,
        label_min_width: int = 125,
        label_margin: tuple[int] = (0, 0),
        name: str = "Editable Legend",
    ) -> None:
        super().__init__(name=name)
        label_series = pd.Series(labels).copy()  # reset_index(drop=True)
        self.label_set = set(label_series.unique())
        if factors is not None:
            self.label_color_factors = factors
        else:
            self.label_color_factors = list(self.label_set)

        if palette_length is None:
            palette_length = 256

        if palette is None:
            self._base_palette = get_palette(
                "glasbey_category10", length=palette_length, scrambled=palette_shuffle
            )
        elif type(palette) is str:
            self._base_palette = get_palette(
                palette, length=palette_length, scrambled=palette_shuffle
            )
        else:
            if palette_length > len(palette):
                self._base_palette = extend_palette(
                    palette, palette_size=palette_length
                )
            else:
                self._base_palette = palette[:palette_length]

            if palette_shuffle:
                self._base_palette = [
                    self._base_palette[x]
                    for x in _palette_index(len(self._base_palette))
                ]

        self.labels = label_series
        self.selectable = selectable
        self.color_picker_width = color_picker_width
        self.color_picker_height = color_picker_height
        self.color_picker_margin = color_picker_margin
        self.label_width = label_width
        self.label_height = label_height
        self.label_max_width = label_max_width
        self.label_min_width = label_min_width
        self.label_margin = label_margin

        self._internal_selection = False

        self.pane = pn.Column(sizing_mode="stretch_height")
        self._rebuild_pane()

    def _color_callback(self, event: param.parameterized.Event) -> None:
        self.label_color_palette = [
            event.new if color == event.old else color
            for color in self.label_color_palette
        ]

    def _label_callback(self, event: param.parameterized.Event) -> None:
        label_mapping = {
            label: event.new if label == event.old else label
            for label in self.labels.unique()
        }
        self.label_color_factors = [
            label_mapping[factor] if factor in label_mapping else factor
            for factor in self.label_color_factors
        ]
        new_labels = self.labels.map(label_mapping)
        self.labels = new_labels
        self.label_set = set(self.labels.unique())

    def _toggle_select(self, event) -> None:
        button = event.obj
        toggle_state = bool(button.clicks % 2)
        if toggle_state:
            button.name = "✓"
            button.button_type = "success"
            indices_to_select = np.where(self.labels == button.label_id)[0]
            new_selection = (
                np.union1d(self.selected, indices_to_select).astype(int).tolist()
            )
            self._internal_selection = True
            self.selected = new_selection
            self._internal_selection = False
        else:
            button.name = ""
            button.button_type = "default"
            indices_to_deselect = np.where(self.labels == button.label_id)[0]
            new_selection = (
                np.setdiff1d(self.selected, indices_to_deselect).astype(int).tolist()
            )
            self._internal_selection = True
            self.selected = new_selection
            self._internal_selection = False

    @param.depends("selected", watch=True)
    def _update_selected(self):
        if self.selectable and not self._internal_selection:
            selected_set = set(self.selected)
            for legend_item in self.pane:
                selection_button = legend_item[2]
                indices_to_test = set(
                    np.where(self.labels == selection_button.label_id)[0]
                )
                if indices_to_test <= selected_set:
                    # Ensure toggle is selected
                    selection_button.clicks = 1
                else:
                    # Ensure toggle is unselected
                    selection_button.clicks = 0

    def _rebuild_pane(self) -> None:
        self.label_set = set(self.labels.unique())
        legend_labels = set([])
        legend_items = []
        label_color_palette = []

        for idx, label in enumerate(self.label_color_factors):
            if label in self.label_set and label not in legend_labels:
                legend_labels.add(label)
                color = self._base_palette[idx]
                label_color_palette.append(color)
                
                legend_item = pn.Row(
                    pn.widgets.ColorPicker(
                        value=color,
                        width=self.color_picker_width,
                        height=self.color_picker_height,
                        margin=self.color_picker_margin,
                    ),
                    pn.widgets.TextInput(
                        value=label,
                        width=self.label_width,
                        height=self.label_height,
                        margin=self.label_margin,
                        max_width=self.label_max_width,
                        min_width=self.label_min_width,
                    ),
                    pn.widgets.Button(
                        name="",
                        button_type="default",
                        width=self.label_height,
                        height=self.label_height,
                        margin=(0, 2),
                    ),
                    sizing_mode="stretch_width",
                )
                legend_items.append(legend_item)
                legend_item[0].param.watch(
                    self._color_callback, "value", onlychanged=True
                )
                legend_item[1].param.watch(
                    self._label_callback, "value", onlychanged=True
                )
                if self.selectable:
                    legend_item[2].label_id = label
                    legend_item[2].on_click(self._toggle_select)
                else:
                    legend_item[2].visible = False
        self.label_color_palette = label_color_palette
        self.pane.clear()
        self.pane.extend(legend_items)
        

    # Reactive requires this to make the model auto-display as requires
    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("labels", watch=True)
    def _update_labels(self) -> None:
        new_label_set = set(self.labels.unique())

        self.label_color_factors = self.label_color_factors + list(
            new_label_set - set(self.label_color_factors)
        )

        if new_label_set != self.label_set:
            self._rebuild_pane()

    def link_to_plot(self, plot):
        self.labels = plot.labels
        
        self.label_color_factors = plot.label_color_factors
        self._base_palette = plot.label_color_palette
        
        if self.selectable:
            self.link(plot, selected="selected", bidirectional=False)
            return self.link(
                plot,
                labels="labels",
                label_color_factors="label_color_factors",
                label_color_palette="label_color_palette",
                bidirectional=True,
            )
        else:
            return self.link(
                plot,
                labels="labels",
                label_color_factors="label_color_factors",
                label_color_palette="label_color_palette",
                bidirectional=True,
            )


class AddToLabelWidget(pn.reactive.Reactive):
    """A widget for adding points to an existing label for use with an editable legend.
    This combines a button and select widget with default options set, and an understanding
    of data point selections and labels for connecting with plots and editable legends.

    Parameters
    ----------
    labels: Array of shape (n_samples,)
        The class labels vector giving the class label of each sample.

    button_type: str (optional, default = "success")
        The panel button type used. See the panel documentation for more details.

    button_text: str (optional, default = "Add to Existing Label")
        The text to display on the button.

    total_width: int or None (optional, default = None)
        The total width of the widget. If ``None`` then let the widget size itself.

    name: str (optional, default = "Add to Existing Label")
        The name of the pane. See panel documentation for more details.
    """

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], item_type=int, doc="Indices of selected samples")

    def __init__(
        self,
        labels: npt.ArrayLike,
        *,
        button_type: str = "success",
        button_text: str = "Add to Existing Label",
        total_width: Optional[int] = None,
        name: str = "Add to Existing Label",
    ) -> None:
        super().__init__(name=name)
        self.labels = pd.Series(labels).copy()
        self.options = list(self.labels.unique())
        self.selector = pn.widgets.Select(
            name="Select Label",
            options=self.options,
            width=total_width // 2,
            align="center",
        )
        self.button = pn.widgets.Button(
            name=button_text,
            button_type=button_type,
            width=total_width // 2,
            align=("center", "end"),
        )
        self.button.on_click(self._on_click)
        self.button.disabled = True
        self.selector.disabled = True
        self.pane = pn.Row(self.selector, self.button)

    def _on_click(self, event: param.parameterized.Event) -> None:
        if len(self.selected) > 0:
            new_labels = self.labels
            new_labels.iloc[self.selected] = self.selector.value
            self.labels = new_labels
            self.selected = []

    @param.depends("selected", watch=True)
    def _toggle_active(self):
        if len(self.selected) > 0:
            self.button.disabled = False
            self.selector.disabled = False
        else:
            self.button.disabled = True
            self.selector.disabled = True

    @param.depends("labels", watch=True)
    def _set_options(self):
        if hasattr(self, "selector"):
            self.options = list(self.labels.unique())
            self.selector.options = self.options

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)


class LabelEditorWidget(pn.reactive.Reactive):
    """A pane for editing class labels, ideally intended to be linked with a PlotPane. The pane itself is composed of
    an editable legend, and a "new label" button. With the editable legend you can edit the names of class labels,
    and alter the colours used in the palette for plotting class labels, while having those changes reflected in the
    labels themselves. The "new label" button is selection aware and can create a new class label based on the
    current selection. The editable legend is then updated accordingly.

    The legend can also be made "selectable" with buttons for selecting items from the legend, with the ``selected``
    param reflecting that (thus, if ``selected`` is linked to a PlotPane the legend selection will be displayed
    in the plot)

    Parameters
    ----------
    labels: Array of shape (n_samples,)
        The class labels vector giving the class label of each sample.

    color_factors: List of str or None (optional, default = [])
        The different factors (distinct class names) to recognise in the legend and display. If ``None`` the factors
        will be derived from the ``labels`` array.

    color_palette: Sequence of str or None (optional, default = None)
        The color palette to use; the order of colours will be matched to the order of factors to create an
        initial mapping from factors to colours. If ``None`` a default palette will be used instead. Note that
        having more colours than factors can be beneficial as new class labels, if created, will take colours
        from the remaining unused colours in the palette.

    selectable_legend: bool (optional, default = False)
        Whether to add selection toggle buttons to each legend item, allowing for selections to be made based
        on class labels in the legend.

    color_picker_width: int (optional, default = 50)
        The width of the colour picker button used for selecting colours in the editable legend.

    color_picker_height: int (optional, default = 50)
        The height of the colour picker button used for selecting colours in the editable legend.

    color_picker_margin: List of int (optional, default = [1, 5])
        The margin of the colour picker button used for selecting colours in the editable legend.

    label_height: int (optional, default = 50)
        The height of the editable legend class name textboxes used for in the editable legend.

    label_width: int (optional, default = 225)
        The width of the editable legend class name textboxes used for in the editable legend.

    label_max_width: int (optional, default = 225)
        The maximum allowable  of the editable legend class name textboxes used for in the editable legend.

    label_min_width: int (optional, default = 125)
        The minimum allowable width  of the editable legend class name textboxes used for in the editable legend.

    label_margin: List of int (optional, default = [0, 0]
        The margin of the editable legend class name textboxes used for in the editable legend.

    newlabel_button_type: str (optional, default = "success")
        The panel button type used. See the panel documentation for more details.

    newlabel_button_text: str (optional, default = "New Label")
        The text to display on the button.

    add_to_label: bool (optional, default = False)
        If set to ``True``, the widget for adding selected points to an existing label will be displayed.
        If set to ``False``, the widget will be hidden.

    add_to_label_button_type: str (optional, default = "success")
        The panel button type used. See the panel documentation for more details.

    add_to_label_button_text: str (optional, default = "Add to Existing Label")
        The text to display on the button for adding to the label.

    title: str (optional, default = "#### Label Editor")
        A markdown title to be placed at the top of the pane.

    width: int or None (optional, default = None)
        The width of the pane, or, if ``None`` let the pane size itself.

    height: int or None (optional, default = None)
        The height of the pane, or, if ``None`` let the pane size itself.

    name: str (optional, default = "Label Editor")
        The panel name of the pane. See panel documentation for more details.
    """

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    label_color_palette = param.List([], item_type=str, doc="Color palette")
    label_color_factors = param.List([], item_type=str, doc="Color palette")
    selected = param.List(default=[], item_type=int, doc="Indices of selected samples")

    def __init__(
        self,
        labels: npt.ArrayLike,
        *,
        color_factors: Optional[List[str]] = None,
        color_palette: Optional[Sequence[str]] = None,
        selectable_legend: bool = False,
        color_picker_width: int = 48,
        color_picker_height: int = 36,
        color_picker_margin: tuple[int] = (1, 5),
        label_height: int = 36,
        label_width: int = 225,
        label_max_width: int = 225,
        label_min_width: int = 125,
        label_margin: tuple[int] = (0, 0),
        newlabel_button_type: str = "success",
        newlabel_button_text: str = "New Label",
        add_to_label: bool = False,
        add_to_label_button_type: str = "success",
        add_to_label_button_text: str = "Add to Existing Label",
        title: str = "#### Label Editor",
        width: Optional[int] = None,
        height: Optional[int] = None,
        name: str = "Label Editor",
    ) -> None:
        super().__init__(name=name)
        self.labels = pd.Series(labels).copy()  # .reset_index(drop=True)

        if color_factors is None:
            color_factors = list(set(labels))
            
        self.legend = LegendWidget(
            labels,
            factors=color_factors,
            palette=color_palette,
            selectable=selectable_legend,
            color_picker_width=color_picker_width,
            color_picker_height=color_picker_height,
            color_picker_margin=color_picker_margin,
            label_height=label_height,
            label_width=label_width,
            label_max_width=label_max_width,
            label_min_width=label_min_width,
            label_margin=label_margin,
        )
        self.new_label_count = 1
        self.new_label_button = pn.widgets.Button(
            name=newlabel_button_text, button_type=newlabel_button_type, width=width,

        )

        self.add_to_label_widget = AddToLabelWidget(
            labels=labels,
            button_type=add_to_label_button_type,
            button_text=add_to_label_button_text,
            total_width=label_width + color_picker_width,
        )
        self.new_label_button.on_click(self._on_click)
        self.new_label_button.disabled = True

        self.add_to_label_widget.link(
            self,
            labels="labels",
            selected="selected",
            bidirectional=True,
        )
        if add_to_label:
            self.pane = pn.WidgetBox(
                title,
                self.legend,
                self.new_label_button,
                self.add_to_label_widget,
                width=width,
                height=height,
                sizing_mode="stretch_height"
            )
        else:
            self.pane = pn.WidgetBox(
                title, self.legend, self.new_label_button, width=width, height=height
            )

    def _on_click(self, event: param.parameterized.Event) -> None:
        if len(self.selected) > 0:
            new_labels = self.labels.copy()
            new_labels.iloc[self.selected] = f"new_label_{self.new_label_count}"
            self.labels = new_labels
            self.new_label_count += 1
            
            self.legend.labels = new_labels
            self.legend._rebuild_pane()

            self.selected = []

    @param.depends("selected", watch=True)
    def _toggle_active(self):
        if len(self.selected) > 0:
            self.new_label_button.disabled = False
        else:
            self.new_label_button.disabled = True

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    def link_to_plot(self, plot):
        """Link this pane to a plot pane using a default set of params that can sensibly be linked.

        Parameters
        ----------
        plot: PlotPane
            The plot pane to link to.

        Returns
        -------
        link:
            The link object.
        """
        self.labels = plot.labels
        self.label_color_factors = plot.label_color_factors
        self.label_color_palette = plot.label_color_palette
        
        self.legend.link_to_plot(plot)
        self.legend._rebuild_pane()

        return self.link(
            plot,
            labels="labels",
            label_color_factors="label_color_factors",
            label_color_palette="label_color_palette",
            selected="selected",
            bidirectional=True,
        )


class TagWidget(pn.reactive.Reactive):
    """An interactive legend to display tags and select points based on them when linked to a PlotPane. Each data point
    can have multiple tags and this widget contains checkboxes to select (checking "Y") or deselect (checking "N") tags.
    Points are highlighted if they have all of the selected tags. Any point that contains a deselected tag is greyed out.

    Parameters
    ----------
    tags: Array of shape (n_samples,)
        A vector giving the tags associated with each sample.

    checkbutton_height: int (optional, default = 50)
        The height of the selectable checkbutton.

    checkbutton_margin: List of int (optional, default = [0, 0]
        The margin of the of the selectable checkbutton.

    name: str (optional, default = "Editable Legend")
        The panel name of the pane. See the panel documentation for more details.

    title: str (optional, default = "#### Tag Selector")
        A markdown string to display as the title of the widget.

    tag_to_int: dict (optional, default = {})
        Dictionary that maps tag strings to integer ids

    int_to_tag: dict (optional, default = {})
        Dictionary that maps tag integer ids to strings
    """

    tags = param.Series(default=pd.Series([], dtype="object"), doc="Tags")
    tag_set = param.List(default=[], item_type=str, doc="Unique set of tags")
    selected = param.List(default=[], item_type=int, doc="Indices of selected samples")
    tag_int_id = param.Integer(default=0, doc="One-up integer ID for mapping tags to strings")
    tag_to_int = param.Dict(default={}, doc="Map tag strings to integers")
    int_to_tag = param.Dict(default={}, doc="Map tag integers to strings")

    def __init__(
        self,
        tags: npt.ArrayLike = None,
        *,
        checkbutton_height: int = 50,
        checkbutton_margin: tuple[int] = (0, 0),
        name: str = "Tags Legend",
        title: str = "#### Tag Selector",
        tag_to_int: dict = {},
        int_to_tag: dict = {},
    ) -> None:
        super().__init__(name=name)
        self.tags = pd.Series([set(t) for t in tags]).copy()
        self.int_to_tag = int_to_tag
        self.tag_to_int = tag_to_int
        self.get_tag_set()
        
        self.checkbutton_height = checkbutton_height
        self.checkbutton_margin = checkbutton_margin
        
        self.title = title

        self.selected_tags = set()
        self.deselected_tags = set()
        
        self._internal_selection = False

        self.pane = pn.Column(
            sizing_mode="stretch_height",
            scroll=True,
        )

        self._rebuild_pane()
        
    
    def get_tag_set(self) -> None:
        if self.tags is not None:
            tag_set = set()
            tags_copy = self.tags.copy()
            tags_copy.apply(lambda x: tag_set.update(x))
            tag_set = [self.int_to_tag[t] for t in tag_set]
            self.tag_set = sorted(list(tag_set), key=str.lower)
        else:
            self.tag_set = []        
            
    def _text_edit_callback(self, event: param.parameterized.Event) -> None:
        # Rename tag with new value
        tag_id = self.tag_to_int[event.old]
        self.tag_to_int[event.new] = tag_id
        self.int_to_tag[tag_id] = event.new
        
        # Remove old tag
        self.tag_to_int.pop(event.old)
        
        # We need to updated the name of the associated checkbox
        # There is probably a smarter way to to this than rebuilding the entire pane
        self._rebuild_pane()
            
    def _toggle_select(self, event) -> None:
        checkbutton = event.obj
        tag = checkbutton.name
        
        # TODO: add warning if both Y & N are selected
        if "Y" in checkbutton.value:
            self.selected_tags.add(self.tag_to_int[tag])
        elif "Y" not in checkbutton.value:
            self.selected_tags.discard(self.tag_to_int[tag])

        if "N" in checkbutton.value:
            self.deselected_tags.add(self.tag_to_int[tag])
        elif "N" not in checkbutton.value:
            self.deselected_tags.discard(self.tag_to_int[tag])
            
        # We want to match points which have a union of the selected tags
        # We then want to remove tags which contain any of the undesired tags
        to_select = np.where([self.selected_tags.issubset(s) for s in self.tags])[0]
        to_remove = np.where(
            [
                True if self.deselected_tags.intersection(s) else False
                for s in self.tags
            ]
        )[0]

        new_selection = np.setdiff1d(to_select, to_remove).tolist()

        # If no points match then grey out all the points
        if self.selected_tags or self.deselected_tags:
            if len(new_selection) == 0:
                new_selection = [-1]
        elif len(self.selected_tags) == 0 and len(self.deselected_tags) == 0:
            new_selection = []

        self._internal_selection = True
        self.selected = new_selection
        self._internal_selection = False

    def _rebuild_pane(self) -> None:
        legend_tags = set([])
        rows = []

        self.get_tag_set()
        
        # Reset selections
        self.selected = []
        self.selected_tags = set()
        self.deselected_tags = set()


        for idx, tag in enumerate(self.tag_set):
            if tag in self.tag_set and tag not in legend_tags:
                legend_tags.add(tag)
                text = pn.widgets.TextInput(value=tag)
                checkbutton_group = pn.widgets.CheckButtonGroup(
                    name=tag,
                    options=["Y", "N"],
                    button_type="default",
                    button_style="outline",
                    align="center",
                    height=self.checkbutton_height,
                    margin=self.checkbutton_margin,
                )

                watcher = checkbutton_group.param.watch(
                    self._toggle_select, ["value"], onlychanged=False
                )
                
                text.param.watch(
                    self._text_edit_callback, "value", onlychanged=True
                )
                
                rows.append(pn.Row(text, checkbutton_group))

        self.pane.clear()
        self.pane.extend(rows)

    # Reactive requires this to make the model auto-display as requires
    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    def _on_reset(self, event):
        self._rebuild_pane()

    def link_to_plot(self, plot):
        self.link(plot, selected="selected", bidirectional=True)
        # Reset selected tags when plot reset tool is used
        plot.pane.object.on_event(Reset, self._on_reset)
        return self.link(
            plot,
            selected="selected",
            # tags="tags",
            bidirectional=True,
        )

    def link_to_tag_editor(self, tag_editor):
        return self.link(
            tag_editor,
            selected="selected",
            tags="tags",
            tag_set="tag_set",
            tag_int_id="tag_int_id", 
            tag_to_int="tag_to_int", 
            int_to_tag="int_to_tag",
            bidirectional=True,
        )

class TagEditorWidget(pn.reactive.Reactive):
    """A pane for editing point tags, ideally intended to be linked with a PlotPane. The pane itself is composed of
    an editable legend, and a "new tag" button. With the editable legend you can edit the names of point tags,
    while having those changes reflected in the tags themselves. The "new tag" button is selection aware and 
    can create a new tag based on the current selection. The editable legend is then updated accordingly.

    Each data point can have multiple tags and this widget contains checkboxes to select (checking "Y") or 
    deselect (checking "N") tags. Points are highlighted if they have all of the selected tags. Any point that 
    contains a deselected tag is greyed out.

    Parameters
    ----------
    tags: Array of shape (n_samples,)
        A vector giving the tags associated with each sample.

    label_height: int (optional, default = 50)
        The height of the editable legend class name textboxes used for in the editable legend.

    label_width: int (optional, default = 225)
        The width of the editable legend class name textboxes used for in the editable legend.

    label_max_width: int (optional, default = 225)
        The maximum allowable  of the editable legend class name textboxes used for in the editable legend.

    label_min_width: int (optional, default = 125)
        The minimum allowable width  of the editable legend class name textboxes used for in the editable legend.

    label_margin: List of int (optional, default = [0, 0]
        The margin of the editable legend class name textboxes used for in the editable legend.

    new_tag_button_type: str (optional, default = "success")
        The panel button type used. See the panel documentation for more details.

    new_tag_button_text: str (optional, default = "Tag Selected Points")
        The text to display on the button.

    title: str (optional, default = "#### Tag Editor")
        A markdown title to be placed at the top of the pane.

    width: int or None (optional, default = None)
        The width of the pane, or, if ``None`` let the pane size itself.

    height: int or None (optional, default = None)
        The height of the pane, or, if ``None`` let the pane size itself.

    name: str (optional, default = "Tag Editor")
        The panel name of the pane. See panel documentation for more details.
    """
    tags = param.Series(default=pd.Series([], dtype="object"), doc="Tags")
    tag_set = param.List(default=[], item_type=str, doc="Unique set of tags")
    selected = param.List(default=[], item_type=int, doc="Indices of selected samples")
    tag_int_id = param.Integer(default=0, doc="One-up integer ID for mapping tags to strings")
    tag_to_int = param.Dict(default={}, doc="Map tag strings to integers")
    int_to_tag = param.Dict(default={}, doc="Map tag integers to strings")

    
    def __init__(
        self,
        tags: npt.ArrayLike,
        *,
        selectable_legend: bool = False,
        label_height: int = 36,
        label_width: int = 225,
        label_max_width: int = 225,
        label_min_width: int = 125,
        label_margin: tuple[int] = (0, 0),
        new_tag_button_type: str = "success",
        new_tag_button_text: str = "Tag Selected Points",
        title: str = "#### Tag Editor",
        width: Optional[int] = None,
        height: Optional[int] = None,
        name: str = "Tag Editor",
    ) -> None:
        super().__init__(name=name)
        tag_series = pd.Series([set(t) for t in tags]).copy()
        self.tag_set = self.get_initial_tag_set(tag_series)
        self._update_tag_mapping()

        self.tags = tag_series.apply(self._map_tags_to_int)

        self.tag_widget = TagWidget(
            self.tags,
            tag_to_int=self.tag_to_int,
            int_to_tag=self.int_to_tag,
        )
        self.tag_widget.link_to_tag_editor(self)
        
        self.default_dropdown_option = "Create New Tag"
        self.options = [self.default_dropdown_option] + self.tag_set
        
        self.new_tag_count = 1
        self.new_tag_button = pn.widgets.Button(name=new_tag_button_text, button_type=new_tag_button_type)
        self.new_tag_button.on_click(self._on_click)
        self.new_tag_button.disabled = True
        
        self.selector = pn.widgets.Select(
            name="",
            options=self.options,
        )
        self.selector.disabled = True

        self.pane = pn.WidgetBox(
            title,
            self.tag_widget,
            pn.Row(self.selector, self.new_tag_button),
            width=width,
            height=height,
            sizing_mode="stretch_height"
        )        

    def get_initial_tag_set(self, tags):
        if tags is not None:
            tag_set = set()
            tags_copy = tags.copy()
            tags_copy.apply(lambda x: tag_set.update(x))
            tag_set = sorted(list(tag_set), key=str.lower)
        else:
            tag_set = []      
        
        return tag_set
    
    @param.depends("tag_set", watch=True)
    def _update_tag_mapping(self):
        for tag in self.tag_set:
            if tag not in self.tag_to_int:
                self.tag_to_int[tag] = self.tag_int_id
                self.tag_int_id += 1
        self.int_to_tag = {id_:tag for tag, id_ in self.tag_to_int.items()}
        
    def _map_tags_to_int(self, tags):
        return set([self.tag_to_int[t] for t in tags])
        
    def _add_tag(self, tags_for_point, tag_name):
        if tags_for_point:
            tags_for_point.add(tag_name)
            return tags_for_point
        else:
            return tags_for_point

    def _on_click(self, event: param.parameterized.Event) -> None:
        if len(self.selected) > 0:
            
            if self.selector.value == self.default_dropdown_option:
                tag_name = f"new_tag_{self.new_tag_count}"
                self.new_tag_count += 1
            else:
                tag_name = self.selector.value
            
            self.tag_set.append(tag_name)
            self._update_tag_mapping()

            new_tags = self.tags.copy()
            new_tags.iloc[self.selected].apply(lambda x: x.add(self.tag_to_int[tag_name]))

            self.tags = new_tags
            self.tag_widget._rebuild_pane()
            self.selected = []

    @param.depends("selected", watch=True)
    def _toggle_active(self):
        if len(self.selected) > 0:
            self.new_tag_button.disabled = False
            self.selector.disabled = False
        else:
            self.new_tag_button.disabled = True
            self.selector.disabled = True
            
    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)


    def link_to_plot(self, plot):
        
        if self.tags.empty:
            default_tags = [set() for _ in range(len(plot.dataframe))]
            self.tags = pd.Series(default_tags, dtype=object)
        
        plot.tags = self.tags
        plot.int_to_tag = self.int_to_tag
        self.link(plot, tags="tags", int_to_tag="int_to_tag", bidirectional=True)
        
        self.tag_widget.link_to_plot(plot)
        self.tag_widget._rebuild_pane()

        return self.link(
            plot,
            selected="selected",
            # tags="tags",
            bidirectional=True,
        )

    