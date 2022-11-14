import panel as pn
import param
import pandas as pd
import numpy as np
import numpy.typing as npt
from bokeh.palettes import Turbo256

from .utils import _palette_index

from typing import *


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

    palette: Sequence of str or None (optional, default = None)
        The color palette to use; the order of colours will be matched to the order of factors to create an
        initial mapping from factors to colours. If ``None`` a default palette will be used instead. Note that
        having more colours than factors can be beneficial as new class labels, if created, will take colours
        from the remaining unused colours in the palette.

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
        palette: Optional[Sequence[str]] = None,
        selectable: bool = False,
        color_picker_width: int = 50,
        color_picker_height: int = 50,
        color_picker_margin: Sequence[int] = [1, 5],
        label_height: int = 50,
        label_width: int = 225,
        label_max_width: int = 225,
        label_min_width: int = 125,
        label_margin: Sequence[int] = [0, 0],
        name: str = "Editable Legend",
    ) -> None:
        super().__init__(name=name)
        label_series = pd.Series(labels).copy()  # reset_index(drop=True)
        self.label_set = set(label_series.unique())
        if factors is not None:
            self.label_color_factors = factors
        else:
            self.label_color_factors = list(self.label_set)
        self.label_color_palette = (
            palette
            if palette is not None
            else [Turbo256[x] for x in _palette_index(256)]
        )
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
        self.pane = pn.Column()
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
        for idx, label in enumerate(self.label_color_factors):
            if label in self.label_set and label not in legend_labels:
                legend_labels.add(label)
                color = self.label_color_palette[idx]
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
                        margin=[0, 2],
                    ),
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


class NewLabelButton(pn.reactive.Reactive):
    """A simple button for generating a new label for use with an editable legend. This simply wraps up a button
    widget with default options set, and an understanding of selections and labels for connecting with plots and
    editable legends.

    Parameters
    ----------
    labels: Array of shape (n_samples,)
        The class labels vector giving the class label of each sample.

    button_type: str (optional, default = "success")
        The panel button type used. See the panel documentation for more details.

    button_text: str (optional, default = "New Label")
        The text to display on the button.

    width: int or None (optional, default = None)
        The width of the button. If ``None`` then let the button size itself.

    name: str (optional, default = "New Label")
        The panel name of the pane. See panel documentation for more details.
    """

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], item_type=int, doc="Indices of selected samples")

    def __init__(
        self,
        labels: npt.ArrayLike,
        *,
        button_type: str = "success",
        button_text: str = "New Label",
        width: Optional[int] = None,
        name: str = "New Label",
    ) -> None:
        super().__init__(name=name)
        self.label_count = 1
        self.pane = pn.widgets.Button(
            name=button_text, button_type=button_type, width=width
        )
        self.pane.on_click(self._on_click)
        self.pane.disabled = True
        self.labels = pd.Series(labels).copy()  # .reset_index(drop=True)

    def _on_click(self, event: param.parameterized.Event) -> None:
        if len(self.selected) > 0:
            new_labels = self.labels
            new_labels.iloc[self.selected] = f"new_label_{self.label_count}"
            self.labels = new_labels
            self.label_count += 1

            self.selected = []

    @param.depends("selected", watch=True)
    def _toggle_active(self):
        if len(self.selected) > 0:
            self.pane.disabled = False
        else:
            self.pane.disabled = True

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
        color_picker_margin: Sequence[int] = [1, 5],
        label_height: int = 36,
        label_width: int = 225,
        label_max_width: int = 225,
        label_min_width: int = 125,
        label_margin: Sequence[int] = [0, 0],
        newlabel_button_type: str = "success",
        newlabel_button_text: str = "New Label",
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
        self.new_label_button = NewLabelButton(
            labels,
            button_type=newlabel_button_type,
            button_text=newlabel_button_text,
            width=label_width + color_picker_width,
        )
        self.legend.link(
            self,
            labels="labels",
            label_color_palette="label_color_palette",
            label_color_factors="label_color_factors",
            selected="selected",
            bidirectional=True,
        )
        self.new_label_button.link(
            self, labels="labels", selected="selected", bidirectional=True,
        )
        self.pane = pn.WidgetBox(
            title, self.legend, self.new_label_button, width=width, height=height
        )

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
        return self.link(
            plot,
            labels="labels",
            label_color_factors="label_color_factors",
            label_color_palette="label_color_palette",
            selected="selected",
            bidirectional=True,
        )
