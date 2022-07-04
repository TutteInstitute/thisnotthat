import panel as pn
import param
import pandas as pd
import numpy as np
import numpy.typing as npt
from bokeh.palettes import Turbo256

from .utils import _palette_index

from typing import *


class LegendPane(pn.reactive.Reactive):

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    label_color_palette = param.List([], item_type=str, doc="Color palette")
    label_color_factors = param.List([], item_type=str, doc="Color palette")

    def __init__(
        self,
        labels: npt.ArrayLike,
        factors: List[str],
        palette: Optional[Sequence[str]] = None,
        *,
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
        self.label_color_factors = factors
        self.label_color_palette = (
            palette
            if palette is not None
            else [Turbo256[x] for x in _palette_index(256)]
        )
        self.labels = label_series
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
                )
                legend_items.append(legend_item)
                legend_item[0].param.watch(
                    self._color_callback, "value", onlychanged=True
                )
                legend_item[1].param.watch(
                    self._label_callback, "value", onlychanged=True
                )
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

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], doc="Indices of selected samples")

    def __init__(
        self,
        labels: npt.ArrayLike,
        *,
        button_type: str = "success",
        button_text: str = "New Label",
        name: str = "New Label",
    ) -> None:
        super().__init__(name=name)
        self.label_count = 1
        self.pane = pn.Column(
            pn.widgets.Button(name=button_text, button_type=button_type)
        )
        self.pane[0].on_click(self._on_click)
        self.labels = pd.Series(labels).copy()  # .reset_index(drop=True)

    def _on_click(self, event: param.parameterized.Event) -> None:
        if len(self.selected) > 0:
            new_labels = self.labels
            new_labels.iloc[self.selected] = f"new_label_{self.label_count}"
            self.labels = new_labels
            self.label_count += 1

            if len(self.pane) > 1:
                self.pane.pop(1)

        elif len(self.pane) < 2:
            self.pane.append(pn.pane.Alert("No data selected!", alert_type="danger"))

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)


class LabelEditorPane(pn.reactive.Reactive):

    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    label_color_palette = param.List([], item_type=str, doc="Color palette")
    label_color_factors = param.List([], item_type=str, doc="Color palette")
    selected = param.List(default=[], doc="Indices of selected samples")

    def __init__(
        self,
        labels: npt.ArrayLike,
        color_factors: Optional[List[str]] = None,
        color_palette: Optional[Sequence[str]] = None,
        *,
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
        name: str = "Label Editor",
    ) -> None:
        super().__init__(name=name)
        self.labels = pd.Series(labels).copy()  # .reset_index(drop=True)

        if color_factors is None:
            color_factors = list(set(labels))

        self.legend = LegendPane(
            labels,
            color_factors,
            color_palette,
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
            labels, button_type=newlabel_button_type, button_text=newlabel_button_text,
        )
        self.legend.link(
            self,
            labels="labels",
            label_color_palette="label_color_palette",
            label_color_factors="label_color_factors",
            bidirectional=True,
        )
        self.new_label_button.link(
            self, labels="labels", selected="selected", bidirectional=True,
        )
        self.pane = pn.Column(self.legend, self.new_label_button)

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)
