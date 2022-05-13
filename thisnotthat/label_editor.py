import panel as pn
import param
import pandas as pd
import numpy as np
import numpy.typing as npt

from typing import *

class LegendPane(pn.reactive.Reactive):
    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    color_palette = param.List([], item_type=str, doc="Color palette")
    color_factors = param.List([], item_type=str, doc="Color palette")

    def _color_callback(self, event: param.parameterized.Event) ->None:
        self.color_palette = [
            event.new if color == event.old else color for color in self.color_palette
        ]

    def _label_callback(self, event: param.parameterized.Event) -> None:
        label_mapping = {
            label: event.new if label == event.old else label
            for label in self.labels.unique()
        }
        self.color_factors = [
            label_mapping[factor] if factor in label_mapping else factor
            for factor in self.color_factors
        ]
        new_labels = self.labels.map(label_mapping)
        self.labels = new_labels
        self.label_set = set(self.labels.unique())

    def _rebuild_pane(self) -> None:
        self.label_set = set(self.labels.unique())
        legend_labels = set([])
        legend_items = []
        for idx, label in enumerate(self.color_factors):
            if label in self.label_set and label not in legend_labels:
                legend_labels.add(label)
                color = self.color_palette[idx]
                legend_item = pn.Row(
                    pn.widgets.ColorPicker(value=color, width=50, margin=[0, 5]),
                    pn.widgets.TextInput(value=label, margin=[0, 0], max_width=225),
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

    def __init__(self, labels: npt.ArrayLike, factors: List[str], palette: List[str]) -> None:
        super().__init__()
        self.label_set = set(np.unique(labels))
        self.color_factors = factors
        self.color_palette = palette
        self.labels = pd.Series(labels)
        self.label_set = set(self.labels)
        self.pane = pn.Column()
        self._rebuild_pane()

    # Reactive requires this to make the model auto-display as requires
    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("labels", watch=True)
    def _update_labels(self) -> None:
        new_label_set = set(self.labels.unique())
        self.color_factors = self.color_factors + list(
            new_label_set - set(self.color_factors)
        )

        if new_label_set != self.label_set:
            self._rebuild_pane()


class NewLabelButton(pn.reactive.Reactive):
    labels = param.Series(default=pd.Series([], dtype="object"), doc="Labels")
    selected = param.List(default=[], doc="Indices of selected samples")

    def _on_click(self, event: param.parameterized.Event) -> None:
        if len(self.selected) > 0:
            new_labels = self.labels
            new_labels[self.selected] = f"new_label_{self.label_count}"
            self.labels = new_labels
            self.label_count += 1

            if len(self.pane) > 1:
                self.pane.pop(1)

        elif len(self.pane) < 2:
            self.pane.append(pn.pane.Alert("No data selected!", alert_type="danger"))

    def __init__(self, labels: npt.ArrayLike) -> None:
        super().__init__()
        self.label_count = 1
        self.pane = pn.Column(
            pn.widgets.Button(name="New Label", button_type="success")
        )
        self.pane[0].on_click(self._on_click)
        self.labels = pd.Series(labels)

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)


class LabelEditorPane(pn.reactive.Reactive):
    @property
    def labels(self) -> pd.Series:
        return self.legend.labels

    @labels.setter
    def labels(self, new_labels: npt.ArrayLike) -> None:
        self.legend.labels = pd.Series(new_labels)

    @property
    def selected(self) -> List[int]:
        return self.new_label_button.selected

    @selected.setter
    def selected(self, selection: List[int]) -> None:
        self.new_label_button.selected = selection

    def __init__(self, labels: npt.ArrayLike, color_factors: List[str], color_palette: List[str]) -> None:
        super().__init__()
        self.legend = LegendPane(labels, color_factors, color_palette)
        self.new_label_button = NewLabelButton(labels)
        self.new_label.link(self.legend, bidirectional=True, labels="labels")
