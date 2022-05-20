import panel as pn
import param
import pandas as pd

from typing import *

class InformationPane(pn.reactive.Reactive):

    selected = param.List(default=[], doc="Indices of selected samples")
    data = param.DataFrame(doc="Source data")

    def __init__(
            self,
            raw_dataframe: pd.DataFrame,
            markdown_template: str,
            *,
            width: int = 200,
            height: int = 600,
            placeholder_text: str = "<center> ... nothing selected ...",
            dedent: bool = False,
            disable_math: bool = False,
            extensions: List[str] = ["extra", "smarty", "codehilite"],
            style: dict = {},
            margin: List[int] = [5, 5],
            name: str = "Information",
    ):
        super().__init__(name=name)
        self.data = raw_dataframe
        self.markdown_template = markdown_template
        self.placeholder_text = placeholder_text
        self.pane = pn.pane.Markdown(
            self.placeholder_text,
            width=width,
            height=height,
            margin=margin,
            dedent=dedent,
            disable_math=disable_math,
            extensions=extensions,
            style=style,
        )

    def _get_model(self, *args, **kwds):
        return self.pane._get_model(*args, **kwds)

    @param.depends("selected", watch=True)
    def _info_pane_update_selection(self) -> None:
        if len(self.selected) == 0:
            self.pane.object = self.placeholder_text
        else:
            substitution_dict = {
                col: self.data[col].iloc[self.selected[-1]]
                for col in self.data.columns
            }
            self.pane.object = self.markdown_template.format(
                **substitution_dict
            )