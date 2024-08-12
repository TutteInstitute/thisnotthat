

import datetime as dt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Expect, expect

from panel.tests.util import serve_component, wait_until

import thisnotthat as tnt

pytestmark = pytest.mark.ui


@pytest.fixture
def df_mixed():
    df = pd.DataFrame({
        'int': [1, 2, 3, 4],
        'float': [3.14, 6.28, 9.42, -2.45],
        'str': ['A', 'B', 'C', 'D'],
        'bool': [True, True, True, False],
        'date': [dt.date(2019, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 10), dt.date(2019, 1, 10)],
        'datetime': [dt.datetime(2019, 1, 1, 10), dt.datetime(2020, 1, 1, 12), dt.datetime(2020, 1, 10, 13), dt.datetime(2020, 1, 15, 13)],
    })
    df.index.name = 'row_num'
    return df


def test_datapane_selection(page, df_mixed):
    widget = tnt.DataPane(df_mixed)
    serve_component(page, widget)

    rows = page.locator('.tabulator-row')
    row_to_select = 0
    c0 = page.get_by_role("gridcell", name="Select Row").nth(row_to_select)
    c0.wait_for()
    c0.click()

    wait_until(lambda: widget.selected == [row_to_select], page)
    assert 'tabulator-selected' in rows.nth(row_to_select).get_attribute('class')
    unselected_rows = [i for i in range(rows.count()) if i != row_to_select]
    for i in unselected_rows:
        assert 'tabulator-selected' not in rows.nth(i).get_attribute('class')

    expected_selected = df_mixed.loc[[row_to_select], :]
    assert widget.selected_dataframe.equals(expected_selected)

