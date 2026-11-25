"""Closed dispatch for the registered local U/V FIT→freeze→PROBE campaigns.

No caller supplies a data root, authorization object, raw reader, fit boundary or model path.
Each label has independent source, data, job, freeze and admission receipts.
"""


def study(name):
    if name == "U":
        from . import (
            surprise_data as data,
            surprise_panel as panel,
            surprise_run as run,
        )
    elif name == "V":
        from . import (
            alternative_data as data,
            alternative_panel as panel,
            alternative_run as run,
        )
    else:
        raise ValueError("unknown two-phase study")
    return data, panel, run
