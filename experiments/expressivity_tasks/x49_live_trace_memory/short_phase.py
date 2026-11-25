"""Closed dispatch for no-fit, whole short-document studies. Never pass an arbitrary raw reader or split."""


def study(name):
    if name == "UC":
        from . import (
            confirm_data as data,
            confirm_panel as panel,
            confirm_run as run,
            confirm_analysis as analysis,
        )
    elif name == "JT":
        from . import (
            transfer_data as data,
            transfer_panel as panel,
            transfer_run as run,
            transfer_analysis as analysis,
        )
    else:
        raise ValueError("unknown frozen short-document study")
    return data, panel, run, analysis
