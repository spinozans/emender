"""Closed dispatch for the registered local U/V/LS/DR/RB/EH/PC FIT→freeze→PROBE campaigns.

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
    elif name == "LS":
        from . import (
            literal_mark_data as data,
            literal_mark_panel as panel,
            literal_mark_run as run,
        )
    elif name == "DR":
        from . import decode_data as data, decode_panel as panel, decode_run as run
    elif name == "RB":
        from . import (
            residual_data as data,
            residual_panel as panel,
            residual_run as run,
        )
    elif name == "EH":
        from . import (
            error_history_data as data,
            error_history_panel as panel,
            error_history_run as run,
        )
    elif name == "PC":
        from . import (
            paired_correction_data as data,
            paired_correction_panel as panel,
            paired_correction_run as run,
        )
    else:
        raise ValueError("unknown two-phase study")
    return data, panel, run


def incumbent(name):
    if name not in ("V", "LS", "DR", "RB", "EH", "PC"):
        raise ValueError("not a post-U study")
    return study(name)
