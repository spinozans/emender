"""Nonlinear Delta Memory model package.

The current optimized NDM implementation is still exposed through historical
class names such as ``E88FusedLM`` and ``LadderLM``. Public documentation should
refer to the architecture family as NDM and to the production instance as
E88/NDM until those internal names are migrated.
"""

from .models import (
    StockElman, StockElmanCell,
    LadderLM, create_ladder_model,
    get_available_levels, get_ladder_level,
)

__version__ = "0.2.0"
