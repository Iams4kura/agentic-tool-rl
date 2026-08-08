"""Reward decomposition and potential-based shaping."""

from agentic_tool_rl.rewards.potential import (
    RewardBreakdown,
    discounted_shaping_sum,
    potential_shaping,
    shape_trajectory,
)

__all__ = [
    "RewardBreakdown",
    "discounted_shaping_sum",
    "potential_shaping",
    "shape_trajectory",
]
