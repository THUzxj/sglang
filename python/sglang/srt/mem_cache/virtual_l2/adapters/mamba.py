"""Mamba/DeltaNet adapter including side-state components."""

from .mha import MHAStageAdapter


class MambaStageAdapter(MHAStageAdapter):
    def __init__(self, host_pool, components=None):
        super().__init__(host_pool)
        self.components = tuple(components or ("temporal", "conv", "side_state"))
