"""Configuration utilities for BWE model."""


class AttrDict(dict):
    """Dictionary that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
