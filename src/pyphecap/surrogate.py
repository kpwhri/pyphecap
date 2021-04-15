from collections import MutableSequence
from dataclasses import dataclass


@dataclass(init=False)
class Surrogate:
    variable_names: list[str]
    lower_cutoff: int = 1
    upper_cutoff: int = 10

    def __init__(self, *variable_names, lower_cutoff=1, upper_cutoff=10):
        self.variable_names = list(variable_names)
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff


class Surrogates(MutableSequence):

    def __init__(self, *surrogates: Surrogate):
        self.data = list(surrogates)
        self._labels = {var for surrogate in surrogates for var in surrogate.variable_names}

    @property
    def variable_names(self):
        return list(self._labels)

    def insert(self, index: int, value: Surrogate) -> None:
        self.data.insert(index, value)

    def __getitem__(self, i):
        return self.data.__getitem__(i)

    def __setitem__(self, i, o: Surrogate) -> None:
        self.data.__setitem__(i, o)

    def __delitem__(self, i) -> None:
        self.data.__delitem__(i)

    def __len__(self) -> int:
        return len(self.data)
