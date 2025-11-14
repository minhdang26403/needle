from typing import Any

import numpy as np

from needle.data import Dataset


class NDArrayDataset(Dataset):
    def __init__(self, *arrays: np.ndarray) -> None:
        self.arrays = arrays

    def __len__(self) -> Any:
        return self.arrays[0].shape[0]

    def __getitem__(self, i: slice | int) -> tuple[np.ndarray, ...]:
        return tuple(a[i] for a in self.arrays)
