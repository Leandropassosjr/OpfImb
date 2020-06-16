from typing import Tuple, List

import numpy as np
from sklearn.utils import shuffle


def join_synth(orig_x: np.ndarray,
               orig_y: np.ndarray,
               synth_x: List[np.ndarray],
               synth_label: int) -> Tuple[np.ndarray, np.ndarray]:
    """Concat and shuffle original dataset with synth samples."""
    filling_y = np.ones(synth_x.shape[0]) * synth_label

    increased_x = np.concatenate([orig_x, synth_x])
    increased_y = np.hstack([orig_y, filling_y])
    return shuffle(increased_x, increased_y)


def concat_shuffle(x, y) -> Tuple[np.ndarray, np.ndarray]:
    return shuffle(np.concatenate(x), np.concatenate(y).astype(int))


def separate(X: np.ndarray, y: np.ndarray, min_label: int):
    min_x = X[y == min_label]
    min_y = y[y == min_label]

    max_x = X[y != min_label]
    max_y = y[y != min_label]
    return min_x, min_y, max_x, max_y
