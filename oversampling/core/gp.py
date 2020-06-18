from typing import List, NamedTuple
import numpy as np

class GaussianParams(NamedTuple):
    sample_ids: List[int]            # index of all samples in this gaussian cluster
    mean: np.ndarray                 # gaussian mean
    std: np.ndarray                  # gaussian (co) variance

    @property
    def n(self):
        return len(self.sample_ids)
