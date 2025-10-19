import numpy as np

from .base import BaseOffDataPruner


class RandomDP(BaseOffDataPruner):

    def __init__(self, dataset):
        super().__init__(dataset)

    def prune(self, ratio):
        # ratio: the left percentage
        total = len(self.dataset)
        budget = round(total * ratio)

        return np.random.choice(range(total), budget, replace=False)