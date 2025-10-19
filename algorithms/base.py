'''
Pruning Data Offline
'''
from abc import abstractmethod


class BaseOffDataPruner():

    def __init__(self, dataset):

        self.dataset = dataset

    @abstractmethod
    def prune(self, ratio):
        pass