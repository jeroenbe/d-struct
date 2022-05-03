import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import pytorch_lightning as pl


import src.utils as ut
import src.simulate as sm

class Data(pl.LightningDataModule):
    def __init__(
            self, 
            dim: int=20,                    # Amount of vars
            s0: int=40,                     # Expected amount of edges
            N: int=1000,                    # Amount of samples
            sem_type: str="mim",            # SEM-type (
                                            #   'mim' -> index model,  
                                            #   'mlp' -> multi layer perceptrion, 
                                            #   'gp' -> gaussian process, 
                                            #   'gp-add' -> addative gp)
            dag_type: str="ER",             # Random graph type (
                                            #   'ER' -> Erdos-Renyi, 
                                            #   'SF' -> Scale Free, 
                                            #   'BP' -> BiPartite)
            batch_size: int=32,
            train_size_ratio: float=1.,     # Ratio of train/test split
            loc: str='data/simulated/',     # Save location for simulated data
        ):
        super().__init__()
        
        self.dim = dim
        self.s0 = s0
        self.N = N
        self.sem_type = sem_type
        self.dag_type = dag_type
        
        self.batch_size = batch_size
        self.train_size_ratio = train_size_ratio

        self.loc = Path(loc)

        self.DAG = None
        self._simulate()
        self._sample()


    def _simulate(self) -> None:
        self.DAG = sm.simulate_dag(self.dim, self.s0, self.dag_type)
        self._id = hash(self.DAG.__repr__() + self.DAG.__array_interface__['data'][0].__repr__())

        path = self.loc / str(self._id)
        
        os.makedirs(path, exist_ok=True)

        np.savetxt(path / 'DAG.csv', self.DAG, delimiter=',')


    def _sample(self) -> None:
        assert self.DAG is not None, "No DAG simulated yet"

        self.X = sm.simulate_nonlinear_sem(self.DAG, self.N, self.sem_type)

        np.savetxt(self.loc / str(self._id) / 'X.csv', self.X, delimiter=',')


    def setup(self, stage: Optional[str]=None) -> None:
        assert self.DAG is not None, "No DAG simulated yet"
        assert self.X is not None, "No SEM simulated yet"

        DX = TensorDataset(torch.from_numpy(self.X))

        _train_size = np.floor(self.N * self.train_size_ratio)
        self.train, self.test = random_split(
            DX, [int(_train_size), int(self.N-_train_size)]
        )


    def resample(self) -> None:
        """
        Resamples a new DAG and SEM
        Resets the train and test sets
        Writes new data and DAG to self.loc 
            without overwriting previous data
        """
        self._simulate()
        self._sample()
        self.setup()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test)
    
    def val_dataloader(self) -> DataLoader:
        pass


class P(Data):
    """
    Wrapper for Data class.
    Should be able to:
        - load data from existing Data class (by id)
        - act as Data class on it's own
    """
    def __init__(self, D: Data, N: int):
        self.D = Data
        self.N = N
    
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Data:
        pass

class RandomP(P):
    def __init__(self, D: Data, N: int):
        super().__init__(D=D, N=N)
    
    def __call__(self, *args: Any, **kwds: Any) -> Data:
        weights = np.random.rand(len(self.D))





