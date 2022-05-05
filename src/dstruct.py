from abc import abstractmethod
from re import X
from typing import Tuple, Any

import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.dsl import NotearsMLP
import src.utils as ut
from src.data import Data



class NOTEARS(nn.Module):
    def __init__(
            self, 
            dim: int,                           # Dims of system
            nonlinear_dims: list=[10, 10, 1],   # Dims for non-linear arch
            rho: float=1.0,                     # NOTEARS parameters
            alpha: float=1.0,                   #   |
            lambda1: float=.0,                  #   |
            lambda2: float=.0,                  #   |

        ):
        super().__init__()
        
        self.dim = dim
        self.notears = NotearsMLP(dims=[dim, *nonlinear_dims])

        self.rho = rho
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2


    def _squared_loss(self, x, x_hat):
        n = x.shape[0]
        return .5 / n * torch.sum((x_hat - x) ** 2)

    def h_func(self):
        return self.notears.h_func()

    def loss(self, x, x_hat):
        loss = self._squared_loss(x, x_hat)
        h_val = self.notears.h_func()
        penalty = .5 * self.rho * h_val * h_val + self.alpha * h_val
        l2_reg = .5 * self.lambda2 * self.notears.l2_reg()
        l1_reg = self.lambda1 * self.notears.fc1_l1_reg()
        
        return loss + penalty + l2_reg + l1_reg


    def forward(self, x: torch.Tensor):
        x_hat = self.notears(x)
        loss = self.loss(x, x_hat)

        return x_hat, loss


class lit_NOTEARS(pl.LightningModule):

    def __init__(
            self,
            model: NOTEARS,
            h_tol: float=1e-8,
            rho_max: float=1e+16,
        ):
        super().__init__()
        
        self.model = model
        self.h = np.inf

        self.h_tol, self.rho_max = h_tol, rho_max

        # We need a way to cope with NOTEARS dual
        #   ascent strategy.
        self.automatic_optimization=False

        self.save_hyperparameters()


    def on_trian_batch_start(self, batch, batch_idx) -> int:
        if self.h <= self.h_tol or self.model.rho >= self.rho_max:
            return -1
        return 1



    def _dual_ascent_step(self, x, optimizer: torch.optim.Optimizer, h: float) -> Tuple[float]:
        h_new = None

        while self.model.rho < self.rho_max:
            def closure():
                optimizer.zero_grad()
                _, loss = self.model(x)
                self.manual_backward(loss)
                return loss
            
            optimizer.step(closure)

            with torch.no_grad():
                h_new = self.model.h_func().item()
            if h_new > .25 * self.h:
                self.model.rho *= 10
            else:
                break
        self.model.alpha += self.model.rho * h_new
        return self.model.alpha, self.model.rho, h_new
        

    def training_step(self, batch, batch_idx) -> Any:
        opt = self.optimizers()

        (X,) = batch

        alpha, rho, h = self._dual_ascent_step(X, opt, self.h)
        self.h = h

        self.logger.experiment.log({
            'h': h, 'alpha': alpha, 'rho': rho
        })

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return ut.LBFGSBScipy(self.model.parameters())


class DStruct:
    def __init__(self, P: np.ndarray, dim: int, dsl: NOTEARS):
        self.P = P
        self.dim = dim
        
        self.As = {
            i: dsl(dim=dim) for i in range(len(P))
        }
    
    def setup(self):
        pass


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    pl.seed_everything(123)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    D = Data(dim=d, s0=s0, N=n, sem_type=sem_type, dag_type=graph_type, batch_size=n)
    D.setup()
    nt = NOTEARS(dim=D.dim)
    NT = lit_NOTEARS(model=nt)

    wb_logger = WandbLogger(project='d-struct-notears')
    wb_logger.watch(NT, log='all')

    trainer = pl.Trainer(NT, datamodule=D, logger=wb_logger)
    trainer.fit()

    



if __name__ == '__main__':
    main()
    

    
