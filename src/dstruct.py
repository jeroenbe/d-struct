from abc import abstractmethod
from re import X
from typing import Tuple, Any

import click, logging

import wandb

import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from src.dsl import NotearsMLP
import src.utils as ut
from src.data import Data

logger = logging.getLogger("My_app")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

ch.setFormatter(ut.CustomFormatter())

logger.addHandler(ch)

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
            w_threshold: float=.3,
        ):
        super().__init__()
        
        self.model = model
        self.h = np.inf

        self.h_tol, self.rho_max = h_tol, rho_max
        self.w_threshold = w_threshold

        # We need a way to cope with NOTEARS dual
        #   ascent strategy.
        self.automatic_optimization=False

        self.save_hyperparameters()



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

        self.log('h', h, on_step=True, logger=True, prog_bar=True)
        self.log('rho', rho, on_step=True, logger=True, prog_bar=True)
        self.log('alpha', alpha, on_step=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return ut.LBFGSBScipy(self.model.parameters())

    def A(self):
        B_est = self.model.notears.fc1_to_adj()
        B_est[np.abs(B_est) < self.w_threshold] = 0
        B_est[B_est > 0] = 1
        return B_est

    def test_step(self, batch, batch_idx) -> Any:
        B_est = self.A()
        B_true = self.trainer.datamodule.DAG

        self.log_dict(ut.count_accuracy(B_true, B_est))



class DStruct:
    def __init__(self, P: np.ndarray, dim: int, dsl: NOTEARS):
        self.P = P
        self.dim = dim
        
        self.As = {
            i: dsl(dim=dim) for i in range(len(P))
        }
    
    def setup(self):
        pass

model_refs = {
    'notears-mlp': lit_NOTEARS,
}

def experiment(
        count: int=10,
        models: dict={                          # Options: 'notears-mlp', 'notears-sob', 'dstruct-nt-mlp', 'dstruct-nt-sob'
            'notears-mlp': {                    # Structure: {
                'model': {},                    #   <model-name>: {
                'train': {}                     #       'model': {<KWArgs for model>},
            }                                   #       'train': {<KWArgs for train>},
        },                                      #   }
                                                # }
        data_config: dict={},                   # KWArgs for Data DataModule
        wb_name: str='synth',                   # subdiv for the w&b project
    ) -> Any:
    
    logger.info(f"Starting: \n * {count} experiments\n * with following models: {list(models.keys())}")
    for exp_id in range(count):
        D = Data(**data_config)
        D.setup()
        logger.info(f"Started exp #{exp_id}")

        for m, config in models.items():
            assert m in model_refs.keys(), f"{m} not yet implemented, please choose from: {list(model_refs.keys())}"

            model = model_refs[m](**config['model'])

            wb_logger = WandbLogger(
                project=f"d-struct-{wb_name}",
                group=m,
                config={
                    'data-id': D._id,
                    'exp-id': exp_id
                })
            wb_logger.watch(model, log="all")

            trainer = pl.Trainer(
                logger=wb_logger,
                log_every_n_steps=1,
                **config['train'])
            
            logger.info(f"Training {m} for exp #{exp_id}")
            trainer.fit(model, datamodule=D)
            trainer.test(model, datamodule=D)
            logger.info(f"Finished {m} for exp #{exp_id}")

            wandb.finish()
            wb_logger.finalize('success')
        logger.info(f"Finished exp #{exp_id}")
    logger.info(f"Finished.")


@click.command()
@click.option("--d", type=int, default=5)
@click.option("--n", type=int, default=200)
@click.option("--s", type=int, default=9)
@click.option("--graph_type", 
    type=click.Choice(['ER', 'SF', 'BP'], case_sensitive=False), 
    default='ER', 
    show_default=True,
    help="ER: Erdos-Renyi, SF: Scale Free, BP: BiPartite")
@click.option("--sem_type", 
    type=click.Choice(['mim', 'mlp', 'gp', 'gp-add'], case_sensitive=False),
    default="mim",
    show_default=True,
    help="mim: Index Model, mlp: Multi-Layered Perceptron, gp: Gaussian Process, gp-add: Additive Gaussian Process ")
@click.option("--epochs", type=int, default=100, show_default=True)
@click.option("--batch_size", type=int, default=256, show_default=True)
@click.option("--seed", type=int, default=None, show_default=True)
@click.option("--nt-h_tol", type=float, default=1e-8, help="MINIMUM h value for NOTEARS.")
@click.option("--nt-rho_max", type=float, default=1e+16, help="MAXIMUM value for rho in dual ascent NOTEARS.")
def main(
        d,
        n,
        s,
        graph_type,
        sem_type,
        epochs,
        batch_size,
        seed,
        nt_h_tol,
        nt_rho_max,
    ):

    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    if seed is not None:
        pl.seed_everything(seed)

    experiment(
        count=3,
        data_config={
            'dim': d,
            's0': s,
            'N': n*2,
            'train_size_ratio': .5,
            'sem_type': sem_type,
            'dag_type': graph_type,
            'batch_size': batch_size,
        },
        models= {
            'notears-mlp': {
                'model': {
                    'model': NOTEARS(dim=d),
                    'h_tol': nt_h_tol,
                    'rho_max': nt_rho_max
                },
                'train': {
                    'max_epochs': epochs,
                    'callbacks': [
                        EarlyStopping(monitor='h', stopping_threshold=nt_h_tol),
                        EarlyStopping(monitor='rho', stopping_threshold=nt_rho_max)
                    ]
                }
            }
        }
    )

if __name__ == '__main__':
    main()
