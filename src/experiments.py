import logging
import click

from typing import Any

import wandb

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

import src.utils as ut

from src.data import Data
from src.dstruct import NOTEARS, lit_NOTEARS



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
    
    ut.logger.info(f"Starting: \n * {count} experiments\n * with following models: {list(models.keys())}")
    for exp_id in range(count):
        D = Data(**data_config)
        D.setup()
        ut.logger.info(f"Started exp #{exp_id}")

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
            
            ut.logger.info(f"Training {m} for exp #{exp_id}")
            trainer.fit(model, datamodule=D)
            trainer.test(model, datamodule=D)
            ut.logger.info(f"Finished {m} for exp #{exp_id}")

            wandb.finish()
            wb_logger.finalize('success')
        ut.logger.info(f"Finished exp #{exp_id}")
    ut.logger.info(f"Finished.")


@click.command()
@click.option("--d", type=int, default=5, help="Amount of variables")
@click.option("--n", type=int, default=200, help="Sample size")
@click.option("--s", type=int, default=9, help="Expected number of edges in the simulated DAG")
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
