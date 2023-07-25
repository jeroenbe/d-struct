from typing import Any

import click
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tabulate import tabulate

import src.utils as ut
import wandb
from src.daggnn import DSTRUCT_DAG_GNN, lit_DAG_GNN
from src.data import BetaP, Data
from src.dstruct import NOTEARS, DStruct, lit_NOTEARS

model_refs = {
    "notears-mlp": lit_NOTEARS,
    "notears-sob": lit_NOTEARS,
    "dstruct-mlp": DStruct,
    "dstruct-sob": DStruct,
    "dag-gnn": lit_DAG_GNN,
    "dstruct-dag-gnn": DSTRUCT_DAG_GNN,
}


def experiment(
    count: int = 10,
    models: dict = {  # Options: 'notears-mlp', 'notears-sob', 'dstruct-nt-mlp', 'dstruct-nt-sob'
        "notears-mlp": {  # Structure: {
            "model": {},  # <model-name>: {
            "train": {},  # 'model': {<KWArgs for model>},
        }  # 'train': {<KWArgs for train>},
    },  # }
    # }
    data_config: dict = {},  # KWArgs for Data DataModule
    wb_name: str = "test_experiment",  # subdiv for the w&b project
) -> Any:
    experiment_summary = f"""
    Starting following setup:
    --> {count} experiments
    --> with following models: {list(models.keys())}
    """

    ut.logger.info(f"\n{tabulate([[experiment_summary]], tablefmt='grid')}\n")

    datasets = []

    # sample DAGS first to ensure we always sample the same DAGS for the same seed
    for _ in range(count):
        Dataset = Data(**data_config)
        datasets.append(Dataset)

    i = 0
    for exp_id in range(count):
        D = datasets[i]
        D.setup()

        ut.logger.info(f"Started exp #{exp_id}")

        for m, config in models.items():
            assert (
                m in model_refs.keys()
            ), f"{m} not yet implemented, please choose from: {list(model_refs.keys())}"

            model = model_refs[m](**config["model"])

            wb_logger = WandbLogger(
                project=f"d-struct-{wb_name}",
                group=m,
                config={"data-id": D._id, "exp-id": exp_id},
            )
            wb_logger.watch(model, log="all")

            trainer = pl.Trainer(
                logger=wb_logger, log_every_n_steps=1, **config["train"]
            )

            ut.logger.info(f"Training [{m}] for exp #{exp_id}")
            trainer.fit(model, datamodule=D)
            trainer.test(model, datamodule=D)
            ut.logger.info(f"Finished {m} for exp #{exp_id}")

            wandb.finish()
            wb_logger.finalize("success")
        ut.logger.info(f"Finished exp #{exp_id}")
        i += 1
    ut.logger.info("Finished.")


@click.command()
@click.option("--d", type=int, default=5, help="Amount of variables")
@click.option("--n", type=int, default=200, help="Sample size")
@click.option(
    "--s", type=int, default=9, help="Expected number of edges in the simulated DAG"
)
@click.option("--K", type=int, default=5, help="Amount of subsets for D-Struct")
@click.option(
    "--graph_type",
    type=click.Choice(["ER", "SF", "BP"], case_sensitive=False),
    default="ER",
    show_default=True,
    help="ER: Erdos-Renyi, SF: Scale Free, BP: BiPartite",
)
@click.option(
    "--sem_type",
    type=click.Choice(["mim", "mlp", "gp", "gp-add"], case_sensitive=False),
    default="mim",
    show_default=True,
    help="mim: Index Model, mlp: Multi-Layered Perceptron, gp: Gaussian Process, gp-add: Additive Gaussian Process ",
)
@click.option("--epochs", type=int, default=100, show_default=True)
@click.option("--batch_size", type=int, default=256, show_default=True)
@click.option("--seed", type=int, default=None, show_default=True)
@click.option("--lmbda", type=int, default=3, show_default=True)
@click.option(
    "--nt-h_tol", type=float, default=1e-8, help="MINIMUM h value for NOTEARS."
)
@click.option(
    "--nt-rho_max",
    type=float,
    default=1e16,
    help="MAXIMUM value for rho in dual ascent NOTEARS.",
)
@click.option("--sort", type=bool, default="False", help="bool - sort batches.")
@click.option(
    "--rand_sort", type=bool, default="False", help="bool - random sort batches."
)
@click.option(
    "--experiment_count", type=int, default=5, help="Amount of seq. experiments to run."
)
def main(
    d,
    n,
    s,
    k,
    graph_type,
    sem_type,
    epochs,
    lmbda,
    batch_size,
    seed,
    nt_h_tol,
    nt_rho_max,
    sort,
    rand_sort,
    experiment_count,
):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    if seed is not None:
        pl.seed_everything(seed)

    experiment(
        count=experiment_count,
        data_config={
            "dim": d,
            "s0": s,
            "N": n * 2,
            "train_size_ratio": 0.5,
            "sem_type": sem_type,
            "dag_type": graph_type,
            "batch_size": batch_size,
        },
        models={
            # "dstruct-dag-gnn": {
            #     "model": {
            #         "dim": d,
            #         "n": n,
            #         "dsl": DAG_GNN,
            #         "dsl_config": {
            #             "dim": d,
            #             "n": n,
            #         },
            #         "p": BetaP(k, bool(sort), bool(rand_sort)),
            #         "K": k,
            #         "lmbda": lmbda,
            #         "s": s,
            #     },
            #     "train": {
            #         "max_epochs": epochs,
            #     }
            # },
            # "dstruct-sob": {
            #     "model": {
            #         "dim": d,
            #         "dsl": NOTEARS,
            #         "dsl_config": {"dim": d, "sem_type": "sob"},
            #         "h_tol": nt_h_tol,
            #         "rho_max": nt_rho_max,
            #         "p": BetaP(k, bool(sort), bool(rand_sort)),
            #         "K": k,
            #         "lmbda": lmbda,
            #         "n": n,
            #         "s": s,
            #         "dag_type": graph_type,
            #     },
            #     "train": {
            #         "max_epochs": epochs,
            #         "callbacks": [
            #             EarlyStopping(monitor="h", stopping_threshold=nt_h_tol),
            #             EarlyStopping(monitor="rho", stopping_threshold=nt_rho_max),
            #         ],
            #     },
            # },
            "dstruct-mlp": {
                "model": {
                    "dim": d,
                    "dsl": NOTEARS,
                    "dsl_config": {"dim": d},
                    "h_tol": nt_h_tol,
                    "rho_max": nt_rho_max,
                    "p": BetaP(k, bool(sort), bool(rand_sort)),
                    "K": k,
                    "lmbda": lmbda,
                    "n": n,
                    "s": s,
                    "dag_type": graph_type,
                },
                "train": {
                    "max_epochs": epochs,
                    "callbacks": [
                        EarlyStopping(monitor="h", stopping_threshold=nt_h_tol),
                        EarlyStopping(monitor="rho", stopping_threshold=nt_rho_max),
                    ],
                },
            },
            "notears-mlp": {
                "model": {
                    "model": NOTEARS(dim=d, sem_type="mlp"),
                    "h_tol": nt_h_tol,
                    "rho_max": nt_rho_max,
                    "n": n,
                    "s": s,
                    "dim": d,
                    "K": k,
                    "dag_type": graph_type,
                },
                "train": {
                    "max_epochs": epochs,
                    "callbacks": [
                        EarlyStopping(monitor="h", stopping_threshold=nt_h_tol),
                        EarlyStopping(monitor="rho", stopping_threshold=nt_rho_max),
                    ],
                },
            },
            # "notears-sob": {
            #     "model": {
            #         "model": NOTEARS(dim=d, sem_type='sob'),
            #         "h_tol": nt_h_tol,
            #         "rho_max": nt_rho_max,
            #         "n": n,
            #         "s": s,
            #         "dim": d,
            #         "K": k,
            #         "dag_type": graph_type,
            #     },
            #     "train": {
            #         "max_epochs": epochs,
            #         "callbacks": [
            #             EarlyStopping(monitor="h", stopping_threshold=nt_h_tol),
            #             EarlyStopping(monitor="rho", stopping_threshold=nt_rho_max),
            #         ],
            #     },
            # },
            # "dag-gnn": {
            #     "model": {
            #         "dim": d,
            #         "n": n,
            #     },
            #     "train": {
            #         "max_epochs": epochs
            #     }
            # }
        },
    )


if __name__ == "__main__":
    main()
