# d-struct: _code_

Experiments are organised as bash files:
```bash
k_sweep_ER.sh
k_sweep_SF.sh
n_sweep_ER.sh
n_sweep_SF.sh
s_sweep_ER.sh
s_sweep_SF.sh
```
run a file as:
```bash
bash <file>
```

Run experiments with custom options:
```bash
python -m src.experiments --help

# Usage: python -m src.experiments [OPTIONS]

# Options:
#   --d INTEGER                     Amount of variables
#   --n INTEGER                     Sample size
#   --s INTEGER                     Expected number of edges in the simulated
#                                   DAG
#   --K INTEGER                     Amount of subsets for D-Struct
#   --graph_type [ER|SF|BP]         ER: Erdos-Renyi, SF: Scale Free, BP:
#                                   BiPartite  [default: ER]
#   --sem_type [mim|mlp|gp|gp-add]  mim: Index Model, mlp: Multi-Layered
#                                   Perceptron, gp: Gaussian Process, gp-add:
#                                   Additive Gaussian Process   [default: mim]
#   --epochs INTEGER                [default: 100]
#   --batch_size INTEGER            [default: 256]
#   --seed INTEGER
#   --lmbda INTEGER                 [default: 3]
#   --nt-h_tol FLOAT                MINIMUM h value for NOTEARS.
#   --nt-rho_max FLOAT              MAXIMUM value for rho in dual ascent
#                                   NOTEARS.
#   --sort BOOLEAN                  bool - sort batches.
#   --rand_sort BOOLEAN             bool - random sort batches.
#   --experiment_count INTEGER      Amount of seq. experiments to run.
#   --help                          Show this message and exit.
```

We use python `3.8.6`, with following required packages:
```bash
numpy
scipy
scikit-learn
pytorch-lightning
igraph
wandb
click
matplotlib
tabulate
```
