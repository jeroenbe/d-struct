# d-struct

MVP run:
```bash
python -m src.experiments --help

# Usage: python -m src.experiments [OPTIONS]

# Options:
#   --d INTEGER                     Amount of variables
#   --n INTEGER                     Sample size
#   --s INTEGER                     Expected number of edges in the simulated
#                                   DAG
#   --graph_type [ER|SF|BP]         ER: Erdos-Renyi, SF: Scale Free, BP:
#                                   BiPartite  [default: ER]
#   --sem_type [mim|mlp|gp|gp-add]  mim: Index Model, mlp: Multi-Layered
#                                   Perceptron, gp: Gaussian Process, gp-add:
#                                   Additive Gaussian Process   [default: mim]
#   --epochs INTEGER                [default: 100]
#   --batch_size INTEGER            [default: 256]
#   --seed INTEGER
#   --nt-h_tol FLOAT                MINIMUM h value for NOTEARS.
#   --nt-rho_max FLOAT              MAXIMUM value for rho in dual ascent
#                                   NOTEARS.
#   --help                          Show this message and exit.
```

Required packages:
```bash
numpy
scipy
scikit-learn
pytorch-lightning
igraph
wandb
click
matplotlib
```
