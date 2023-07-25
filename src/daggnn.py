from typing import Any, Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import src.utils as ut
from src.data import P


# Yue et al.
class DAGGNN_MLPEncoder(nn.Module):
    """MLP encoder module."""

    def __init__(
        self,
        n_in,
        n_xdims,
        n_hid,
        n_out,
        adj_A,
        batch_size,
        do_prob=0.0,
        factor=True,
        tol=0.1,
    ):
        super(DAGGNN_MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True)
        )
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(
            torch.ones_like(torch.from_numpy(adj_A)).double()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):
        if torch.sum(self.adj_A != self.adj_A):
            print("nan error \n")

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.0 * self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = ut.preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = self.fc2(H1)
        logits = torch.matmul(x + self.Wa, adj_Aforz) - self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa


# Yue et al.
class DAGGNN_MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(
        self,
        n_in_node,
        n_in_z,
        n_out,
        encoder,
        data_variable_size,
        batch_size,
        n_hid,
        do_prob=0.0,
    ):
        super(DAGGNN_MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
        self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa
    ):
        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = ut.preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(input_z + Wa, adj_A_new1) - Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt


class DAG_GNN(nn.Module):
    def __init__(
        self,
        dim: int,
        n: int,
        tau_A: float = 0.0,  # DAG-GNN params
        lambda_A: float = 0.1,  # |
        c_A: float = 1.0,  # |
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n = n

        self.A = np.zeros((self.dim, self.dim))

        self.tau_A = tau_A
        self.lambda_A = lambda_A
        self.c_A = c_A

        self.encoder = DAGGNN_MLPEncoder(
            n_in=self.dim,
            n_xdims=self.dim,
            n_hid=self.dim,
            n_out=self.dim,
            adj_A=self.A,
            batch_size=256,
        )

        self.decoder = DAGGNN_MLPDecoder(
            n_in_node=self.dim,
            n_in_z=self.dim,
            n_out=self.dim,
            encoder=self.encoder,
            data_variable_size=self.dim,
            batch_size=256,
            n_hid=self.dim,
        )

        off_diag = np.ones([self.dim, self.dim]) - np.eye(self.dim)

        rel_rec = np.array(ut.encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(ut.encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        rel_rec = torch.DoubleTensor(rel_rec)
        rel_send = torch.DoubleTensor(rel_send)

        self.rel_rec = Variable(rel_rec)
        self.rel_send = Variable(rel_send)

        self.prox_plus = torch.nn.Threshold(0.0, 0.0)

        self.triu_indices = ut.get_triu_offdiag_indices(self.dim)
        self.tril_indices = ut.get_tril_offdiag_indices(self.dim)

    def _h_A(self, A, m):
        expm_A = ut.matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    def stau(self, w, tau):
        w1 = self.prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def forward(self, x: torch.Tensor):
        return None

    def enc(self, X):
        return self.encoder(X, self.rel_rec, self.rel_send)

    def dec(self, X, edges, origin_A, adj_A_tilt_encoder, Wa):
        return self.decoder(
            X,
            edges,
            self.dim,
            self.rel_rec,
            self.rel_send,
            origin_A,
            adj_A_tilt_encoder,
            Wa,
        )

    def loss(self, preds, target, variance, logits, graph):
        # reconstruction accuracy loss
        loss_nll = ut.nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = ut.kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = graph  # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = self.tau_A * torch.sum(torch.abs(one_adj_A))

        h_A = self._h_A(graph, self.dim)
        loss += (
            self.lambda_A * h_A
            + 0.5 * self.c_A * h_A * h_A
            + 100.0 * torch.trace(graph * graph)
            + sparse_loss
        )

        return loss


class lit_DAG_GNN(pl.LightningModule):
    def __init__(
        self,
        dim: int,
        n: int,
        tau_A: float = 0.0,
        lambda_A: float = 0.1,
        c_A: float = 1.0,
        lr: float = 0.001,
        lr_decay: float = 30,
        gamma: float = 0.1,
        w_threshold: float = 0.3,
        recursive_dag_search: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.n = n

        self.dag_gnn = DAG_GNN(dim=dim, n=n, tau_A=tau_A, lambda_A=lambda_A, c_A=c_A)

        self.w_threshold = w_threshold
        self.recursive_dag_search = recursive_dag_search

        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma

        self.graph = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.dag_gnn.parameters(),
            lr=self.lr,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_decay,
            gamma=self.gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, batch_idx):
        (X,) = batch
        (
            enc_x,
            logits,
            origin_A,
            adj_A_tilt_encoder,
            z_gap,
            z_positive,
            myA,
            Wa,
        ) = self.dag_gnn.enc(X)
        edges = logits

        (dec_x, output, adj_A_tilt_decoder) = self.dag_gnn.dec(
            X, edges, origin_A, adj_A_tilt_encoder, Wa
        )

        target = X
        preds = output
        variance = 0.0

        loss = self.dag_gnn.loss(
            preds=preds,
            target=target,
            variance=variance,
            graph=origin_A,
            logits=logits,
        )

        self.log("loss", loss)

        self.graph = origin_A.data.clone()
        self.graph.diagonal().zero_()
        self.graph = self.graph.numpy()

        return loss

    def _get_dag(self, As: np.ndarray) -> np.ndarray:
        As_temp = np.abs(As.copy())

        As_temp[np.where(As_temp == As_temp[As_temp > 0].min())] = 0

        intermediate_dag = As_temp.copy()
        intermediate_dag[intermediate_dag > 0] = 1

        if ut.is_dag(intermediate_dag):
            return intermediate_dag
        else:
            return self._get_dag(As_temp)

    def get_A(self, threshold) -> np.ndarray:
        B_est = self.graph.copy()

        if self.recursive_dag_search:
            B_est = self._get_dag(B_est)
        else:
            B_est[np.abs(B_est) <= threshold] = 0
            B_est[np.abs(B_est) > threshold] = 1

        return B_est

    def test_step(self, batch, batch_idx) -> Any:
        for threshold in np.linspace(start=0, stop=1, num=100):
            B_est = self.get_A(threshold)
            if ut.is_dag(B_est):
                print(f"Is DAG for {threshold}")
                self.log_dict({"DAG_threshold": threshold})
                break

        B_true = self.trainer.datamodule.DAG
        print(f"B_est: {B_est}")
        print(f"B_true: {B_true}")
        self.log_dict(ut.count_accuracy(B_true, B_est))


class DSTRUCT_DAG_GNN(pl.LightningModule):
    def __init__(
        self,
        dim: int,
        n: int,
        dsl: Callable,
        dsl_config: dict,
        p: P,
        K: int = 5,
        s: int = 9,
        lmbda: int = 2,
        tau_A: float = 0.0,
        lambda_A: float = 0.1,
        lr: float = 0.001,
        lr_decay: float = 30,
        gamma: float = 0.1,
        w_threshold: float = 0.3,
        recursive_dag_search: bool = True,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.K = K
        self.dim = dim
        self.lmbda = lmbda
        self.s = s
        self.p = p

        self.recursive_dag_search = True

        self.dsl_list = nn.ModuleList([dsl(**dsl_config) for i in range(self.K)])
        self._As = np.array([None for i in range(self.K)])

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizers, schedulers = [], []
        for dsl in self.dsl_list:
            optimizers.append(torch.optim.Adam(dsl.parameters(), lr=self.lr))
            schedulers.append(
                torch.optim.lr_scheduler.StepLR(
                    optimizers[-1], step_size=self.lr_decay, gamma=self.gamma
                )
            )
        optimizers.append(torch.optim.Adam(self.parameters(), lr=self.lr))

        return optimizers, schedulers

    def training_step(self, batch, batch_idx, optimizer_idx) -> Any:
        (X,) = batch
        subsets = self.p(X)

        total_loss = 0

        for i, dsl in enumerate(self.dsl_list):
            subset = subsets[i]

            (
                enc_x,
                logits,
                origin_A,
                adj_A_tilt_encoder,
                z_gap,
                z_positive,
                myA,
                Wa,
            ) = dsl.enc(subset)
            edges = logits

            (dec_x, output, adj_A_tilt_decoder) = dsl.dec(
                subset, edges, origin_A, adj_A_tilt_encoder, Wa
            )

            target = subset
            preds = output
            variance = 0.0

            loss = dsl.loss(
                preds=preds,
                target=target,
                variance=variance,
                graph=origin_A,
                logits=logits,
            )

            total_loss += loss
            self._As[i] = origin_A.data.clone().numpy()

        total_loss += self.lmbda * self._loss(X)

        self.log("loss", total_loss.item())

        return total_loss

    def _loss(self, X):
        As, A_comp = self.forward(X)

        mask = torch.ones(A_comp.shape)
        mask.diagonal().zero_()

        A_comp.detach()
        A_comp.diagonal().zero_()

        loss = 0
        mse = nn.MSELoss()
        for A_est in As:
            loss += mse(A_est * mask, A_comp)

        return loss

    def forward(self, X=None, threshold=0.5, grad: bool = True) -> Any:
        if grad:
            subsets = self.p(X)
            As = tuple([dsl.enc(subsets[i])[2] for i, dsl in enumerate(self.dsl_list)])
            _As = torch.stack(As).mean(dim=0)
        else:
            As = self._As
            _As = As.mean(axis=0)

            if self.recursive_dag_search:
                _As = self._get_dag(_As)
            else:
                _As[np.abs(_As) > threshold] = 1
                _As[np.abs(_As) <= threshold] = 0

        return As, _As

    def _get_dag(self, As: np.ndarray) -> np.ndarray:
        As_temp = np.abs(As.copy())

        As_temp[np.where(As_temp == As_temp[As_temp > 0].min())] = 0

        intermediate_dag = As_temp.copy()
        intermediate_dag[intermediate_dag > 0] = 1

        if ut.is_dag(intermediate_dag):
            return intermediate_dag
        else:
            return self._get_dag(As_temp)

    def A(self, threshold=0.5) -> np.ndarray:
        _, A = self.forward(threshold=threshold, grad=False)
        return A

    def test_step(self, batch, batch_idx) -> Any:
        for threshold in np.linspace(start=0, stop=1, num=100):
            B_est = self.A(threshold)
            if ut.is_dag(B_est):
                print(f"Is DAG for {threshold}")
                self.log_dict({"DAG_threshold": threshold})
                break

        B_true = self.trainer.datamodule.DAG
        print(f"B_est: {B_est}")
        print(f"B_true: {B_true}")
        self.log_dict(ut.count_accuracy(B_true, B_est))
