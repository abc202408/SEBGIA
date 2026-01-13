import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn.conv import APPNP as APPNPModel


class APPNP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, K=10, alpha=0.1, dropout=0.5, lr=0.01,
                 weight_decay=5e-4, device=None):
        super(APPNP, self).__init__()
        # Two-layer MLP for prediction

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr

        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)

        # APPNP Propagation Layer
        # K: Number of iterations
        # alpha: Teleport probability
        self.prop = APPNPModel(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        # Predict (MLP)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # Propagate (APPNP)
        x = self.prop(x, edge_index)

        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def fit(self, x, edge_index, y, idx_train, idx_val, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        """Train the GAT model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        if initialize:
            self.initialize()

        # self.data = pyg_data[0].to(self.device)
        self.x = x.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.y = y.to(self.device)
        self.idx_train = idx_train.to(self.device)
        self.idx_val = idx_val.to(self.device)

        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        x = self.x
        edge_index = self.edge_index
        labels = self.y
        idx_train = self.idx_train
        idx_val = self.idx_val
        # idx_train, idx_val = self.data.idx_train, self.data.idx_val

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(
                i, best_loss_val))
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GAT performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        labels = self.y
        output = self.forward(self.x, self.edge_index)
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self, x, edge_index):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GAT
        """

        self.eval()
        return self.forward(x, edge_index)
