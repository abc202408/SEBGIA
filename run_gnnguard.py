import os
import argparse
import utils
import torch
from model import SEBGIA
from defense.GNNGuard import EGCNGuard
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import RandomNodeSplit
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import to_undirected


def get_sparse_tensor(edge_idx, n):
    row = edge_idx[0]
    col = edge_idx[1]
    sp_edge_idx = SparseTensor(row=row, col=col, value=torch.ones(col.size(0)), sparse_sizes=torch.Size((n, n)),
                               is_sorted=True)
    return sp_edge_idx


def get_optimal_alpha(dataset, alpha=5):
    if dataset == 'ogbarxiv':
        alpha = 5
    elif dataset == 'reddit':
        alpha = 20
    elif dataset == 'ogbproducts':
        alpha = 5

    return alpha


def run_gnnguard(dataset_name=None, k_hop=2, alpha=None, sampling=True, batch_size=4096):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='ogbproducts',
                        choices=['ogbarxiv', 'ogbproducts', 'reddit', 'computers', 'photo'], help='dataset')
    parser.add_argument('--gnn_epochs', type=int, default=500, help='The traversal number of gnn model.')
    parser.add_argument('--epochs', type=int, default=500, help='The traversal number of backdoor.')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--layer_norm_first', default=True, action="store_true")
    parser.add_argument('--use_ln', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--surrogate_model', type=str, default='PrSGC', choices=['PrSGC', 'GCN'])

    args = parser.parse_args()

    # cuda_id = torch.cuda.device_count() - 1
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    utils.set_seed(args.seed)

    if dataset_name is not None:
        dataset = dataset_name
    else:
        dataset = args.dataset

    print(f'=== loading {dataset} dataset ===')

    data = utils.load_dataset(dataset)
    # Obtain the optimal alpha for each dataset
    if alpha is None:
        alpha = get_optimal_alpha(dataset)

    edge_index, x, y = data.edge_index, data.x, data.y
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    sp_edge_index = get_sparse_tensor(edge_index, x.shape[0]).to(device)

    agent = SEBGIA(edge_index, x, y, idx_train, idx_val, idx_test, k_hop=k_hop, alpha=alpha, sampling=sampling,
                   batch_size=batch_size, surrogate_model=args.surrogate_model, lr=0.01, epochs=args.epochs,
                   device=device)
    millisecond = agent.train()

    model = EGCNGuard(x.shape[1], args.nhid, data.y.max().item() + 1, args.num_layers, idx_train, idx_val, idx_test,
                      args.dropout, device, layer_norm_first=args.layer_norm_first,
                      use_ln=args.use_ln).to(device)
    model.fit(sp_edge_index, x, y, train_iters=args.gnn_epochs)

    print('model test before attack', '=>' * 30)
    clean_acc = model.test()

    idx_test = data.idx_test
    poisoned_edge_index, poisoned_x, poisoned_y = agent.get_poisoned_graph(idx_test)

    poisoned_edge_index = poisoned_edge_index.to('cpu')
    sp_poisoned_edge_index = get_sparse_tensor(poisoned_edge_index, poisoned_x.shape[0]).to(device)

    logits = model.predict(poisoned_x, sp_poisoned_edge_index)

    print('after attack', '=>' * 30)
    acc = float(utils.accuracy(logits[idx_test], poisoned_y[idx_test]))
    mis_rate = round(1 - acc, 4)
    print(f'the accuracy after attack: {acc}')
    print(f'the misclassification rate: {mis_rate}')
    print(f'the run time: {millisecond}')

    return mis_rate, clean_acc, millisecond


if __name__ == '__main__':
    run_gnnguard()
