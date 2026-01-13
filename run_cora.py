import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import os
from deeprobust.graph.utils import get_train_val_test
from model import SEBGIA
import argparse
from models.gcn import GCN
from models.gat import GAT
from models.appnp import APPNP
import utils
from cogdl.datasets.grb_data import GRBDataset
# from ogb.nodeproppred import PygNodePropPredDataset


def run_cora(model_type='gcn', dataset='Cora', ptb_rate=0.01, sampling=True, batch_size=8192):
    """
    model_type: gat, gcn, appnp
    dataset: Cora, PubMed, grb-cora
    ptb_rate: 0.01, 0.03. 0.05
    """
    print('current parameters: ', model_type, dataset, ptb_rate)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--gnn_epochs', type=int, default=500,
                        help='The traversal number of gnn model.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='The traversal number of backdoor.')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--layer_norm_first',
                        default=True, action="store_true")
    parser.add_argument('--use_ln', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--surrogate_model', type=str,
                        default='GCN', choices=['PrSGC', 'GCN', 'SGC'])

    args = parser.parse_args()

    utils.set_seed(21)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 下载数据集
    sep = os.sep
    dataset_dir = f'{os.getcwd()}{sep}dataset{sep}'
    if dataset == 'grb-cora' or dataset == 'grb-citeseer':
        dataset_dir = dataset_dir + dataset + sep
        data = GRBDataset(root=dataset_dir, name='grb-cora')[0]
    # elif dataset == 'ogbn-arxiv':
    #     data = PygNodePropPredDataset(name=dataset, root=dataset_dir)[0]
    #     data.y = data.y.flatten()
    else:
        data = Planetoid(root=dataset_dir, name=dataset)[0]

    x = data.x  # 节点特征 [2708, 1433]
    y = data.y  # 节点标签 [2708]
    edge_index = data.edge_index  # 边的连接 [2, 10556]

    if dataset == 'grb-cora' or dataset == 'grb-citeseer':
        edge_index = torch.stack(edge_index, dim=0)

    idx_train, idx_test, idx_attach = get_train_val_test(data.x.shape[0],
                                                         val_size=0.2,
                                                         test_size=ptb_rate,
                                                         stratify=data.y)

    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_test = torch.tensor(idx_test, dtype=torch.long)
    idx_attach = torch.tensor(idx_attach, dtype=torch.long)

    agent = SEBGIA(edge_index, x, y, idx_train, idx_test, idx_test, k_hop=2, alpha=0.01, sampling=sampling,
                   batch_size=batch_size, surrogate_model=args.surrogate_model, lr=0.01, epochs=args.epochs,
                   device=device)
    agent.train()

    if model_type == 'appnp':
        print(f'The attacked model is APPNP.')
        model = APPNP(nfeat=x.shape[1],
                      nhid=64,
                      nclass=y.max().item() + 1,
                      device=device).to(device)
    elif model_type == 'gat':
        print(f'The attacked model is GAT.')
        model = GAT(nfeat=x.shape[1],
                    nhid=8, heads=8,
                    nclass=y.max().item() + 1,
                    device=device).to(device)
    else:
        print(f'The attacked model is GCN.')
        model = GCN(nfeat=x.shape[1],
                    nhid=64,
                    nclass=y.max().item() + 1,
                    device=device).to(device)

    model.fit(x, edge_index, y, idx_train, idx_test)

    poisoned_edge_index, poisoned_x, poisoned_y = agent.get_poisoned_graph(
        idx_attach)

    logits = model.predict(poisoned_x, poisoned_edge_index)
    print('=>' * 30)
    acc = float(utils.accuracy(logits[idx_test], poisoned_y[idx_test]))
    print(f'the accuracy after attack: {acc}')

    return acc


if __name__ == '__main__':
    run_cora()
