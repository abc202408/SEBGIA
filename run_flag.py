import os
import torch
import argparse
import utils
from model import SEBGIA
from defense.FLAG import FLAG


def run_flag(dataset_name=None, alpha=5, sampling=False, batch_size=3072):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='ogbproducts',
                        choices=['ogbarxiv', 'ogbproducts', 'reddit', 'computers', 'photo'], help='dataset')
    parser.add_argument('--gnn_epochs', type=int, default=500, help='The traversal number of gnn model.')
    parser.add_argument('--epochs', type=int, default=500, help='The traversal number of backdoor.')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--layer_norm_first', default=True, action="store_true")
    parser.add_argument('--use_ln', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--surrogate_model', type=str, default='PrSGC', choices=['PrSGC', 'GCN'])

    args = parser.parse_args()

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    utils.set_seed(args.seed)

    if dataset_name is not None:
        dataset = dataset_name
    else:
        dataset = args.dataset

    print(f'=== loading {dataset} dataset ===')
    data = utils.load_dataset(dataset)

    edge_index, x, y = data.edge_index, data.x, data.y
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    agent = SEBGIA(edge_index, x, y, idx_train, idx_val, idx_test, alpha=alpha, sampling=sampling,
                   batch_size=batch_size, surrogate_model=args.surrogate_model, lr=0.01, epochs=args.epochs,
                   device=device)
    millisecond = agent.train()

    print('train test gnn model', '=>' * 30)

    model = FLAG(x.shape[1], args.nhid, y.max().item() + 1, args.num_layers, idx_train, idx_val, idx_test,
                 args.step_size, args.m, args.dropout, device=device, layer_norm_first=args.layer_norm_first,
                 use_ln=args.use_ln, gnn_epochs=args.gnn_epochs).to(device)
    model.fit(edge_index, x, y)

    print('model test before attack', '=>' * 30)
    clean_acc = model.test()

    idx_atk = data.idx_test
    poisoned_edge_index, poisoned_x, poisoned_y = agent.get_poisoned_graph(idx_atk)

    logits = model.predict(poisoned_x, poisoned_edge_index)

    print('after attack', '=>' * 30)
    acc = float(utils.accuracy(logits[idx_test], poisoned_y[idx_test]))
    mis_rate = round(1 - acc, 4)
    print(f'the accuracy after attack: {acc}')
    print(f'the misclassification rate: {mis_rate}')
    print(f'the run time: {millisecond}')

    return mis_rate, clean_acc, millisecond


if __name__ == '__main__':
    run_flag()
