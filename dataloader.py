import torch
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB
from torch_geometric.data import Data

from lossfunc import normalize_adj


def get_dataset(ds, seed):
    if ds in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/Planetoid/', name=ds)
        dataset.x = dataset[0].x
        dataset.y = dataset[0].y
        dataset.edge_index = dataset[0].edge_index
        dataset.train_mask = dataset[0].train_mask
        dataset.val_mask = dataset[0].val_mask
        dataset.test_mask = dataset[0].test_mask
    elif ds in ['Computers', 'Photo']:
        dataset = Amazon(root='./data/Amazon/', name=ds)
        # Split the edges into train, validation and test sets
        dataset = set_train_val_test_split(seed, dataset)
    else:
        raise Exception('Unknown dataset.')
    data, edge_index, target = dataset.x, dataset.edge_index, dataset.y
    num_nodes = data.shape[0]
    adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), edge_index), shape=(num_nodes, num_nodes))
    # Normalize the adjacency matrix
    adj = normalize_adj(adj)
    # Convert the normalized adjacency matrix back to a PyTorch tensor
    adj = torch.FloatTensor(adj.toarray())
    if torch.cuda.is_available():
        adj = adj.cuda()
        data = data.cuda()
    dataset.adj = adj
    dataset.target = target
    dataset.x = data
    return dataset


import numpy as np

def set_train_val_test_split(seed: int, data: Data) -> Data:
    rnd_state = np.random.RandomState(seed)

    data.y = data[0].y
    data.x = data[0].x
    data.edge_index = data[0].edge_index
    num_nodes = data.y.shape[0]

    # Number of nodes for each class in the training and validation sets
    num_train_per_class = 20
    num_val_per_class = 30

    # Create masks for test, validation, and train sets
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Randomly shuffle the nodes using rnd_state
    perm = rnd_state.permutation(num_nodes)

    # Assign train nodes for each class
    train_mask_per_class = []
    val_mask_per_class = []
    for c in range(data.y.max() + 1):
        class_nodes = perm[np.where(data.y[perm] == c)[0]]
        train_mask_per_class.extend(class_nodes[:num_train_per_class])
        val_mask_per_class.extend(class_nodes[num_train_per_class:num_train_per_class + num_val_per_class])

    # Assign remaining nodes as test nodes
    test_mask[:] = True
    test_mask[torch.cat((torch.tensor(train_mask_per_class), torch.tensor(val_mask_per_class)))] = False

    # Assign train and validation nodes
    train_mask[train_mask_per_class] = True
    val_mask[val_mask_per_class] = True

    # Set the masks to the dataset's split_idx attribute
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def check_mask_overlap(train_mask, val_mask, test_mask):
    train_indices = set(torch.nonzero(train_mask).view(-1).tolist())
    val_indices = set(torch.nonzero(val_mask).view(-1).tolist())
    test_indices = set(torch.nonzero(test_mask).view(-1).tolist())

    if len(train_indices.intersection(val_indices)) > 0:
        print("Overlap between train_mask and val_mask.")
    if len(train_indices.intersection(test_indices)) > 0:
        print("Overlap between train_mask and test_mask.")
    if len(val_indices.intersection(test_indices)) > 0:
        print("Overlap between val_mask and test_mask.")
