import torch
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def KL(alpha, c):
    beta = torch.ones((1, c))
    if torch.cuda.is_available():
        beta = beta.cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rand_prop(features, drop_rate, training):
    n = features.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        if torch.cuda.is_available():
            features = masks.cuda() * features
        else:
            features = masks * features
    else:
        features = features * (1. - drop_rate)
    return features


def Norm(x, min=0):
    x = x.detach().cpu().numpy()
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = torch.tensor(scaler.fit_transform(x))
    if torch.cuda.is_available():
        norm_x = norm_x.cuda()

    return norm_x


def MessagePro(data, norm_A, K):
    X_list = []
    X_list.append(Norm(data))
    for _ in range(K):
        X_list.append(torch.spmm(norm_A, X_list[-1]))
    return X_list


def add_noise(data, mu, sigma, mask):
    data = data.detach().cpu().numpy()
    n, m = data[mask].shape
    noise = np.random.normal(mu, sigma, size=(n, m))
    data[mask] += noise
    if torch.cuda.is_available():
        data = torch.FloatTensor(data).cuda()
    else:
        data = torch.FloatTensor(data)
    return data


def MessagePro2(dataset, K, args):
    data, norm_A, mask = dataset.x, dataset.adj, dataset.test_mask
    X_list = []
    X_list.append(Norm(data))
    noise_list = []
    noise_list.append(Norm(add_noise(data, args.mu, args.sigma, mask)))
    for _ in range(K):
        X_list.append(torch.spmm(norm_A, X_list[-1]))
        noise_list.append(torch.spmm(norm_A, noise_list[-1]))
    dataset.X_list = X_list
    dataset.noise_list = noise_list
    return dataset


def Diss(alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    evidence = alpha - 1
    belief = evidence / S
    term_Bal = belief.unsqueeze(2) * Bal(belief.unsqueeze(1), belief.unsqueeze(2))
    term_Bal = torch.sum(term_Bal, dim=1) - belief * Bal(belief, belief)
    term_bj = torch.sum(belief, dim=1).unsqueeze(-1) - belief + 1e-7
    dis_un = belief * (term_Bal / term_bj)
    return dis_un.sum(axis=1)


def getDisn(alpha):
    evi = alpha - 1
    s = torch.sum(alpha, axis=1, keepdims=True)
    blf = evi / s
    idx = np.arange(alpha.shape[1])
    diss = 0
    Bal = lambda bi, bj: 1 - torch.abs(bi - bj) / (bi + bj + 1e-7)
    for i in idx:
        score_j_bal = [blf[:, j] * Bal(blf[:, j], blf[:, i]) for j in idx[idx != i]]
        score_j = [blf[:, j] for j in idx[idx != i]]
        diss += blf[:, i] * sum(score_j_bal) / (sum(score_j) + 1e-7)
    return diss


def get_dissonance(alpha):
    evidence = alpha - 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    belief = evidence / S
    belief_k = belief.unsqueeze(-1)  # [batch size, num classes, 1]
    belief_j = belief.unsqueeze(1)  # [batch size, 1, num classes]
    balances = 1 - torch.abs(belief_k - belief_j) / (belief_k + belief_j + 1e-7)  # Symmetric
    zero_diag = torch.ones_like(balances[0])
    zero_diag.fill_diagonal_(0)
    balances *= zero_diag.unsqueeze(0)  # Set diagonal as 0
    diss_numerator = (belief.unsqueeze(1) * balances).sum(dim=-1)  # [batch size, num classes]
    diss_denominator = belief.sum(dim=-1, keepdim=True) - belief + 1e-7  # [batch size, num classes]
    diss = (belief * diss_numerator / diss_denominator).sum(dim=-1)
    return diss


def Bal(b_i, b_j):
    bb = b_i + b_j + 1e-7
    result = 1 - torch.abs(b_i - b_j) / bb
    return result


def has_nan(tensor):
    return torch.isnan(tensor).any()


def ce_loss(p, alpha, c):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # print('A', A.mean())
    return A


def mse_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1.
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(m * (1 - m) / (S + 1), dim=1, keepdim=True)
    #     print('cls',(A+B).mean())
    return A + B


def KLL(p, E, c, kl, dis):
    alpha = E + 1.0
    label = F.one_hot(p, num_classes=c)
    alp = E * (1 - label) + 1
    C = kl * KL(alp, c)
    D = dis * get_dissonance(alpha).unsqueeze(-1)
    # print('C', KL(alp, c).mean())
    # print('D', get_dissonance(alpha).mean())
    return C + D


def normalize_adj(adj):
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    # Compute degree matrix
    rowsum = np.array(adj.sum(1))
    # Compute D^{-1/2}
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # Compute D^{-1/2}AD^{-1/2}
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
