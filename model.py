import torch
import torch.nn as nn

from layer import MLPLayer
from lossfunc import ce_loss, KLL


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, num_class, input_droprate, dropout, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, num_class)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_mlp = nn.Softplus()
        self.act = nn.Tanh()
        self.input_drop = nn.Dropout(input_droprate)
        self.hidden_drop = nn.Dropout(dropout)

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = self.input_drop(x)
        x = self.act(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_drop(x)
        x = self.act_mlp(self.layer2(x))
        return x


class EFGNN(nn.Module):
    def __init__(self, args):
        super(EFGNN, self).__init__()
        self.views = args.num_hops
        self.classes = args.num_class
        self.lambda_epochs = args.lambda_epochs
        self.kl = args.kl
        self.dis = args.dis
        self.Classifiers = MLP(args.input_dim, args.hid_dim, args.num_class,
                               args.input_droprate, args.dropout, args.use_bn)

    def forward(self, X, y, mask):
        evidence = dict()
        for i in range(len(X)):
            evidence[i] = self.infer(X[i])
        loss = 0
        evidence_a = 0
        alpha = dict()
        # Evidence Add
        for v_num in range(self.views + 1):
            alpha[v_num] = evidence[v_num] + 1
            evidence_a += evidence[v_num]
        alpha_a = evidence_a + 1
        alpha_a, u_a, p = self.cal_u(alpha_a)
        loss += ce_loss(y[mask], alpha_a[mask], self.classes)
        loss += KLL(y[mask], evidence_a[mask], self.classes, self.kl, self.dis)
        loss = torch.mean(loss)
        return evidence, evidence_a, u_a, loss

    def infer(self, input):
        return self.Classifiers(input)

    def cal_u(self, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        b = E / (S.expand(E.shape))
        u = self.classes / S
        p = b + 1 / self.classes * u
        return alpha, u, p
