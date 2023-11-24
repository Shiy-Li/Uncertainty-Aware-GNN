import torch
from tqdm import trange
import copy
import numpy as np

from lossfunc import AverageMeter, rand_prop
from model import UGN
from utils import build_optimizer


def train(dataset, args):
    losses = []
    test_accs = []
    test_losss = []
    best_acc = 0
    patience_t = 0
    best_loss = np.inf
    model = UGN(args)
    if torch.cuda.is_available():
        model.cuda()
        dataset.y = dataset.y.cuda()
    X_list = dataset.X_list
    target = dataset.y
    mask = dataset.train_mask
    scheduler, optimizer = build_optimizer(args, model.parameters())
    # for epoch in trange(1,args.epochs+1,desc='Training',unit='Epochs'):
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        data_list = []
        loss_meter = AverageMeter()
        for k in range(args.num_hops + 1):
            data_list.append(rand_prop(X_list[k], args.dropnode_rate, training=True))
        optimizer.zero_grad()
        if args.model_type == 'GCN':
            evidence, evidence_a, u_a, loss = model(dataset.x.cuda(), target, mask, dataset.edge_index.cuda())
        elif args.model_type == 'UGN':
            evidence, evidence_a, u_a, loss = model(data_list, target, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_meter.update(loss.item())
        losses.append(total_loss)
        if args.is_val:
            test_loss, test_acc, evidence, evidence_a, u_a, _ = test(dataset, model, args)
            test_accs.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
                patience_t = 0
            else:
                patience_t += 1
            if patience_t >= args.patience_period:
                return test_accs, losses, best_model, best_acc
        else:
            test_accs.append(test_accs[-1])
    return test_accs, losses, best_model, best_acc


def test(dataset, test_model, args):
    test_model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    X_list = dataset.X_list
    noise_list = dataset.noise_list
    target = dataset.y
    if torch.cuda.is_available():
        target = target.cuda()
    data_list = []
    if args.noise:
        print('noise!!!')
        for k in range(args.num_hops + 1):
            data_list.append(rand_prop(noise_list[k], args.dropnode_rate, training=False))
    else:
        for k in range(args.num_hops + 1):
            data_list.append(rand_prop(X_list[k], args.dropnode_rate, training=False))
    mask = dataset.val_mask if args.is_val else dataset.test_mask
    data_num += target[mask].size(0)
    with torch.no_grad():
        if args.model_type == 'UGN':
            evidence, evidence_a, u_a, loss = test_model(data_list, target, mask)
            _, predicted = torch.max(evidence_a.data, 1)
        elif args.model_type == 'GCN':
            evidence, evidence_a, u_a, loss = test_model(dataset.x.cuda(), target, mask, dataset.edge_index.cuda())
            _, predicted = torch.max(evidence_a.data, 1)
    correct_num += (predicted[mask] == target[mask]).sum().item()
    if args.save_model:
        pred = evidence_a.detach().cpu()
        std_each = pred.std(dim=1, keepdim=True)
        return std_each, loss.item(), correct_num / data_num, evidence, evidence_a, u_a, target
    return loss.item(), correct_num / data_num, evidence, evidence_a, u_a, target


def accu(dataset, evidence, target):
    mask = dataset.test_mask
    data_num, correct_num = 0, 0
    data_num += target[mask].size(0)
    _, predicted = torch.max(evidence.data, 1)
    correct_num += (predicted[mask] == target[mask]).sum().item()
    return correct_num / data_num
