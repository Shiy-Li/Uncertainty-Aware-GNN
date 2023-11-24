# 参数设置
import matplotlib.pyplot as plt
import argparse
from dataloader import get_dataset
from lossfunc import MessagePro2
from train_test import train, test


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000, help='the train epochs')
    parser.add_argument('--lambda-epochs', type=int, default=40,
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--feat-per-layer', type=list, default=[16, 7], help='the feature dimension of per layer')
    parser.add_argument('--hid-dim', type=int, default=32)
    parser.add_argument('--input-dim', type=int, default=16, help='the hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.8, help='the dropout')
    parser.add_argument('--dataset', type=str, default='cora', help='the dataset')
    parser.add_argument('--num-class', type=int, default=0, help='the num_class')
    parser.add_argument('--opt', type=str, default='adam', help='the optimizer')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch_size')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='the weight of decay')
    parser.add_argument('--opt-scheduler', type=str, default='none')
    parser.add_argument('--num-hops', type=int, default=2)
    parser.add_argument('--model-type', type=str, default='UGN')
    parser.add_argument('--save-model', type=bool, default=False)
    parser.add_argument('--opt-decay-step', type=int, default=[50, 100, 200])
    parser.add_argument('--opt-decay-rate', type=float, default=0.5)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--patience_period', type=int, default=200)
    parser.add_argument('--rg', type=float, default=0.9)
    parser.add_argument('--dropnode-rate', type=float, default=0.8)
    parser.add_argument('--input-droprate', type=float, default=0.5)
    parser.add_argument('--is-val', type=bool, default=True)
    parser.add_argument('--use-bn', type=bool, default=False)
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--mu', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--dp', type=float, default=0.)
    parser.add_argument('--kl', type=float, default=0.)
    parser.add_argument('--dis', type=float, default=0.)
    args = parser.parse_known_args()[0]
    return args


def main(args):
    dataset = get_dataset(args.dataset, 12345)
    dataset = MessagePro2(dataset, args.num_hops, args)
    args.input_dim = dataset.num_node_features
    args.num_class = dataset.num_classes
    args.save_model = False
    args.is_val = True
    val_accs, loss_meter_avg, best_model, best_acc = train(dataset, args)
    if args.painting:
        plt.title(dataset.name)
        plt.plot(loss_meter_avg, label="training loss" + " - " + args.model_type + "- " + str(args.num_hops))
        plt.plot(val_accs, label="test accuracy" + " - " + args.model_type + " - " + str(args.num_hops))
        plt.legend()
        plt.show()
    args.is_val = False
    args.save_model = True
    std, test_loss, acc, evidence, evidence_a, u_a, target = test(dataset, best_model, args)
    # torch.cuda.empty_cache()
    # print('Test set results ====> acc: {:.4f}'.format(acc))
    return acc


