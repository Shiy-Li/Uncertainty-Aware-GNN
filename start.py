import warnings

from lossfunc import set_seeds
from main import parameter_parser, main

warnings.filterwarnings('ignore')
args = parameter_parser()
args.noise = False
args.painting = False
args.dataset = 'Cora'
args.model_type = 'EFGNN'
args.kl = 0.05
args.dis = 0.5
args.hid_dim = 32
args.input_droprate = 0.5
args.dropout = 0.3
args.dropnode_rate = 0.3
args.weight_decay = 2e-2
args.lr = 2e-2
args.num_hops = 32
seed = 1
set_seeds(seed)
main(args)
