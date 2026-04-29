from argparse import ArgumentParser
import os, datetime
import pandas as pd
import torch
import loaders.load_cic as optc
import loaders.load_cic_flow as lanl
from models.recurrent import GRU, LSTM, EmptyModel
from models.argus import detector_lanl_rref, detector_optc_rref
from classification import classification

# Reproducibility
import numpy as np
import random
seed = 0
random.seed(seed) # python random generator
np.random.seed(seed) # numpy random generator

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def args():
    ap = ArgumentParser()
    ap.add_argument('-d', '--delta', type=float, default=1)
    ap.add_argument('-e', '--encoder_name', type=str.upper,default="ARGUS")
    ap.add_argument('-r', '--rnn', choices=['GRU', 'LSTM', 'NONE'], type=str.upper, default="GRU")
    ap.add_argument('-H', '--hidden', type=int, default=32)
    ap.add_argument('-z', '--zdim', type=int, default=16)
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    # The end of testing time, see load_lanl.TIMES
    ap.add_argument('-te', '--te_end',  type=str.lower, default="test")
    ap.add_argument('--fpweight', type=float, default=0.6)
    # For future new data sets
    ap.add_argument('--dataset', default='LANL', type=str)
    ap.add_argument('--lr', default=0.01, type=float)
    ap.add_argument('--patience', default=3, type=int)
    ap.add_argument('--nratio', default=1, type=int)
    ap.add_argument('--epochs', default=200, type=int)
    ap.add_argument('--flows', action='store_false')
    ap.add_argument('--data_dir', default='/scratch/asawan15/GIDSREP/1-DATA_PROCESSING/cic_2017',
                    help='Base data directory. euler/ and argus_flow/ are resolved from here.')
    ap.add_argument('--kde', action='store_true',
                    help='Enable KDE timestamp-diff edge features (L_cic_flow only)')
    ap.add_argument('--kde_file', type=str, default='kde_vectors_argus.pkl',
                    help='Path to pre-computed KDE pickle')
    ap.add_argument('--kde_dim', type=int, default=10,
                    help='Dimensionality of KDE density vectors')
    ap.add_argument('--red', action='store_true',
                    help='Use reduced graph from argus_flow_red/')
    args = ap.parse_args()
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'
    readable = str(args)
    print(readable)
    model_str = '%s -> %s ' % (args.encoder_name , args.rnn)
    print(model_str)
    args.dataset = args.dataset+'_'+args.encoder_name

    # Parse dataset info
    if args.dataset.startswith('O'):
        optc.LANL_FOLDER = args.data_dir.rstrip('/') + '/euler/'
        args.loader = optc.load_lanl_dist
        args.tr_start = 0
        args.tr_end = optc.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        #make the test end as an input param
        args.te_times = [(args.tr_end, optc.TIMES[args.te_end])]
        args.delta = int(args.delta * (60))
    elif args.dataset.startswith('L'):
        if args.red:
            lanl.LANL_FOLDER = args.data_dir.rstrip('/') + '/argus_flow_red/'
        else:
            lanl.LANL_FOLDER = args.data_dir.rstrip('/') + '/argus_flow/'
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        #make the test end as an input param
        args.te_times = [(args.tr_end, lanl.TIMES[args.te_end])]
        # args.delta = 1
        args.delta = int(args.delta * (60))
    
    else:
        raise NotImplementedError('Only OpTC and LANL data sets are supported.')

    # Convert from str to function pointer
    if (args.encoder_name == 'ARGUS') and (args.dataset.startswith('L')):
        args.encoder = detector_lanl_rref
    elif (args.encoder_name == 'ARGUS') and (args.dataset.startswith('O')):
        args.encoder = detector_optc_rref
    else:
        raise NotImplementedError("wrong encoder", args.encoder_name, args.dataset)

    # KDE file path (only used when --kde is set with L_cic_flow)
    args.kde_file_path = args.kde_file if (args.kde and args.dataset.startswith('L')) else None

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM
    else:
        args.rnn = EmptyModel
    return args, readable, model_str

if __name__ == '__main__':
    args, argstr, modelstr = args()
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    torch.set_num_threads(12)
    print(device)
    OUTPATH = './Exps/result/'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/' # Output folder for results.txt (ending in delimeter)
    if not os.path.exists(OUTPATH): os.makedirs(OUTPATH)
    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]
    stats = classification(args, rnn_args, worker_args, OUTPATH, device)
    f = open('results_cic.txt', 'a')
    f.write('\n')
    f.write(str(argstr) + '\n')
    f.write('LR: ' + str(args.lr) + '\n')
    f.write(str(stats[0])+'\n')

