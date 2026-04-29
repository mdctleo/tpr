from argparse import ArgumentParser
import os
import pickle

import pandas as pd
import torch
#import loaders.load_lanl as lanl
import loaders.load_cic as lanl
from loaders.load_hv import configure_euler_for_hv
from hv_attacks_config import HV_ATTACKS
from models.recurrent import GRU, LSTM, EmptyModel
from models.embedders import \
    detector_gcn_rref, detector_gat_rref, detector_sage_rref, \
    predictor_gcn_rref, predictor_gat_rref, predictor_sage_rref 

from spinup import run_all
import spinup as _spinup
import torch
import numpy as np
import random
# seed = 0
# random.seed(seed) # python random generator
# np.random.seed(seed) # numpy random generator

# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

DEFAULT_TR = {
    'anom_lr': 0.05,
    'epochs': 100,
    'min': 1,
    'nratio': 10,
    'val_nratio': 1
}

OUTPATH = '' # Output folder for results.txt (ending in delimeter)

def get_args():
    global DEFAULT_TR

    ap = ArgumentParser()

    ap.add_argument(
        '-d', '--delta',
        type=float, default=0.5
    )

    ap.add_argument(
        '-w', '--workers',
        type=int, default=4
    )

    ap.add_argument(
        '-T', '--threads',
        type=int, default=1
    )

    ap.add_argument(
        '-e', '--encoder',
        choices=['GCN', 'GAT', 'SAGE'],
        type=str.upper,
        default="GCN"
    )

    ap.add_argument(
        '-r', '--rnn',
        choices=['GRU', 'LSTM', 'NONE'],
        type=str.upper,
        default="GRU"
    )

    ap.add_argument(
        '-H', '--hidden',
        type=int,
        default=32
    )

    ap.add_argument(
        '-z', '--zdim',
        type=int,
        default=16
    )

    ap.add_argument(
        '-n', '--ngrus',
        type=int,
        default=1
    )

    ap.add_argument(
        '-t', '--tests',
        type=int, 
        default=1
    )

    ap.add_argument(
        '-l', '--load',
        action='store_true'
    )

    ap.add_argument(
        '--fpweight',
        type=float,
        default=0.6
    )

    ap.add_argument(
        '--nowrite',
        action='store_true'
    )

    ap.add_argument(
        '--impl', '-i',
        type=str.upper,
        choices=['DETECT', 'PREDICT', 'D', 'P', 'PRED'],
        default="DETECT"
    )

    # For future new data sets
    ap.add_argument(
        '--dataset',
        default='LANL', 
        type=str.upper
    )

    ap.add_argument(
        '--lr',
        default=0.01,#0.005,
        type=float
    )
    ap.add_argument(
        '--patience',
        default=10,#5, 
        type=int
    )

    ap.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to data directory (euler/ or euler_red/). '
             'Overrides the default cic2017/ symlink.'
    )
    ap.add_argument(
        '--kde', action='store_true',
        help='Enable KDE timestamp-diff edge features'
    )
    ap.add_argument(
        '--kde_file', type=str, default='kde_vectors_euler.pkl',
        help='Path to pre-computed KDE pickle'
    )
    ap.add_argument(
        '--kde_dim', type=int, default=20,
        help='Dimensionality of KDE density vectors'
    )
    ap.add_argument(
        '--kde_decode', action='store_true',
        help='Use KDE at decode step only (no NNConv in encoder). '
             'Loads kde_dict onto TData for decode-time lookup.'
    )
    ap.add_argument(
        '--red', action='store_true',
        help='Use reduced graph (euler_red/) with (first_ts, last_ts, count) flow features'
    )
    ap.add_argument(
        '--attack', type=str, default=None,
        help='HyperVision attack name (e.g. charrdos). Implies --dataset HV_<attack>'
    )

    args = ap.parse_args()
    args.te_end = None
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'

    readable = str(args)
    print(readable)

    model_str = '%s -> %s (%s)' % (args.encoder , args.rnn, args.impl)
    print(model_str)
    
    # Parse dataset info 
    if args.dataset.lower().startswith('c'):
        # Handle --data_dir and --red overrides
        if args.data_dir:
            lanl.LANL_FOLDER = args.data_dir.rstrip('/') + '/'
        elif args.red:
            lanl.LANL_FOLDER = './cic2017_red/'
        # else: uses default './cic2017/'

        _spinup.DATASET_TAG = 'cic'

        # KDE: only pass kde_file if --kde or --kde_decode is set
        _kde_file = args.kde_file if (args.kde or args.kde_decode) else None

        # For --red config, we also read flow features (first_ts, last_ts, count)
        # from the reduced-graph data
        _use_flows = args.red

        # When --kde_decode: KDE is at decode step only.
        # Don't inject into eas (encoder stays plain GCNConv).
        # When --kde (without --kde_decode): KDE goes into eas for NNConv encoder.
        args.kde_decode_mode = args.kde_decode

        args.loader = lanl.load_lanl_dist
        args.kde_file = _kde_file
        args.use_flows = _use_flows
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        args.te_times = [(args.tr_end, lanl.TIMES['all'])]
        args.delta = int(args.delta * (60))
        args.manual = False 

    elif args.attack or args.dataset.lower().startswith('hv_'):
        # ---- HyperVision dataset ----
        attack_name = args.attack or args.dataset[3:].lower()
        hv_config = 'red' if args.red else ('kde' if (args.kde or args.kde_decode) else 'baseline')
        atk = configure_euler_for_hv(attack_name, config='red' if args.red else 'baseline')

        _spinup.DATASET_TAG = f'hv_{attack_name}_{hv_config}'

        # Store HV metadata for post-processing
        args._hv_attack_name = attack_name
        args._hv_config = 'kde' if (args.kde or args.kde_decode) else ('reduced' if args.red else 'baseline')
        args._hv_attack_info = atk

        _kde_file = args.kde_file if (args.kde or args.kde_decode) else None
        # For HV, resolve KDE file path to the per-attack pickle produced by
        # EULER/compute_kde_features_hv.py  →  compute_kde_features_euler.py
        if _kde_file and _kde_file == 'kde_vectors_euler.pkl':
            from hv_attacks_config import HV_DATA_ROOT
            _kde_file = os.path.join(HV_DATA_ROOT, attack_name, f'kde_vectors_euler_{attack_name}.pkl')
        _use_flows = args.red
        args.kde_decode_mode = args.kde_decode

        args.loader = lanl.load_lanl_dist
        args.kde_file = _kde_file
        args.use_flows = _use_flows
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None
        args.te_times = [(args.tr_end, lanl.TIMES['all'])]
        # For HV, delta is in microseconds directly (no ×60 conversion)
        args.delta = int(args.delta)
        args.manual = False

    else:
        raise NotImplementedError('Only CIC-IDS 2017 and HyperVision datasets are supported')

    # Convert from str to function pointer
    if args.encoder == 'GCN':
        args.encoder = detector_gcn_rref if args.impl[0] == 'D' \
            else predictor_gcn_rref
    elif args.encoder == 'GAT':
        args.encoder = detector_gat_rref if args.impl[0] == 'D' \
            else predictor_gat_rref
    else:
        args.encoder = detector_sage_rref if args.impl[0] == 'D' \
            else predictor_sage_rref

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM 
    else:
        args.rnn = EmptyModel

    return args, readable, model_str


def _save_hv_euler_predictions(args, stats_list, all_raw_preds_list):
    """Save predictions_latest.pkl for HyperVision EULER runs."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from save_hv_predictions import save_predictions

    attack_name = args._hv_attack_name
    config = args._hv_config
    hv_attack_info = args._hv_attack_info

    # Load node_map for IP reverse-mapping
    node_map = pickle.load(open(lanl.LANL_FOLDER + 'nmap.pkl', 'rb'))

    # Use the first run's results (if multiple tests, use first)
    if stats_list and stats_list[0]:
        s = stats_list[0][0]  # first test run, first stat dict
    else:
        print("[run.py] WARNING: No stats available for HV predictions")
        return

    raw_preds = all_raw_preds_list[0] if all_raw_preds_list else []

    if not raw_preds:
        print("[run.py] WARNING: No raw predictions available for HV predictions")
        return

    edge_df = pd.DataFrame(raw_preds)

    # Compute TN and FN from the data
    tp = int(s.get('TP', 0))
    fp = int(s.get('FP', 0))
    total_attack = int((edge_df['label'] == 1).sum()) if len(edge_df) > 0 else 0
    total_benign = int((edge_df['label'] == 0).sum()) if len(edge_df) > 0 else 0
    fn = total_attack - tp
    tn = total_benign - fp

    metrics = {
        'auc': s.get('AUC', 0),
        'ap': s.get('AP', 0),
        'tpr': s.get('TPR', 0),
        'fpr': s.get('FPR', 0),
        'precision': s.get('P', 0),
        'f1': s.get('F1', 0),
        'cutoff': 0,  # cutoff not in stats dict directly
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'train_time_sec': s.get('tr_time', 0),
        'test_time_sec': s.get('FwdTime', 0),
    }

    save_predictions(
        attack_name=attack_name,
        model='euler',
        config=config,
        edge_predictions=edge_df,
        metrics=metrics,
        node_map=node_map,
        hv_attack_info=hv_attack_info,
        base_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hypervision_results'),
    )


if __name__ == '__main__':
    args, argstr, modelstr = get_args() 
    DEFAULT_TR['lr'] = args.lr
    DEFAULT_TR['patience'] = args.patience

    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        # Need to tell workers to output in embed dim
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]

    results = [
        run_all(
            args.workers, 
            args.rnn, 
            rnn_args,
            args.encoder, 
            worker_args, 
            args.delta,
            args.load,
            args.fpweight,
            args.impl,
            args.loader, 
            args.tr_start,
            args.tr_end, 
            args.val_times,
            args.te_times,
            DEFAULT_TR,
            kde_file=args.kde_file,
            use_flows=args.use_flows,
            kde_decode=getattr(args, 'kde_decode_mode', False)
        )
        for _ in range(args.tests)
    ]

    # Unpack (stats, raw_preds) tuples
    stats = [r[0] for r in results]
    all_raw_preds = [r[1] for r in results]

    # Save HyperVision predictions if running an HV attack
    if hasattr(args, '_hv_attack_name'):
        _save_hv_euler_predictions(args, stats, all_raw_preds)

    # Don't write out if nowrite
    if args.nowrite:
        exit() 

    results_file = OUTPATH + _spinup.DATASET_TAG + '_results.txt'
    f = open(results_file, 'a')
    f.write(str(argstr) + '\n')
    f.write('LR: ' + str(args.lr) + '\n')
    f.write(modelstr + '\n')

    dfs = [pd.DataFrame(s) for s in list(zip(*stats))]
    dfs = pd.concat(dfs, axis=0)

    for m in dfs['Model'].unique():
        df = dfs[dfs['Model'] == m]

        compressed = pd.DataFrame(
            [df.mean(), df.sem()],
            index=['mean', 'stderr']
        ).to_csv().replace(',', '\t') # For easier copying into Excel

        full = df.to_csv(index=False, header=False)
        full = full.replace(',', ', ')

        f.write(m + '\n')
        f.write(str(compressed) + '\n')
        f.write(full + '\n')

    f.close()