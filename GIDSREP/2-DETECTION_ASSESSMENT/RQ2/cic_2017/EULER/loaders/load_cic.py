from copy import deepcopy
import os 
import pickle 
from joblib import Parallel, delayed

import numpy as np
import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized

DATE_OF_EVIL_LANL = 29136#150885
FILE_DELTA = 100000

# Input where LANL data cleaned with .clean_lanl.py is stored
LANL_FOLDER = './cic2017/'#'./data/'
assert LANL_FOLDER, 'Please fill in the LANL_FOLDER variable:  loaders/load_cic.py'

COMP = 0
USR = 1
SPEC = 2
X_DIM = 17688

TIMES = {

    'all'  : 374762#1270000#5011199  # Full
}

torch.set_num_threads(1)

def empty_lanl():
    return make_data_obj([],None,None)

def load_lanl_dist(workers, start=0, end=635015, delta=8640, is_test=False, ew_fn=std_edge_w, kde_file=None, use_flows=False, kde_decode=False):
    if start == None or end == None:
        return empty_lanl()

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    workers = min(num_slices, workers)

    # Can't distribute the job if not enough workers
    if workers <= 1:
        return load_partial_lanl(start, end, delta, is_test, ew_fn, kde_file=kde_file, use_flows=use_flows, kde_decode=kde_decode)

    per_worker = [num_slices // workers] * workers 
    remainder = num_slices % workers

    # Give everyone a balanced number of tasks 
    # put remainders on last machines as last task 
    # is probably smaller than a full delta
    if remainder:
        for i in range(workers, workers-remainder, -1):
            per_worker[i-1] += 1

    kwargs = []
    prev = start 
    for i in range(workers):
        end_t = prev + delta*per_worker[i]
        kwargs.append({
            'start': prev, 
            'end': min(end_t-1, end),
            'delta': delta,
            'is_test': is_test,
            'ew_fn': ew_fn,
            'kde_file': kde_file,
            'use_flows': use_flows,
            'kde_decode': kde_decode
        })
        prev = end_t
    
    # Now start the jobs in parallel 
    datas = Parallel(n_jobs=workers, prefer='processes')(
        delayed(load_partial_lanl_job)(i, kwargs[i]) for i in range(workers)
    )

    # Helper method to concatonate one field from all of the datas
    data_reduce = lambda x : sum([getattr(datas[i], x) for i in range(workers)], [])

    # Just join all the lists from all the data objects
    print("Joining Data objects")
    x = datas[0].xs
    eis = data_reduce('eis')
    masks = data_reduce('masks')
    ews = data_reduce('ews')
    node_map = datas[0].nmap

    if is_test:
        ys = data_reduce('ys')
        cnt = data_reduce('cnt')
    else:
        ys = None
        cnt = None

    # Collect edge attributes if any worker produced them
    eas = None
    if not isinstance(datas[0].eas, None.__class__):
        eas = data_reduce('eas')

    # Merge kde_dict from all workers (they all load the same pickle)
    _kde_dict = getattr(datas[0], 'kde_dict', None)

    # After everything is combined, wrap it in a fancy new object
    print("Done")
    return TData(
        eis, x, ys, masks, ews=ews, nmap=node_map, eas=eas, kde_dict=_kde_dict,
        cnt=cnt
    )
 

# wrapper bc its annoying to send kwargs with Parallel
def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data


def make_data_obj(eis, ys, ew_fn, ews=None, eas=None, kde_dict=None, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt+1)
    
    # Build time-partitioned edge lists
    eis_t = []
    masks = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)

        # This is training data if no ys present
        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    # Balance the edge weights if they exist
    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    # Finally, return Data object
    return TData(
        eis_t, x, ys, masks, ews=ews, nmap=nm, eas=eas, kde_dict=kde_dict,
        cnt=cnt
    )

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files 
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, ew_fn=standardized, kde_file=None, use_flows=False, kde_decode=False):
    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    eas = []
    temp_flows = {}   # per-snapshot flow aggregation

    # Load pre-computed KDE vectors (if provided)
    kde_dict = {}
    kde_zero = []
    if kde_file is not None:
        raw = pickle.load(open(kde_file, 'rb'))
        kde_dict = raw['kde_dict'] if isinstance(raw, dict) and 'kde_dict' in raw else raw
        _kde_dim = len(next(iter(kde_dict.values())))
        kde_zero = [0.0] * _kde_dim
        print(f'  Loaded KDE vectors: {len(kde_dict)} edges, dim={_kde_dim}')

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    # Helper functions (trims the trailing \n)
    # euler/ line format: ts,src,dst,dur,bytes,pkts,label
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[-1][:-1]))
    # Parse flow features from columns 3,4,5
    fmt_flow = lambda x : (float(x[3]), float(x[4]), float(x[5]))

    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1]+1)
        else:
            edges_t[et] = (is_anom, 1)

    def add_flow(et, dur, bytes_, pkts):
        if et in temp_flows:
            temp_flows[et][0].append(dur)
            temp_flows[et][1].append(bytes_)
            temp_flows[et][2].append(pkts)
        else:
            temp_flows[et] = [[dur], [bytes_], [pkts]]


    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_marked = False
    keep_reading = True
    next_split = start+delta 

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime 
    while keep_reading:
        while line:
            l = line.split(',')
            
            # Scan to the correct part of the file
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(ts-old_ts)
                old_ts = ts 
                curtime = ts 
                continue
            
            ts, src, dst, label = fmt_line(l)
            et = (src,dst)

            # Not totally necessary but I like the loading bar
            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit 
            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    y,ew = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))

                    # Build edge attributes if use_flows or kde_dict (encoder mode only)
                    # When kde_decode=True, KDE goes to TData.kde_dict, not eas
                    _kde_for_eas = kde_dict if (kde_dict and not kde_decode) else {}
                    if use_flows or _kde_for_eas:
                        eas_flows_dim = 6 if use_flows else 0
                        fs = {}
                        for eij in edges_t.keys():
                            if use_flows and eij in temp_flows:
                                fl = temp_flows[eij]
                                feat = [
                                    np.mean(fl[0]), np.std(fl[0]),   # dur mean/std
                                    np.mean(fl[1]), np.std(fl[1]),   # bytes mean/std
                                    np.mean(fl[2]), np.std(fl[2]),   # pkts mean/std
                                ]
                            elif use_flows:
                                feat = [0.0] * eas_flows_dim
                            else:
                                feat = []
                            # Concatenate KDE features (encoder mode only)
                            if _kde_for_eas:
                                feat = feat + list(_kde_for_eas.get(eij, kde_zero))
                            fs[eij] = feat
                        all_feats = list(zip(*fs.values()))
                        eas.append(torch.tensor(all_feats, dtype=torch.float32))

                    if is_test:
                        ys.append(torch.tensor(y))
                    edges_t = {}
                    temp_flows = {}
                    

                # If the list was empty, just keep going if you can
                curtime = next_split 
                next_split += delta

                # Break out of loop after saving if hit final timestep
                if ts >= end:
                    keep_reading = False 
                    break 

            # Skip self-loops
            if et[0] == et[1]:
                line = in_f.readline()
                continue

            # Track flow features if needed
            if use_flows:
                dur, bytes_, pkts = fmt_flow(l)
                add_flow(et, dur, bytes_, pkts)

            add_edge(et, is_anom=label)
            line = in_f.readline()

        in_f.close() 
        cur_slice += FILE_DELTA 

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            keep_reading=False
            break

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()

    return make_data_obj(
        edges, ys, ew_fn,
        ews=ews, eas=eas if eas else None, node_map=node_map,
        kde_dict=kde_dict if kde_dict else None
    )


if __name__ == '__main__':
    data = load_lanl_dist(2, start=0, end=21600, delta=21600)
    print(data)