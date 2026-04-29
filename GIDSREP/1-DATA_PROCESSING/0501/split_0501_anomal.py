import numpy as np
import pickle
import pandas as pd
import argparse
from datetime import datetime, timedelta
import csv
import os
import time
import pandas as pd
import numpy as np

def parse_args():
    args = argparse.ArgumentParser("walk")
    args.add_argument('-i', '--input', type=str, default='0501.log',help='fileame of the origin log')
    args.add_argument('-n', '--name', type=str, default='',help='Name of the dataset')

    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    
    input_file = '{}'.format(args.input)
    output_file='{}_anomal.csv'.format(args.name)
    cols=["ts "     ,'uid '   , 'id.orig_h'  ,     'id.orig_p' ,      'id.resp_h'     , ' id.resp_p'    ,   'proto'  , 'service' ,'duration'     ,   'orig_bytes'  ,    'resp_bytes '   ,  'conn_state '  , 'local_orig '   ,  'local_resp '   ,  'missed_bytes'  ,  'history' ,'orig_pkts'      , 'orig_ip_bytes',  ' resp_pkts '  ,    'resp_ip_bytes' ,  'tunnel_parents','label']
    data=pd.read_csv(input_file,sep='\t',header=None,names=cols)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    data.drop(columns=["ts", "uid",'id.orig_p','id.resp_p','proto','service','local_orig','local_resp','history','tunnel_parents'], inplace=True)
    data.drop(columns=['conn_state'], inplace=True)
    data.replace('-', 0, inplace=True)
    data.to_csv(output_file,header=0)
    print('Dividing line between train_data and test_data: ')
    print(data[data['label']==1].iloc[0].name)

