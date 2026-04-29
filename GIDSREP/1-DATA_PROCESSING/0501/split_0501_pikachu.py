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
    args.add_argument('-n', '--name', type=str, default='0501rands2_wizard-sipder_VF3',help='Name of the dataset')
    args.add_argument('-t', '--time', type=float, default=5,help='time of snapshot(min)')
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    input_file = '{}'.format(args.input)

    output_file = '{}_{}.csv'.format(args.name,int(args.time*60))
    start_time = time.time()

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(input_file, 'r') as input, open(output_file, 'w+', newline='') as output:
        csv_writer = csv.writer(output)
        csv_writer.writerow(['timestamp', 'src_computer', 'dst_computer', 'label'])
        flow = input.readline()
        num = 0
        while flow:
            num += 1
            if num % 1000000 == 0:
                print(num, time.time()-start_time)
            flow = flow.split('\t')
            ts = flow[0]
            src = flow[2]
            dst = flow[4]
            label = flow[-1]
        
            csv_writer.writerow([str(num), ts, src, dst, label])
            flow = input.readline()

    #sort
    data=pd.read_csv(output_file,header=0)
    data=data.sort_values(by='timestamp', ascending=True,ignore_index=True)
    data=data.iloc[data[data['timestamp']<1682870100].shape[0]:]
    data=data.reset_index(drop=True)
    data.to_csv(output_file)

    initial_time = min(data['timestamp'])
    data['delta'] = data['timestamp'] - initial_time
    data['snapshot'] = data['delta'] // (60*args.time)
    data = data.drop(['delta'], axis=1).reset_index(drop=True)
    # trainwin in pikachu 
    print("trainwin:")
    sp_time=data[data['label']==1].iloc[0]['timestamp']
    print(data[data['timestamp']<sp_time]['snapshot'].nunique())
    #print(data['snapshot'].value_counts())
    
    data.to_csv(output_file)
