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
DELTA=100000
def parse_args():
    args = argparse.ArgumentParser("walk")
    args.add_argument('-i', '--input', type=str, default='0501.log',help='fileame of the origin log')
    args.add_argument('-n', '--name', type=str, default='n9000',help='Name of the dataset')
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    input_file = '{}'.format(args.input)
    output_file = '{}_tmp.csv'.format(args.name)
    data_folder = '/{}/'.format(args.name)
    if not os.path.exists(data_folder): os.makedirs(data_folder)
    start_time = time.time()

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(input_file, 'r') as input, open(output_file, 'w+', newline='') as output:
        csv_writer = csv.writer(output)
        csv_writer.writerow(['timestamp', 'src_computer', 'dst_computer','dur','bytes','pkts', 'label'])
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
            if flow[8]=='-':
                dur=0
            else:
                dur=flow[8]
            if flow[9]=='-':
                ori_byte=0
            else:
                ori_byte=int(flow[9])
            if flow[10]=='-':
                resp_bytes=0
            else:
                resp_bytes=int(flow[10])
            if flow[16]=='-':
                orig_pkts=0
            else:
                orig_pkts=int(flow[16])
            if flow[18]=='-':
                resp_pkts=0
            else:
                resp_pkts=int(flow[18])
            label = flow[-1][:-1]
        
            csv_writer.writerow([str(num), ts, src, dst,dur,resp_bytes+ori_byte,orig_pkts+resp_pkts, label])
            flow = input.readline()
    #排序
    data=pd.read_csv(output_file,header=0)
    data=data.sort_values(by='timestamp', ascending=True,ignore_index=True)
    data=data.iloc[data[data['timestamp']<1682870100].shape[0]:]
    data=data.reset_index(drop=True)
    data.to_csv(output_file)

    start = datetime.fromtimestamp(float(data.iloc[0]['timestamp']))
    data1=data.values
    #data.to_csv(output_file)

    #生成对应格式
    data1=data.values
    min=data1[0][0]
    for i in range(len(data1)):
        data1[i][0]=int((data1[i][0]-min)*1000)
    #保存time
    time_set=np.array((data1[data[data['label']==1].index[0]][0],data1[-1][0]),dtype=np.int32)
    print(time_set)
    np.save(data_folder+'time.npy',time_set)
    # data_tr=(1682942400-min)*1000
    # print(data_tr)
    # print(data1[-1][0])
    # print(data[data['timestamp']>=1682942400].index[0])
    nmap = {} 
    nid = [0]

    def get_or_add(n):
        if n not in nmap:
            nmap[n] = nid[0]
            nid[0] += 1

        return nmap[n]
    
    fmt_line = lambda ts,src,dst,dur,bytes,pkts,label: (
        '%s,%s,%s,%s,%s,%s,%s\n' % (
            int(ts), get_or_add(src), get_or_add(dst),dur,bytes,pkts,
            label
        ), 
        int(ts)
    )


    cur_time=0
    
    DST = data_folder
    f_out = open(DST + str(cur_time) + '.txt', 'w+')

    for tokens in data1: 
        l, ts = fmt_line(tokens[0], tokens[1], tokens[2],tokens[3],tokens[4],tokens[5],tokens[6])

        # After ts progresses at least 10,0000 mseconds, make a new file
        if ts >= cur_time+DELTA:
            cur_time += DELTA
            f_out.close()
            f_out = open(DST + str(cur_time) + '.txt', 'w+')
        
        f_out.write(l)
    f_out.close()

    nmap_rev = [None] * (max(nmap.values()) + 1)
    for (k,v) in nmap.items():
        nmap_rev[v] = k

    with open(DST + 'nmap.pkl', 'wb+') as f:
        pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    data_inf=np.array((len(nmap_rev),data[data['label']==1].index[0]),dtype=np.int32)
    print(data_inf)
    np.save(data_folder+'data_info.npy',data_inf)