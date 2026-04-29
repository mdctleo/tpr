import pickle
import networkx
#from utils import DataUtils, GraphUtils
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

SRC = 'auth.txt' # Location of auth.txt
RED = 'redteam.txt' # Location of redteam.txt
DST = 'all_20.csv' # output file

snapshot= 3600  #  length of snapshot (s)


def preprocess_lanl_data(data_folder,red):
        auth_df = pd.DataFrame()
        for chunk_df in tqdm(pd.read_csv(data_folder,  sep=',', header=None,names=['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type','auth_ori','success'], dtype={'timestamp': np.int32, 'src_user': str, 'dst_user': str,
                                     'src_computer': str, 'dst_computer': str, 'auth_type': 'category',
                                     'logon_type': 'category'}, chunksize=10000000)):
            chunk_df = chunk_df[~((chunk_df['auth_type'] == '?') | (chunk_df['logon_type'] == '?'))]
            chunk_df = chunk_df[~((chunk_df['src_user'].str.contains(r'ANONYMOUS(?!$)')) | (
                chunk_df['src_user'].str.contains(r'LOCAL(?!$)')) | (chunk_df['src_user'].str.contains(r'NETWORK(?!$)')) | (
                chunk_df['src_user'].str.contains(r'ADMIN(?!$)')))]
            chunk_df = chunk_df[chunk_df['src_computer'] != chunk_df['dst_computer']]
            chunk_df = chunk_df.drop(['dst_user', 'auth_type', 'logon_type','auth_ori','success'], axis=1).reset_index(
                drop=True)
            # auth_df = auth_df.append(chunk_df, ignore_index=True)
            auth_df = pd.concat([auth_df, chunk_df], ignore_index=True)
        rt_df = pd.read_csv(red, header=0)
        rt_df.columns = ['timestamp', 'src_user', 'src_computer', 'dst_computer']
        filter_col_name = ['timestamp', 'src_user', 'src_computer', 'dst_computer']  # rt_df.columns.tolist()
        comm_df = pd.merge(auth_df.reset_index(), rt_df.reset_index(), how='inner', on=filter_col_name)
        # print("Anomalous rows: \n", comm_df)

        anom_row_index = comm_df.index_x.to_list()
        # print("Anom rows index: ", anom_row_index)

        # label row as anom or norm
        auth_df['label'] = 0
        auth_df.loc[anom_row_index, 'label'] = 1
        initial_time = min(auth_df['timestamp'])
        auth_df['delta'] = auth_df['timestamp'] - initial_time
        auth_df['snapshot'] = auth_df['delta'] // snapshot
        auth_df = auth_df.drop(['delta'], axis=1).reset_index(
            drop=True)
        print("trainwin")
        sp_time=auth_df[auth_df['label']==1].iloc[0]['timestamp']
        print(auth_df[auth_df['timestamp']<sp_time]['snapshot'].nunique())
        auth_df.to_csv("auth_all.csv")

def lanl_user_subset(data_folder,dst):
        lanl_df = pd.read_csv(data_folder, header=0, index_col=0, dtype={'timestamp': np.int32, 'src_user': str, 'src_computer': str, 'dst_computer': str, 'label': np.bool,
                                     'snapshot': int})
        anom_user_df = lanl_df[lanl_df['label'] == 1]
        anom_row_index = anom_user_df.index.to_list()
        print("Total Anom Edges: ", len(anom_row_index))
        print("Anom rows index: ", anom_row_index)

        anom_user = list(set(lanl_df.loc[anom_row_index, 'src_user'].tolist()))
        print("Anomalous Users: ", len(anom_user), anom_user)
        all_user = lanl_df.src_user.unique()
        print("total users: ", len(all_user))

        # anom_user = ['U748@DOM1', 'U1723@DOM1', 'U636@DOM1', 'U6115@DOM1', 'U620@DOM1']#, 'U737@DOM1', 'U825@DOM1', 'U1653@DOM1', 'U293@DOM1',

        norm_users = np.setdiff1d(all_user, anom_user).tolist()
        print("Norm users: ", len(norm_users))
        norm_users = random.sample(norm_users, len(anom_user) * 20)
        all_users = norm_users + anom_user
        print("all users: ", len(all_users), all_users)
        all_user_df = lanl_df[lanl_df['src_user'].isin(all_users)]
        all_user_df.to_csv(dst)

preprocess_lanl_data(data_folder=SRC,red=RED)

lanl_user_subset("auth_all.csv",DST)