import numpy as np
import pickle
import argparse
from datetime import datetime, timedelta
import csv
import os
import time
import pandas as pd
import numpy as np

# Set dataset paths
# =============
SRC = 'auth_optc.txt' # Location of auth_optc.txt
DST = 'test.csv' # output file
snapshot= 3600  #  length of snapshot (s)
# =============
optc=pd.read_csv(SRC,header=None,names=['timestamp', 'src_computer', 'dst_computer', 'label','1','2','3','4','5'])

optc=optc.drop(['1','2','3','4','5'], axis=1).reset_index(drop=True)

initial_time = min(optc['timestamp'])
optc['delta'] = optc['timestamp'] - initial_time
optc['snapshot'] = optc['delta'] // snapshot

optc = optc.drop(['delta'], axis=1).reset_index(drop=True)
print("trainwin:")
sp_time=optc[optc['label']==1].iloc[0]['timestamp']
print(optc[optc['timestamp']<sp_time]['snapshot'].nunique())

optc.to_csv(DST)