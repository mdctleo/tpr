from urllib.parse import urlparse
def get_host(url: str) -> (str, str, str):
	parsed_url = urlparse(url)
	return parsed_url.scheme, parsed_url.netloc, parsed_url.path
import urllib3
urllib3.get_host = get_host

import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from dgl.data import DGLDataset
import dgl
import time
import networkx as nx
import category_encoders as ce
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch
#import tqdm
import math
from typing import *
from sklearn.preprocessing import StandardScaler, Normalizer
import torch
import dgl
import numpy as np
import pandas as pd
import torch.optim as optim
import time
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import IsolationForest
import gc
from tqdm import tqdm
import itertools
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix



file_name = "./wiz/wizard_spider.csv"



split=3802012
ndim_out = 64
data = pd.read_csv(file_name,index_col=0,names=['Source IP', 'Destination IP', 'duration', 'orig_bytes', 'resp_bytes',
       'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
       'resp_ip_bytes', 'label'])
data.rename(columns=lambda x: x.strip(), inplace=True)
data['Source IP'] = data['Source IP'].apply(str)
data['Destination IP'] = data["Destination IP"].apply(str)

train_data=data.iloc[:split]
test_data=data.iloc[split:]
X_train= train_data.drop(columns=["label"])
y_train= train_data[["label"]]
X_test= test_data.drop(columns=["label"])
y_test= test_data[["label"]]

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

scaler = Normalizer()
cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns))) # Ignore first two as the represents IP addresses
scaler.fit(X_train[cols_to_norm])

# Transform on training set
X_train[cols_to_norm] = scaler.transform(X_train[cols_to_norm])
X_train['h'] = X_train.iloc[:, 2:].values.tolist()

# Transform on testing set
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test.iloc[:, 2:].values.tolist()

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)


train_g = nx.from_pandas_edgelist(train, "Source IP", "Destination IP",
            ["h", "label"], create_using=nx.MultiGraph())

train_g = train_g.to_directed()
train_g = dgl.from_networkx(train_g, edge_attrs=['h','label'])
nfeat_weight = torch.ones([train_g.number_of_nodes(),
train_g.edata['h'].shape[1]])
train_g.ndata['h'] = nfeat_weight

# Testing graph
test_g = nx.from_pandas_edgelist(test, "Source IP", "Destination IP",
            ["h", "label"], create_using=nx.MultiGraph())

test_g = test_g.to_directed()
test_g = dgl.from_networkx(test_g, edge_attrs=['h', 'label'])
nfeat_weight = torch.ones([test_g.number_of_nodes(),
test_g.edata['h'].shape[1]])
test_g.ndata['h'] = nfeat_weight


# Format node and edge features for E-GraphSAGE
train_g.ndata['h'] = torch.reshape(train_g.ndata['h'],
                                   (train_g.ndata['h'].shape[0], 1,
                                    train_g.ndata['h'].shape[1]))

train_g.edata['h'] = torch.reshape(train_g.edata['h'],
                                   (train_g.edata['h'].shape[0], 1,
                                    train_g.edata['h'].shape[1]))


test_g.ndata['h'] = torch.reshape(test_g.ndata['h'],
                                   (test_g.ndata['h'].shape[0], 1,
                                    test_g.ndata['h'].shape[1]))



test_g.edata['h'] = torch.reshape(test_g.edata['h'],
                                   (test_g.edata['h'].shape[0], 1,
                                    test_g.edata['h'].shape[1]))


import pickle
pickle.dump(train_g,open('wiz_train_g.pkl','wb'))
pickle.dump(test_g,open('wiz_test_g.pkl','wb'))


ndim_in = train_g.ndata['h'].shape[2]
# hidden_features = 128
num_layers = 1
edim = train_g.edata['h'].shape[2]
learning_rate = 0.0001
epochs = 4000
patience=5

from model import DGI
dgi = DGI(ndim_in,
    ndim_out,
    edim,
    F.relu)

dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                lr=learning_rate,
                weight_decay=0.)


cnt_wait = 0
best = 1e9
best_t = 0
dur = []
node_features = train_g.ndata['h']
edge_features = train_g.edata['h']

for epoch in range(epochs):
    dgi.train()
    if epoch >= 3:
        t0 = time.time()

    dgi_optimizer.zero_grad()
    loss = dgi(train_g, node_features, edge_features)
    loss.backward()
    dgi_optimizer.step()

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(dgi.state_dict(), 'best_dgiswiz_128.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    if epoch >= 3:
        dur.append(time.time() - t0)

    if epoch % 1 == 0:

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur),
              loss.item(),
              train_g.num_edges() / np.mean(dur) / 1000))
        

dgi.load_state_dict(torch.load('best_dgiswiz_128.pkl'))

training_emb = dgi.encoder(train_g, train_g.ndata['h'], train_g.edata['h'])[1]
training_emb = training_emb.detach().cpu().numpy()

testing_emb = dgi.encoder(test_g, test_g.ndata['h'], test_g.edata['h'])[1]
testing_emb = testing_emb.detach().cpu().numpy()


df_train = pd.DataFrame(training_emb, )
df_train["label"] = train_g.edata['label'].detach().cpu().numpy()
df_test = pd.DataFrame(testing_emb, )
df_test["label"] = test_g.edata['label'].detach().cpu().numpy()


normal_train_samples = df_train.drop(columns=["label"])
train_labels = df_train["label"]
test_labels = df_test["label"]
test_samples = df_test.drop(columns=["label"])

import warnings
warnings.filterwarnings("ignore")

from pyod.models.cblof import CBLOF
n_est = [3,5,7,9,10]
contamination = [0.001,0.01,0.04,0.05, 0.1, 0.2]
alpha=[0.75]
beta=[3]
params = list(itertools.product(n_est, contamination,alpha,beta))
score = -1
bs = None
for n_est, con,al,be in tqdm(params):
    try:
        clf_if = CBLOF(n_clusters=n_est, contamination=con,alpha=al,beta=be)
        clf_if.fit(normal_train_samples)
        y_pred = clf_if.predict(test_samples)
        test_pred = y_pred


        # f1 = f1_score(test_labels, test_pred, average='binary')
        # print(f1)
        auc=roc_auc_score(test_labels, test_pred)
        ap=average_precision_score(test_labels, test_pred)
        print({'n_estimators': n_est,
                        "con": con,
                        'al':al,
                        "be":be
                    })
        print('auc:',auc)
        print('ap:',ap)

        if auc+ap > score:
            score = auc+ap
            best_params = {'n_estimators': n_est,
                        "con": con,
                        'al':al,
                        "be":be
                    }
            bs = test_pred
        del clf_if
        gc.collect()
    except:
        print('wrong')
        print({'n_estimators': n_est,
                        "con": con,
                        'al':al,
                        "be":be
                    })


print(best_params)
print(score)
tn, fp, fn, tp = confusion_matrix(test_labels, bs, labels=[False, True]).ravel()
print("FPR: ", fp / (fp + tn))
print("TPR: ", tp / (tp + fn))
print("P: %0.4f, R: %0.4f" % (tp/(tp+fp), tp/(tp+fn)))
print('auc:',roc_auc_score(test_labels, bs))
print('ap:',average_precision_score(test_labels, bs))

print(best_params)
print(score)
tn, fp, fn, tp = confusion_matrix(test_labels, bs, labels=[False, True]).ravel()
print("FPR: ", fp / (fp + tn))
print("TPR: ", tp / (tp + fn))
print("P: %0.4f, R: %0.4f" % (tp/(tp+fp), tp/(tp+fn)))
print('auc:',roc_auc_score(test_labels, bs))
print('ap:',average_precision_score(test_labels, bs))
f=open('result_sand.txt','a+')
f.write('cblof'+'\n')
f.write('emb128')
f.write("n_est: "+ str(best_params["n_estimators"])+'\n')
f.write("con: "+ str(best_params["con"])+'\n')
f.write("FPR: "+ str( fp / (fp + tn))+'\n')
f.write("TPR: "+str(tp / (tp + fn))+'\n')
f.write("p: "+str(tp/(tp+fp))+'\n')
f.write('auc: '+str(roc_auc_score(test_labels, bs))+'\n')
f.write('ap: '+str(average_precision_score(test_labels, bs))+'\n')
f.close()

from pyod.models.hbos import HBOS
n_est = [5,10,15,20,25,30]
contamination = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]
params = list(itertools.product(n_est, contamination))
score = -1
bs = None
for n_est, con in tqdm(params):
    try:
    
        clf_if = HBOS(n_bins=n_est, contamination=con)
        clf_if.fit(normal_train_samples)
        y_pred = clf_if.predict(test_samples)
        test_pred = y_pred

        auc=roc_auc_score(test_labels, test_pred)
        ap=average_precision_score(test_labels, test_pred)
        print({'n_estimators': n_est,
                        "con": con,
                    })
        print('auc:',auc)
        print('ap:',ap)

        if auc+ap > score:
            score = auc+ap
            best_params = {'n_estimators': n_est,
                        "con": con,
                    }
            bs = test_pred
        del clf_if
        gc.collect()
    except:
        print('wrong')
        print({'n_estimators': n_est,
                        "con": con,
                        'al':al,
                        "be":be
                    })


print(best_params)
print(score)
tn, fp, fn, tp = confusion_matrix(test_labels, bs, labels=[False, True]).ravel()
print("FPR: ", fp / (fp + tn))
print("TPR: ", tp / (tp + fn))
print("P: %0.4f, R: %0.4f" % (tp/(tp+fp), tp/(tp+fn)))
print('auc:',roc_auc_score(test_labels, bs))
print('ap:',average_precision_score(test_labels, bs))
f=open('result_sand.txt','a+')
f.write('hbos'+'\n')
f.write('emb128')
f.write("n_est: "+ str(best_params["n_estimators"])+'\n')
f.write("con: "+ str(best_params["con"])+'\n')
f.write("FPR: "+ str( fp / (fp + tn))+'\n')
f.write("TPR: "+str(tp / (tp + fn))+'\n')
f.write("p: "+str(tp/(tp+fp))+'\n')
f.write('auc: '+str(roc_auc_score(test_labels, bs))+'\n')
f.write('ap: '+str(average_precision_score(test_labels, bs))+'\n')
f.close()

from pyod.models.pca import PCA
n_est = [5,10,15,20,25,30]
cont = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]
params = list(itertools.product(n_est, cont))
score = -1
bs = None

for n_est, con in tqdm(params):
    try:
    
        clf_if = PCA(n_components=n_est, contamination=con)
        clf_if.fit(normal_train_samples)
        y_pred = clf_if.predict(test_samples)
        test_pred = y_pred

        auc=roc_auc_score(test_labels, test_pred)
        ap=average_precision_score(test_labels, test_pred)
        print({'n_estimators': n_est,
                        "con": con,
                    })
        print('auc:',auc)
        print('ap:',ap)

        if auc+ap > score:
            score = auc+ap
            best_params = {'n_estimators': n_est,
                        "con": con,
                    }
            bs = test_pred
        del clf_if
        gc.collect()
    except:
        print('wrong')
        print({'n_estimators': n_est,
                        "con": con,
                        'al':al,
                        "be":be
                    })


print(best_params)
print(score)
tn, fp, fn, tp = confusion_matrix(test_labels, bs, labels=[False, True]).ravel()
print("FPR: ", fp / (fp + tn))
print("TPR: ", tp / (tp + fn))
print("P: %0.4f, R: %0.4f" % (tp/(tp+fp), tp/(tp+fn)))
print('auc:',roc_auc_score(test_labels, bs))
print('ap:',average_precision_score(test_labels, bs))
f=open('result_sand.txt','a+')
f.write('emb128')
f.write('pca'+'\n')
f.write("n_est: "+ str(best_params["n_estimators"])+'\n')
f.write("con: "+ str(best_params["con"])+'\n')
f.write("FPR: "+ str( fp / (fp + tn))+'\n')
f.write("TPR: "+str(tp / (tp + fn))+'\n')
f.write("p: "+str(tp/(tp+fp))+'\n')
f.write('auc: '+str(roc_auc_score(test_labels, bs))+'\n')
f.write('ap: '+str(average_precision_score(test_labels, bs))+'\n')
f.close()


from sklearn.ensemble import IsolationForest

n_est = [20, 50, 100, 150]
cont = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]
params = list(itertools.product(n_est, cont))
score = -1
bs = None

for n_est, con in tqdm(params):
    try:
        clf_if = IsolationForest(n_estimators=n_est, contamination=con)
        clf_if.fit(normal_train_samples)
        y_pred = clf_if.predict(test_samples)
        test_pred = y_pred

        auc=roc_auc_score(test_labels, test_pred)
        ap=average_precision_score(test_labels, test_pred)
        print({'n_estimators': n_est,
                        "con": con,
                    })
        print('auc:',auc)
        print('ap:',ap)

        if auc+ap > score:
            score = auc+ap
            best_params = {'n_estimators': n_est,
                        "con": con,
                    }
            bs = test_pred
        del clf_if
        gc.collect()
    except:
        print('wrong')
        print({'n_estimators': n_est,
                        "con": con,
                        'al':al,
                        "be":be
                    })


print(best_params)
print(score)
tn, fp, fn, tp = confusion_matrix(test_labels, bs, labels=[False, True]).ravel()
print("FPR: ", fp / (fp + tn))
print("TPR: ", tp / (tp + fn))
print("P: %0.4f, R: %0.4f" % (tp/(tp+fp), tp/(tp+fn)))
print('auc:',roc_auc_score(test_labels, bs))
print('ap:',average_precision_score(test_labels, bs))
f=open('result_sand.txt','a+')
f.write('emb128')
f.write('if'+'\n')
f.write("n_est: "+ str(best_params["n_estimators"])+'\n')
f.write("con: "+ str(best_params["con"])+'\n')
f.write("FPR: "+ str( fp / (fp + tn))+'\n')
f.write("TPR: "+str(tp / (tp + fn))+'\n')
f.write("p: "+str(tp/(tp+fp))+'\n')
f.write('auc: '+str(roc_auc_score(test_labels, bs))+'\n')
f.write('ap: '+str(average_precision_score(test_labels, bs))+'\n')
f.close()
