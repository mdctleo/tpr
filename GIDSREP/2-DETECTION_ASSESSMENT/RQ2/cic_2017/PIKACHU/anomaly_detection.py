# ******************************************************************************
# anomaly_detection.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/11/21   Paudel     Initial version,
# ******************************************************************************
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, log_loss, classification_report
from sklearn.metrics import average_precision_score as ap_score
from sklearn import metrics
import numpy as np
import pandas as pd
import random
from scipy.special import softmax
import pickle
from multiprocessing import Pool

from timeit import default_timer as timer
import os
import tensorflow as tf


def aggregate_neighbors(node_emb, node_list, u, n_u):
    CNu = [node_emb[node_list.index(str(n))] for n in n_u]
    Cu = node_emb[node_list.index(str(u))]
    H = 1 / (1 + len(CNu)) * np.add(Cu, sum(Cn for Cn in CNu))
    return np.array(H)

def predict_prob(w, X):
    return softmax(np.dot(X, w.T))
    # return softmax(np.dot(X, w) + b)

def calculate_edge_probability(w, node_emb, node_list, G,
                               kde_dict=None, reverse_node_map=None,
                               kde_dim=0, kde_zero=None):
    edge_scores = []
    for u, v, data in G.edges(data=True):
        if 1==1:#(u, v, data['anom']) not in edges:
            Nu = [n for n in G.neighbors(u)]
            Hu = aggregate_neighbors(node_emb, node_list, u, Nu)
            if kde_dict is not None:
                kde_vec = _lookup_kde(u, v, kde_dict, reverse_node_map, kde_zero)
                Hu = np.concatenate([Hu, kde_vec])
            Hu = np.reshape(Hu, (1, Hu.shape[0]))
            Pv = predict_prob(w, Hu)

            Nv = [n for n in G.neighbors(v)]
            Hv = aggregate_neighbors(node_emb, node_list, v, Nv)
            if kde_dict is not None:
                kde_vec = _lookup_kde(v, u, kde_dict, reverse_node_map, kde_zero)
                Hv = np.concatenate([Hv, kde_vec])
            Hv = np.reshape(Hv, (1, Hv.shape[0]))
            Pu = predict_prob(w, Hv)
            score = ((1-Pv[0, v]) + (1-Pu[0, u]))/2
            edge_scores.append([u, v, score, data['snapshot'], data['time'], data['anom']])
    return edge_scores


def _lookup_kde(u_id, v_id, kde_dict, reverse_node_map, kde_zero):
    """Look up KDE vector for edge (u_id, v_id) using IP-keyed dict."""
    if reverse_node_map is None:
        return kde_zero
    u_ip = reverse_node_map.get(u_id)
    v_ip = reverse_node_map.get(v_id)
    if u_ip is None or v_ip is None:
        return kde_zero
    return kde_dict.get((u_ip, v_ip), kde_zero)

class AnomalyDetection:
    def __init__(self, args, node_list, node_map, node_embeddings, idx, kde_dict=None):
        self.args = args
        self.node_list = node_list
        self.node_map = node_map
        self.node_embeddings = node_embeddings
        self.idx = idx
        self.kde_dict = kde_dict
        # Build reverse map: int_id → ip_str
        self.reverse_node_map = {v: k for k, v in node_map.items()} if kde_dict else None
        if kde_dict:
            sample_vec = next(iter(kde_dict.values()))
            self.kde_dim = len(sample_vec)
            self.kde_zero = np.zeros(self.kde_dim, dtype=np.float32)
            print(f"  KDE enabled: dim={self.kde_dim}, edges={len(kde_dict)}")
        else:
            self.kde_dim = 0
            self.kde_zero = None

    def get_ip(self, node):
        for k, v in self.node_map.items():
            if v == node:
                return k
        return None

    def aggregate_neighbors_object(self, u, n_u):
        CNu = [self.node_embeddings[self.idx][self.node_list.index(str(n))] for n in n_u]
        Cu = self.node_embeddings[self.idx][self.node_list.index(str(u))]
        H = 1/(1+len(CNu))*np.add(Cu, sum(Cn for Cn in CNu))
        return np.array(H)

    def initialize_parameters(self, k, v):
        w = np.random.randn(v, k) * 0.0001
        return w

    def propagate(self, w, X, Y):
        m = X.shape[1]
        #calculate activation function
        p = softmax(np.dot(X, w.T))
        # find the cost (cross entropy)
        cost = log_loss(Y, p)

        # find gradient (back propagation)
        dw = (1 / m) * np.dot((p - Y).T, X)

        cost = np.squeeze(cost)
        grads = {"dw": dw}
        return grads, cost

    def gradient_descent(self, w, X, Y, iterations, learning_rate):
        costs = []
        for i in range(iterations):
            grads, cost = self.propagate(w, X, Y)
            # update parameters
            w = w - learning_rate * grads["dw"]
            costs.append(cost)
            print("Cost after iteration %i/%i:      %f" % (i, iterations, cost))
        params = {"w": w}
        return params, costs

    def predict(self, w, X):
        return softmax(np.dot(X, w.T))

    def get_train_edges(self, train_graphs, s = 10):
        data_x = []
        data_y = []
        for G in tqdm(train_graphs):
            print("G: ", len(G.nodes()), len(G.edges()))
            for u in G.nodes():
                N = [n for n in G.neighbors(u)]
                for v in N:
                    if len(N) > 1:
                        n_minus_v = [n for n in N if n != v]
                        support_set = random.choices(n_minus_v, k=s)
                    else:
                        support_set = random.choices(N, k=s)
                    H = self.aggregate_neighbors_object(u, support_set)
                    if self.kde_dict is not None:
                        kde_vec = _lookup_kde(u, v, self.kde_dict,
                                              self.reverse_node_map, self.kde_zero)
                        H = np.concatenate([H, kde_vec])
                    y = np.zeros(len(self.node_list))
                    y[v] = 1
                    data_x.append(H)
                    data_y.append(y)
            self.idx += 1

        return np.array(data_x), np.array(data_y)

    def print_result(self, percentile, threshold, true_label, pred_label):
        print("\n====BEST ANOMALY DETECTION RESULTS====")
        print("Percentile : ", percentile, " Threshold : ", threshold)
        print("---------------------------------------\n")
        print(metrics.classification_report(true_label, pred_label))
        print("Confusion Matrix: \n", confusion_matrix(true_label, pred_label, labels=[False, True]))
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
        print("(tn, fp, fn, tp): ", tn, fp, fn, tp)

    def calculate_performance_metrics(self, edge_scores, result_file,tr,te):
        print("\n\nCalculating Performance Metrices....")
        true_label = list(edge_scores['label'])
        scores = list(edge_scores['score'])
        fpr, tpr, thresholds = metrics.roc_curve(true_label, scores, pos_label=1)
        ap = ap_score(true_label, scores)
        # print("FPR: ", list(fpr))
        # print("TPR: ", list(tpr))
        fw = 0.5
        tw = 1 - fw
        fn = np.abs(tw * tpr - fw * (1 - fpr))
        best = np.argmin(fn, 0)
        print("\n\nOptimal cutoff %0.10f achieves TPR: %0.5f FPR: %0.5f on train data"
              % (thresholds[best], tpr[best], fpr[best]))
        print("Final AUC: ", metrics.auc(fpr, tpr))
        print("AUC: ", metrics.roc_auc_score(true_label, scores))
        print("AP:" ,ap)
        edge_scores['pred'] = np.where(edge_scores['score'] >= thresholds[best], True, False)
        true_label = list(edge_scores['label'])
        pred_label = list(edge_scores['pred'])
        print("\n\n======= CLASSIFICATION REPORT =========\n")
        print(classification_report(true_label, pred_label))
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
        print("Confusion Matrix: \n", confusion_matrix(true_label, pred_label, labels=[False, True]))
        print("FPR: ", fp / (fp + tn))
        print("TPR: ", tp / (tp + fn))
        print("P: %0.4f, R: %0.4f" % (tp/(tp+fp), tp/(tp+fn)))
        f= open('./results/results.txt','a')
        f.write(self.args.dataset+':'+'\n')
        f.write("AUC: "+str(metrics.roc_auc_score(true_label, scores))+'\n')
        f.write("ap: "+str(ap)+'\n')
        f.write("FPR: "+str(fp / (fp + tn))+'\n')
        f.write("TPR: "+str(tp / (tp + fn))+'\n')
        f.write("P: "+str(tp/(tp+fp))+'\n')
        f.write("tn: "+str(tn)+'\n')
        f.write("fp: "+str(fp)+'\n')
        f.write("fn: "+str(fn)+'\n')
        f.write("tp: "+str(tp)+'\n')
        f.write("cut_off: "+str(thresholds[best])+'\n')
        f.write("tr_time: "+str(tr/10)+'\n')
        f.write("te_time: "+str(te)+'\n')
        f.write('\n')
        f.close()

        reported_anom = edge_scores[edge_scores['pred'] == True]
        reported_anom['src'] = reported_anom['src'].apply(self.get_ip)
        reported_anom['dest'] = reported_anom['dest'].apply(self.get_ip)
        reported_anom.to_csv("results/" + result_file)

        # Build metrics dict for post-processing
        metrics_dict = {
            'auc': float(metrics.roc_auc_score(true_label, scores)),
            'ap': float(ap),
            'tpr': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
            'cutoff': float(thresholds[best]),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'train_time_sec': float(tr),
            'test_time_sec': float(te),
        }

        return edge_scores, metrics_dict

    def anomaly_detection(self, graphs, param_file):
        print("\n\nEstimating Edge Probability Distribution....")
        learning_rates = [self.args.alpha] #[0.1, 0.01, 0.001, 0.0001, 0.00001]
        support_sets = [self.args.support]#[2, 5, 15, 20, 25]
        i=0
        for s in support_sets:
            train_x, train_y = self.get_train_edges(graphs[:self.args.trainwin], self.args.support)
            for lr in learning_rates:
                f= open('./results/results.txt','a')
                f.write("lr: "+str(lr)+'\n')
                f.write("s: "+str(s)+'\n')
                f.close()
                self.args.alpha = lr
                self.args.support = s
                prob_param_file = param_file + '_' + str(self.args.alpha) + '_' + str(self.args.support) + '.pickle'
                print("++++++++++ Parameters +++++++ ")
                print("Learning Rate: ", lr)
                print("# of Support Set: ", s)
                print("Param File: ", prob_param_file)
                if self.args.train:
                    w = self.initialize_parameters(self.args.dimensions + self.kde_dim, len(self.node_list))

                    print("\n\nGenerating Training Edges....")
                    self.idx = 0
                    #train_x, train_y = self.get_train_edges(graphs[:self.args.trainwin], self.args.support)

                    print("\n\nStarting Gradient Descent....")
                    tr_time = timer()
                    parameters, costs = self.gradient_descent(w, train_x, train_y, self.args.iter, self.args.alpha)
                    train_time=timer() - tr_time
                    w = parameters["w"]

                    with open(prob_param_file, 'wb') as f:
                        pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

                with open(prob_param_file, 'rb') as f:
                    param = pickle.load(f)
                w = param['w']
                total_cpu = 1 # os.cpu_count()
                print("\nNumber of CPU Available: ", total_cpu)

                graph_tuple = [(w, self.node_embeddings[self.args.trainwin + idx], self.node_list, G) for idx, G in enumerate(graphs[self.args.trainwin:])]
                te_time = timer()
                '''
                with Pool(total_cpu) as pool:
                    all_graph_edges = pool.starmap(calculate_edge_probability, graph_tuple)
                pool.close()
                train_time=0
                '''
                train_time=0
                
                all_graph_edges=[]
                for i in tqdm(graph_tuple):
                    all_graph_edges.append(calculate_edge_probability(
                        i[0], i[1], i[2], i[3],
                        kde_dict=self.kde_dict,
                        reverse_node_map=self.reverse_node_map,
                        kde_dim=self.kde_dim,
                        kde_zero=self.kde_zero))
                 
                print("\nEdge Probability Estimation Completed...   [%s Sec.]" % (timer() - te_time))
                test_time=timer() - te_time
                edge_scores = [edges for g_edges in all_graph_edges for edges in g_edges]
                edge_scores = pd.DataFrame(edge_scores, columns=['src', 'dest',  'score', 'snapshot', 'time', 'label'])
                result_file = self.args.dataset + '_d' + str(self.args.dimensions) + 'all_users.csv'
                edge_scores_df, metrics_dict = self.calculate_performance_metrics(edge_scores, result_file,train_time,test_time)

        return edge_scores_df, metrics_dict