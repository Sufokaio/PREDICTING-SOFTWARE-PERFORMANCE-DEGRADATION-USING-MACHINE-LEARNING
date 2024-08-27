import argparse
from typing import Tuple
from unittest.result import failfast
import scipy.sparse as sp
import numpy as np
import random as rd
import itertools
from random_choice import randint_choice
from sklearn.model_selection import train_test_split

from util.setting import log
from util.load_data import Data

class GNNLoader(Data):
    def __init__(self, args:argparse.Namespace) -> None:
        super().__init__(args)
        print("wsh2")
        self.adj_type = args.adj_type

        self.curr_degradation_type = -1

        # generate sparse adjacency matrices for system entity inter_data
        log.info("Converting interactions into sparse adjacency matrix")
        adj_list = self._get_relational_adj_list(self.inter_data)

        # generate normalized (sparse adjacency) metrices
        log.info("Generating normalized sparse adjacency matrix")
        self.norm_list = self._get_relational_norm_list(adj_list)

        # load the norm matrix (used for information propagation)
        self.A_in = sum(self.norm_list)

        # mess_dropout
        self.mess_dropout = eval(args.mess_dropout)

        # split functions into training/validation/testing sets for code performance degradation prediction
        if args.perf_test_supervised:
            log.info('Generating code perf degradation prediction training, validation, and testing sets')
            self.perf_test_size = args.perf_test_size
            self.perf_val_size = args.perf_val_size

        # sample positive/negative pairs from code performance degradation prediction dataset
        if args.perf_test_unsupervised:
            log.info('Sampling positive and negative pairs for performance degradation')
            self.degr_pos_pairs, self.degr_neg_pairs = self._sample_perf_pair()
            self.n_perf_test_unsupervised = len(self.degr_pos_pairs)

        # sample functions from code clone dataset
        if args.cluster_test:
            #log.info('Sampling functions for code cluster')
            self.cluster_test_data = self._sample_cluster()
            self.n_cluster_test = len(self.cluster_test_data)

    def _get_relational_adj_list(self, inter_data) -> Tuple[list, list]:
        def _np_mat2sp_adj(np_mat:np.array, row_pre=0, col_pre=0) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
            # to-node interaction: A: A->B
            a_rows = np_mat[:, 0] + row_pre # all As
            a_cols = np_mat[:, 1] + col_pre # all Bs
            # must use float 1. (int is not allowed)
            a_vals = [1.] * len(a_rows)

            # from-node interaction: A: B->A
            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            # self.n_entity + 1: 
            # we add a `ghost` entity to support parallel AST node embedding 
            # retrival for program statements
            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(self.n_entity + 1, self.n_entity + 1))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(self.n_entity + 1, self.n_entity + 1))

            return a_adj, b_adj

        adj_mat_list = []

        r, r_inv = _np_mat2sp_adj(inter_data)
        adj_mat_list.append(r)
        # Todo: whether r_inv (inverse directions) helps infer code representations
        adj_mat_list.append(r_inv)

        return adj_mat_list
    
    def _get_relational_norm_list(self, adj_list:str) -> list:
        # Init for 1/Nt
        def _si_norm(adj):
            rowsum = np.array(adj.sum(axis=1))
            # np.power(rowsum, -1).flatten() may trigger divide by zero
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj

        # Init for 1/(Nt*Nh)^(1/2)
        def _bi_norm(adj):
            rowsum = np.array(adj.sum(axis=1))
            # np.power(rowsum, -1).flatten() may trigger divide by zero
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            # Different from KGAT's implementation
            # bi_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_norm

        if self.adj_type == 'bi':
            norm_list = [_bi_norm(adj) for adj in adj_list]
        else:
            norm_list = [_si_norm(adj) for adj in adj_list]

        return norm_list

    def _sample_cpg_split(self, inter_data:list) -> Tuple[list, list, list]:
        # we use the whole dataset to pretrain the model
        inter_train_data = inter_data

        _, inter_test_val_data = train_test_split(
            inter_data, test_size=self.ssl_test_size + self.ssl_val_size,
            random_state=2022)
        
        inter_val_data, inter_test_data = train_test_split(
            inter_test_val_data, test_size=self.ssl_test_size / (self.ssl_test_size + self.ssl_val_size),
            random_state=2022)

        return inter_train_data, inter_val_data, inter_test_data

    def sample_pos_pair(self, l, num):
        """randomly sample num (e.g., 100) non-repeated pairs from list (l) """
        pairs = itertools.combinations(l, 2)
        pool = tuple(pairs)
        n = len(pool)
        indices = sorted(rd.sample(range(n), num))
        return tuple(pool[i] for i in indices)

    def sample_neg_pair(self, l1, l2, num):
        "Random selection from itertools.product(*args, **kwds)"
        pairs = itertools.product(l1, l2)
        pool = tuple(pairs)
        n = len(pool)
        indices = sorted(rd.sample(range(n), num))
        return tuple(pool[i] for i in indices)

    def _sample_perf_pair(self, sample_degr_num=20000):
        """"""
        degr_pos_pairs = []
        degr_neg_pairs = []

        all_degr_family = self.all_degr_family

        exist_family_pair = [-1 for _ in range(self.classification_num)]
        for idx, family in enumerate(all_degr_family):
            while True:
                idx_neg = randint_choice(self.classification_num, size=1, replace=False)
                if idx_neg != idx and exist_family_pair[idx_neg] != idx:
                    exist_family_pair[idx] = idx_neg
                    break

            degr_pos_pairs_family = self.sample_pos_pair(family, sample_degr_num)
            degr_neg_pairs_family = self.sample_neg_pair(family, all_degr_family[idx_neg], sample_degr_num)
            
            degr_pos_pairs.extend(degr_pos_pairs_family)
            degr_neg_pairs.extend(degr_neg_pairs_family)
        
        log.debug('Code perf degr: #positive pairs {}'.format(len(degr_pos_pairs)))
        log.debug('Code perf degr: #negative pairs {}'.format(len(degr_neg_pairs)))

        assert(len(degr_pos_pairs) == len(degr_neg_pairs))

        return degr_pos_pairs, degr_neg_pairs  
    
    def _sample_cluster(self):
        """"""
        cluster_test_data = []

        all_degr_family = self.all_degr_family
        for idx, family in enumerate(all_degr_family):
            cluster_test_data.extend([[func, idx] for func in family])
            
        log.debug('Code Cluster: #samples {}'.format(len(cluster_test_data)))

        return cluster_test_data

    def _sample_classification_split(self):
        """"""
        class_train_data, class_val_data, class_test_data = [], [], []
        
        all_degr_family = self.all_degr_family

        for idx, family in enumerate(all_degr_family):
            class_train_family, class_test_val_family = train_test_split(
                family, test_size=self.class_test_size+self.class_val_size,
                random_state=2022)
        
            class_val_family, class_test_family = train_test_split(
                class_test_val_family, test_size=self.class_test_size/(self.class_test_size+self.class_val_size),
                random_state=2022)

            class_train_data.extend([[func, idx] for func in class_train_family])
            class_val_data.extend([[func, idx] for func in class_val_family])
            class_test_data.extend([[func, idx] for func in class_test_family])
        
        log.debug('Code Classification [n_train, n_val, n_test] = [%d, %d, %d]' % (len(class_train_data), len(class_val_data), len(class_test_data)))

        if self.split_label:
            f = open('oj_classification_split.txt', 'w')

            # training/validation/test: 0/1/2
            for i in range(len(class_train_data)):
                f.write("%d %d %d\n" % (class_train_data[i][0][0], class_train_data[i][1], 0))

            for i in range(len(class_val_data)):
                f.write("%d %d %d\n" % (class_val_data[i][0][0], class_val_data[i][1], 1))

            for i in range(len(class_test_data)):
                f.write("%d %d %d\n" % (class_test_data[i][0][0], class_test_data[i][1], 2))

            f.close()

        return class_train_data, class_val_data, class_test_data

    def _generate_perf_c4b_split(self, curr_degradation_type):
        self.curr_degradation_type = curr_degradation_type
        self.perf_train_data, self.perf_val_data, self.perf_test_date = self._sample_perf_c4b_split()
        self.n_perf_train, self.n_perf_val, self.n_perf_test_supervised = len(self.perf_train_data), len(self.perf_val_data), len(self.perf_test_date)
        # batch iter
        self.perf_data_iter = self.n_perf_train // self.batch_size_perf
        if self.n_perf_train % self.batch_size_perf:
            self.perf_data_iter += 1

    def _sample_perf_c4b_split(self):
        perf_train_data, perf_val_data, perf_test_date = [], [], []
        all_degradation_type = self.all_degradation_type
        for idx, degradation_type in enumerate(all_degradation_type):
            if (idx == 1 or idx == self.curr_degradation_type):

                # perf_train_family, perf_test_val_family = train_test_split(
                #     degradation_type, test_size=self.perf_test_size+self.perf_val_size,
                #     random_state=999)
                
                # perf_val_family, perf_test_family = train_test_split(
                #     perf_test_val_family, test_size=self.perf_test_size/(self.perf_test_size+self.perf_val_size),
                #     random_state=999)
                
                perf_train_family, perf_test_val_family = train_test_split(
                    degradation_type, test_size=self.perf_test_size+self.perf_val_size,
                    shuffle=False)
                
                perf_val_family, perf_test_family = train_test_split(
                    perf_test_val_family, test_size=self.perf_test_size/(self.perf_test_size+self.perf_val_size),
                    shuffle=False)

                perf_train_data.extend([[pair, [idx]] for pair in perf_train_family])
                perf_val_data.extend([[pair, [idx]] for pair in perf_val_family])
                perf_test_date.extend([[pair, [idx]] for pair in perf_test_family])

        log.debug('Performance degradation (Type %d) [n_train, n_val, n_test] = [%d, %d, %d]' % (self.curr_degradation_type, len(perf_train_data), len(perf_val_data) , len(perf_test_date)))

        if self.split_label:
            f = open('c4b_perf_split.txt', 'w')

            # training/validation/test: 0/1/2
            for i in range(len(perf_train_data)):
                f.write("%d %d %d %d\n" % (perf_train_data[i][0][0][0], perf_train_data[i][0][1][0], perf_train_data[i][1][0], 0))

            for i in range(len(perf_val_data)):
                f.write("%d %d %d %d\n" % (perf_val_data[i][0][0][0], perf_val_data[i][0][1][0], perf_val_data[i][1][0], 1))

            for i in range(len(perf_test_date)):
                f.write("%d %d %d %d\n" % (perf_test_date[i][0][0][0], perf_test_date[i][0][1][0], perf_test_date[i][1][0], 2))

            f.close()
        else:
            # Identify positives and negatives from different degradation types
            for i in range(len(perf_train_data)):
                if perf_train_data[i][1][0] == 1:
                    # True Negatives
                    perf_train_data[i][1][0] = 0
                else:
                    # True Positive
                    perf_train_data[i][1][0] = 1

            for i in range(len(perf_val_data)):
                if perf_val_data[i][1][0] == 1:
                    # True Negatives
                    perf_val_data[i][1][0] = 0
                else:
                    # True Positive
                    perf_val_data[i][1][0] = 1

            for i in range(len(perf_test_date)):
                if perf_test_date[i][1][0] == 1:
                    # True Negatives
                    perf_test_date[i][1][0] = 0
                else:
                    # True Positive
                    perf_test_date[i][1][0] = 1

        return perf_train_data, perf_val_data, perf_test_date

    def transfer_s_to_e(self, s_list:list) -> np.array:
        """"""
        s2e = []
        for s in s_list:
            s2e.append(self.stat_dict[s])
        return np.array(s2e)

    def _convert_csr_to_sparse_tensor_inputs(self, X:sp.csr_matrix) -> Tuple[list, list, list]:
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def extract_stat_from_inter(self, inter_data:list) -> Tuple[list, list]:
        s1 = []
        s2 = []
        
        for pair in inter_data:
            s1.append(pair[0])
            s2.append(pair[1])

        return s1, s2

    def generate_perf_train_batch(self, i_batch:int) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_perf
        if i_batch == self.perf_data_iter - 1:
            end = self.n_perf_train
        else:
            end = (i_batch + 1) * self.batch_size_perf
        
        f_y = self.perf_train_data[start: end]

        f1_perf = [f[0][0] for f in f_y]
        f2_perf = [f[0][1] for f in f_y]
        y_perf = [f[1] for f in f_y]

        batch_data['f1_perf'] = f1_perf
        batch_data['f2_perf'] = f2_perf
        batch_data['y_perf'] = y_perf

        return batch_data

    def generate_perf_val_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_perf
        if last_batch:
            end = self.n_perf_val
        else:
            end = (i_batch + 1) * self.batch_size_perf

        f_y = self.perf_val_data[start: end]

        f1_perf = [f[0][0] for f in f_y]
        f2_perf = [f[0][1] for f in f_y]
        y_perf = [f[1] for f in f_y]

        batch_data['f1_perf'] = f1_perf
        batch_data['f2_perf'] = f2_perf
        batch_data['y_perf'] = y_perf

        return batch_data

    def generate_perf_test_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_perf
        if last_batch:
            end = self.n_perf_test_supervised
        else:
            end = (i_batch + 1) * self.batch_size_perf

        f_y = self.perf_test_date[start: end]

        f1_perf = [f[0][0] for f in f_y]
        f2_perf = [f[0][1] for f in f_y]
        y_perf = [f[1] for f in f_y]

        batch_data['f1_perf'] = f1_perf
        batch_data['f2_perf'] = f2_perf
        batch_data['y_perf'] = y_perf

        return batch_data

    def generate_perf_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.f1_perf: batch_data['f1_perf'],
            model.f2_perf: batch_data['f2_perf'],
            model.y_perf: batch_data['y_perf'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0]
        }
        return feed_dict

    def generate_classification_train_batch(self, i_batch:int) -> dict:
        """"""
        batch_data = {}

        start = i_batch * self.batch_size_classification
        if i_batch == self.classification_data_iter:
            end = self.n_classification_train
        else:
            end = (i_batch + 1) * self.batch_size_classification

        f_y = self.classification_train_data[start: end]

        batch_data['f_classification'] = [f[0] for f in f_y]
        # labels, e.g., 2
        batch_data['y_classification'] = [f[1] for f in f_y]
        batch_data['y_classification'] = np.array(batch_data['y_classification']).flatten()

        return batch_data

    def generate_classification_val_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_classification
        if last_batch:
            end = self.n_classification_val
        else:
            end = (i_batch + 1) * self.batch_size_classification

        f_y = self.classification_val_data[start: end]

        batch_data['f_classification'] = [f[0] for f in f_y]
        # labels, e.g., 2
        batch_data['y_classification'] = [f[1] for f in f_y]

        return batch_data

    def generate_classification_test_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_classification
        if last_batch:
            end = self.n_classification_test
        else:
            end = (i_batch + 1) * self.batch_size_classification

        f_y = self.classification_test_data[start: end]

        batch_data['f_classification'] = [f[0] for f in f_y]
        # labels, e.g., 2
        batch_data['y_classification'] = [f[1] for f in f_y]

        return batch_data

    def generate_classification_train_feed_dict(self, model, batch_data):
        feed_dict = {
            # Classification
            model.f_classification: batch_data['f_classification'],
            model.y_classification: batch_data['y_classification'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        
        return feed_dict

    def generate_classification_val_feed_dict(self, model, batch_data):
        feed_dict = {
            model.f_classification: batch_data['f_classification'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict

    def generate_perf_batch(self, i_batch:int, last_batch:bool, pos:bool) -> dict:
        batch_data = {}

        start = i_batch * self.batch_size_perf
        if last_batch:
            end = self.n_perf_test_unsupervised
        else:
            end = (i_batch + 1) * self.batch_size_perf
        
        if pos:
            f1_perf = [pair[0] for pair in self.degr_pos_pairs[start:end]]
            f2_perf = [pair[1] for pair in self.degr_pos_pairs[start:end]]
        else:
            f1_perf = [pair[0] for pair in self.degr_neg_pairs[start:end]]
            f2_perf = [pair[1] for pair in self.degr_neg_pairs[start:end]]

        batch_data['f1_perf'] = f1_perf
        batch_data['f2_perf'] = f2_perf

        return batch_data

    def generate_perf_feed_dict(self, model, batch_data):
        feed_dict = {
            model.f1_perf: batch_data['f1_perf'],
            model.f2_perf: batch_data['f2_perf'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict

    def generate_cluster_batch(self, i_batch:int, last_batch:bool) -> dict:
        batch_data = {}
        
        start = i_batch * self.batch_size_cluster
        if last_batch:
            end = self.n_cluster_test
        else:
            end = (i_batch + 1) * self.batch_size_cluster

        f_y= self.cluster_test_data[start:end]

        batch_data['f_cluster'] = [f[0] for f in f_y]
        batch_data['cluster_label'] = [f[1] for f in f_y]

        return batch_data
    
    def generate_cluster_feed_dict(self, model, batch_data):
        feed_dict = {
            model.f_cluster: batch_data['f_cluster'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict
