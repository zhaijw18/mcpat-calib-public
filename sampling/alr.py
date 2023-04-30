import math

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from progressbar import *


class PowerGS():
    def __init__(
        self,
        ml_model,
        train_data,
        cb_labels,
        init_nums,
        labeled_nums,
    ):
        self.ml_model = ml_model
        self.init_nums = init_nums
        self.labeled_nums = labeled_nums

        self.train_data = train_data
        self.labeled_data = np.array([])
        self.unlabeled_data = train_data

        self.train_cb_labels = cb_labels
        self.labeled_cb_labels = np.array([])
        self.unlabeled_cb_labels = cb_labels
        self.config_num = np.max(cb_labels[:, 0]) + 1
        self.c_count = [0] * self.config_num
        self.benchmark_num = np.max(cb_labels[:, 1]) + 1
        self.b_count = [0] * self.benchmark_num
        

    def sampling(self, progress):
        init_idx, rest_idx = self.get_init_samples()
        self.labeled_data = self.train_data[init_idx]
        self.unlabeled_data = self.train_data[rest_idx]
        self.labeled_cb_labels = self.train_cb_labels[init_idx]
        self.unlabeled_cb_labels = self.train_cb_labels[rest_idx]
        
        for i in range(0, self.config_num):
            self.c_count[i] += list(self.labeled_cb_labels[:, 0]).count(i)

        for i in range(0, self.benchmark_num):
            self.b_count[i] += list(self.labeled_cb_labels[:, 1]).count(i)

        widgets = [progress, Percentage(), ' ', Bar('#'),' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=10*self.labeled_nums).start()

        while (self.labeled_data.shape[0] < self.labeled_nums):
            pbar.update(10 * self.labeled_data.shape[0] + 1)
            cbxy_dis = self.get_PowerGS_dis()
            self.get_new_sample(cbxy_dis)

        pbar.finish()

        return self.labeled_cb_labels[:, -1]
        

    def get_init_samples(self):
        kmeans = KMeans(n_clusters=self.init_nums, random_state=9)
        kmeans.fit_predict(self.train_data[:, :-1])
        centers = kmeans.cluster_centers_
        init_idx = []
        for i in range(0, centers.shape[0]):
            idx = -1
            minDist = np.inf
            for j in range(0, self.train_data[:, :-1].shape[0]):
                temp_dis = np.linalg.norm(centers[i] - self.train_data[:, :-1][j])
                if temp_dis < minDist:
                    minDist, idx = temp_dis, j
            init_idx.append(idx)
        rest_idx = [x for x in range(0, self.train_data[:, :-1].shape[0]) if not x in init_idx] 

        return init_idx, rest_idx
    

    def get_new_sample(self, cbxy_dis):
        orders = np.array(cbxy_dis).argsort()[::-1]
        # new_sample = self.unlabeled_data[orders[0]]
        self.labeled_data = np.insert(self.labeled_data, self.labeled_data.shape[0], self.unlabeled_data[orders[0]], axis=0)
        self.unlabeled_data = np.delete(self.unlabeled_data, orders[0], axis=0)
        self.labeled_cb_labels = np.insert(self.labeled_cb_labels, self.labeled_cb_labels.shape[0], self.unlabeled_cb_labels[orders[0]], axis=0)
        self.c_count[self.unlabeled_cb_labels[orders[0]][0]] += 1
        self.b_count[self.unlabeled_cb_labels[orders[0]][1]] += 1
        self.unlabeled_cb_labels = np.delete(self.unlabeled_cb_labels, orders[0], axis=0)


    def get_x_dis(self, labeled_feature, unlabeled_feature):
        dis = cdist(unlabeled_feature, labeled_feature, metric='euclidean')
        dis_list = []
        for i in range(0, dis.shape[0]):
            dis_list.append(min(dis[i]))
        
        return dis_list


    def get_y_dis(self):
        self.ml_model.fit(self.labeled_data[:, :-1], self.labeled_data[:, -1])
        pred = self.ml_model.predict(self.unlabeled_data[:, :-1])
        dis_list = self.get_x_dis(self.labeled_data[:, -1].reshape(-1,1), np.array(pred).reshape(-1,1))
        
        return dis_list


    def get_PowerGS_dis(self):
        y_dis = self.get_y_dis()
        x_dis = self.get_x_dis(self.labeled_data[:, :-1], self.unlabeled_data[:, :-1])

        c_weight = np.ones((self.unlabeled_data.shape[0], ))
        b_weight = np.ones((self.unlabeled_data.shape[0], ))
        
        c_weight = [1-1/(1+math.exp(-self.c_count[i])) for i in self.unlabeled_cb_labels[:, 0]]
        b_weight = [1-1/(1+math.exp(-self.b_count[i])) for i in self.unlabeled_cb_labels[:, 1]]

        cbxy_dis = np.multiply(np.array(y_dis), np.array(x_dis))
        cbxy_dis = np.multiply(np.array(c_weight), np.array(cbxy_dis))
        cbxy_dis = np.multiply(np.array(b_weight), np.array(cbxy_dis))

        return cbxy_dis.tolist()
