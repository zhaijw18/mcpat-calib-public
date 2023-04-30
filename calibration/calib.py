import os
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold

from calibration.feature_processing import feature_selection, get_features
from calibration.ml_model import init_dynamic_model, init_leakage_model

warnings.filterwarnings('ignore')
try:
    from sklearn.externals import joblib
except ImportError:
    import joblib


def load_name(file_name):
    name = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')  
            name.append(line)
    return name


def mape(y_true, y_pred):
    return np.mean(np.abs((np.array(y_pred) - np.array(y_true)) / np.array(y_true))) * 100


def shuffle_validation(model, targets, features, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True)
    ml_shuffle = np.zeros(shape=(features.shape[0],))
    for train_index, test_index in kf.split(targets):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        model.fit(X_train, y_train.ravel())
        shuffle_pred = model.predict(X_test)
        ml_shuffle[test_index] = shuffle_pred
    MAPE = mape(targets, ml_shuffle)
    r2 = metrics.r2_score(targets, ml_shuffle)

    return ml_shuffle, MAPE, r2


def print_Shuffle(MAPE, r2):
    print("Shuffle-Split:  MAPE = {:.2f}%,  R Square = {:.3f}".format(MAPE, r2))


def config_validation(model, config_label, targets, features):
    n_splits = np.max(config_label)+1
    kf = KFold(n_splits=n_splits)
    ml_config = np.zeros(shape=(features.shape[0],))
    for train_index, test_index in kf.split(targets):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        model.fit(X_train, y_train.ravel())       
        config_pred = model.predict(X_test)
        ml_config[test_index] = config_pred
    MAPE = mape(targets, ml_config)
    r2 = metrics.r2_score(targets, ml_config)

    return ml_config, MAPE, r2


def print_config(MAPE, r2):
    print("Config-Split:   MAPE = {:.2f}%,  R Square = {:.3f}".format(MAPE, r2))


def benchmark_validation(model, targets, features, n_splits):
    total_sample_num = features.shape[0]
    split_space = total_sample_num // n_splits
    ml_benchmark = np.zeros(shape=(total_sample_num,))
    for i in range(0, n_splits):
        test_index = [x * n_splits + i  for x in range(0, split_space)]
        train_index = [x for x in range(0, total_sample_num) if not x in test_index]
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        model.fit(X_train, y_train.ravel())
        benchmark_pred = model.predict(X_test)
        ml_benchmark[test_index] = benchmark_pred
    MAPE = mape(targets, ml_benchmark)
    r2 = metrics.r2_score(targets, ml_benchmark)

    return ml_benchmark, MAPE, r2

def print_Benchmark(MAPE, r2):
    print("Benchmark-Splt: MAPE = {:.2f}%,  R Square = {:.3f}".format(MAPE, r2))


class Calib_data():
    def __init__(
        self, configs
    ):

        self.configs = configs
        self.gt_data = pd.read_csv(os.path.join(configs["dataset-path"], "power-demo.csv"))
        self.total_targets = np.array(self.gt_data["Total Power"]).astype(np.float64)
        self.leakage_targets = np.array(self.gt_data["Leak Power"]).astype(np.float64)
        self.dynamic_targets = np.array(self.gt_data["Dynamic Power"]).astype(np.float64)
        self.gt_name = self.gt_data["config-benchmark"]
        self.mcpat_leakage = self.gt_data['McPAT.Leakage'].astype(np.float64)
        self.mcpat_dynamic = self.gt_data['McPAT.Dynamic'].astype(np.float64)
        self.mcpat_total = self.gt_data['McPAT.Total'].astype(np.float64)

        self.feature_data = pd.read_csv(os.path.join(configs["dataset-path"], "feature-demo.csv"))
        self.config_label = np.array(self.feature_data["Config_ID"]).astype(np.int64).reshape(-1, 1)
        self.benchmark_label = np.array(self.feature_data["Bench_ID"]).astype(np.int64).reshape(-1, 1)
        self.benchmark_num, self.config_num = np.max(self.benchmark_label)+1, np.max(self.config_label)+1
        self.total_sample_num = self.benchmark_num * self.config_num

        self.sample_index = np.arange(0, self.total_sample_num).reshape(self.total_sample_num, 1)
        self.config_benchmark_label = np.hstack((self.config_label, self.benchmark_label))
        self.config_benchmark_label = np.hstack((self.config_benchmark_label, self.sample_index))

        self.dynamic_shuffle_pred = np.zeros((self.total_sample_num, ))
        self.dynamic_config_pred = np.zeros((self.total_sample_num, ))
        self.dynamic_benchmark_pred = np.zeros((self.total_sample_num, ))
        self.total_shuffle_pred = np.zeros((self.total_sample_num, ))
        self.total_config_pred = np.zeros((self.total_sample_num, ))
        self.total_benchmark_pred = np.zeros((self.total_sample_num, ))
        self.leakage_pred = np.zeros((self.total_sample_num, ))
        self.leakage_config_pred = np.zeros((self.config_num, ))

    def feature_processing(self):
        self.dynamic_features, self.dynamic_features_name, self.leakage_features, self.leakage_feature_name = get_features(self.feature_data)
        self.mcpat_features, self.mcpat_features_name   = self.dynamic_features[:, 0:38],  self.dynamic_features_name[0:38]
        self.config_features, self.config_features_name = self.dynamic_features[:, 38:56], self.dynamic_features_name[38:56]
        self.event_features, self.event_features_name   = self.dynamic_features[:, 56:],   self.dynamic_features_name[56:]

        self.leakage_config_targets = np.zeros((self.config_num, ))
        self.leakage_config_features = np.zeros((self.config_num, 2))
        for i in range(self.config_num):
            self.leakage_config_targets[i] = np.sum(self.leakage_targets[i*self.benchmark_num : (i+1)*self.benchmark_num]) / self.benchmark_num
            self.leakage_config_features[i][1] = np.sum(self.leakage_features[i*self.benchmark_num : (i+1)*self.benchmark_num, 0]) / self.benchmark_num
            self.leakage_config_features[i][0] = np.sum(self.leakage_features[i*self.benchmark_num : (i+1)*self.benchmark_num, 1]) / self.benchmark_num

    def feature_selection(self, feature, feature_name, selected_num):
        my_mape = metrics.make_scorer(mape, greater_is_better=False)
        selected_idx, self.selected_features = feature_selection(feature, self.dynamic_targets, selected_num, my_mape, 0)
        self.selected_features_name = np.array(feature_name)[list(selected_idx)]


    def compute_total_power(self):
        print("\n[INFO]: Total Power Calibration")
        self.total_shuffle_pred = np.sum([self.leakage_pred, self.dynamic_shuffle_pred], axis = 0)
        print_Shuffle(mape(self.total_targets, self.total_shuffle_pred), metrics.r2_score(self.total_targets, self.total_shuffle_pred))
        self.total_config_pred = np.sum([self.leakage_pred, self.dynamic_config_pred], axis = 0)
        print_config(mape(self.total_targets, self.total_config_pred), metrics.r2_score(self.total_targets, self.total_config_pred))
        self.total_benchmark_pred = np.sum([self.leakage_pred, self.dynamic_benchmark_pred], axis = 0)
        print_Benchmark(mape(self.total_targets, self.total_benchmark_pred), metrics.r2_score(self.total_targets, self.total_benchmark_pred))


    def save_modeling_results(self):
        calib_result={
                "config-benchmark": self.gt_name,
                "GT.Leakage": self.leakage_targets,
                "GT.Dynamic": self.dynamic_targets,
                "GT.Total": self.total_targets,
                "McPAT-7nm.Leakage": self.mcpat_leakage,
                "McPAT-7nm.Dynamic": self.mcpat_dynamic,
                "McPAT-7nm.Total": self.mcpat_total,
                "ML.Leakage": self.leakage_pred,
                "ML-Shuffle.Dynamic": self.dynamic_shuffle_pred,
                "ML-Shuffle.Total": self.total_shuffle_pred,
                "ML-Config.Dynamic": self.dynamic_config_pred,
                "ML-Config.Total": self.total_config_pred,
                "ML-Benchmark.Dynamic": self.dynamic_benchmark_pred,
                "ML-Benchmark.Total": self.total_benchmark_pred
                }
        data = pd.core.frame.DataFrame(calib_result)
        data.to_csv(os.path.join(self.configs["results-path"], "modeling-results.csv"), index=0)


class Calib():
    def __init__(
        self,
        configs
    ):
        self.configs = configs
        self.leakage_model = init_leakage_model(configs["leakage-calib-model"])
        self.dynamic_model = init_dynamic_model(configs["dynamic-calib-model"])

    def leakage_calib(self, calib_data):
        calib_data.leakage_config_pred, MAPE, r2 = \
            config_validation(self.leakage_model, calib_data.config_label,  calib_data.leakage_config_targets, calib_data.leakage_config_features)
        print("\n[INFO]: Leakage Power Calibration")
        print_config(MAPE, r2)

        calib_data.leakage_pred = np.array([val for val in calib_data.leakage_config_pred for i in range(calib_data.benchmark_num)])

        return calib_data


    def save_leakage_model(self, calib_data):
        self.leakage_model.fit(calib_data.leakage_config_features, calib_data.leakage_config_targets)
        joblib.dump(
                self.leakage_model,
                os.path.join(self.configs["results-path"], "cailb-models", "leakage-calib.pt")
        )
       
    
    def dynamic_calib(self, calib_data, feature):
        print("\n[INFO]: Dynamic Power Calibration")
        calib_data.dynamic_shuffle_pred, MAPE, r2 = shuffle_validation(self.dynamic_model, calib_data.dynamic_targets, feature, 10)
        print_Shuffle(MAPE, r2)
        calib_data.dynamic_config_pred, MAPE, r2 = config_validation(self.dynamic_model, calib_data.config_label, calib_data.dynamic_targets, feature)
        print_config(MAPE, r2)
        calib_data.dynamic_benchmark_pred, MAPE, r2 = benchmark_validation(self.dynamic_model, calib_data.dynamic_targets, feature, 10)
        print_Benchmark(MAPE, r2)

        return calib_data


    def save_dynamic_model(self, calib_data):
        self.dynamic_model.fit(calib_data.selected_features, calib_data.dynamic_targets)
        joblib.dump(
                self.dynamic_model,
                os.path.join(self.configs["results-path"], "cailb-models", "dynamic-calib.pt")
        )
