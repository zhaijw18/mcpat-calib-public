import argparse
import sys

import numpy as np
import yaml
from sklearn.model_selection import KFold

from calibration.calib import Calib, Calib_data
from flow import modeling_flow
from sampling.alr import PowerGS


def get_configs(fyaml):
    with open(fyaml, 'r') as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            configs = yaml.load(f)
    return configs


def parse_args():
    def initialize_parser(parser):
        parser.add_argument('-c', '--configs',
            required=True,
            type=str,
            default='example.yml',
            help='YAML file to be handled')
        return parser

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = initialize_parser(parser)
    return parser.parse_args()


def mape(y_true, y_pred):
    return np.mean(np.abs((np.array(y_pred) - np.array(y_true)) / np.array(y_true))) * 100


def al_sampling_test(calib_data, alr_model, percentage):

    ml_model = alr_model
    dynamic_mape_list = []
    total_mape_list = []
    
    iter = 0
    kf = KFold(n_splits=calib_data.config_num)
    print("[INFO]: Config-Split CV, Sampling PCT = {:.2f}%".format(percentage*100))
    for train_index, test_index in kf.split(calib_data.dynamic_targets):
        train_X, test_X = calib_data.selected_features[train_index], calib_data.selected_features[test_index]
        train_Y, test_Y = calib_data.dynamic_targets[train_index], calib_data.dynamic_targets[test_index]
        total_X, total_Y = calib_data.total_targets[train_index], calib_data.total_targets[test_index]
        leak_X, leak_Y = calib_data.leakage_pred[train_index], calib_data.leakage_pred[test_index]
        
        sample_idx = np.arange(0, train_X.shape[0]).reshape(-1, 1)
        cb_label = np.hstack((calib_data.config_benchmark_label[train_index, 0:2], sample_idx))

        train_data = np.hstack((train_X, train_Y.reshape(-1, 1)))
        labled_nums = percentage * train_data.shape[0]

        powergs = PowerGS(ml_model, train_data, cb_label, train_data.shape[1]-1, labled_nums)
        progress = "Split %s: " % iter
        sample_idx = powergs.sampling(progress)

        ml_model.fit(train_data[sample_idx, :-1], train_data[sample_idx, -1])
        test_pred = ml_model.predict(test_X)
        total_pred = np.sum([leak_Y, test_pred], axis = 0)
        
        dynamic_mape = mape(test_Y, test_pred)
        dynamic_mape_list.append(dynamic_mape)
        total_mape = mape(total_Y, total_pred)
        total_mape_list.append(total_mape)
        print("Split {}: Test MPAE = {:.2f}%".format(iter, total_mape))
        iter += 1

    print("[INFO]: Config-Split, Sampling PCT = {:.2f}%, Average Test MAPE = {:.2f}%".format(percentage*100, np.average(total_mape_list)))   
        

def calib_test(configs):
    calib_data = Calib_data(configs)
    calib_data.feature_processing()

    calib = Calib(configs)

    mcpat_mape = mape(calib_data.total_targets, calib_data.mcpat_total)
    print("[INFO]: McPAT-7nm Resluts")
    print("Total Samples:  MAPE = {:.2f}%".format(mcpat_mape))

    ### Leakage Power Calibration
    calib_data = calib.leakage_calib(calib_data)
    calib.save_leakage_model(calib_data)

    ### Dynamic Power Calibration
    # ### Total Features
    # calib_data = calib.dynamic_calib(calib_data, feature=calib_data.dynamic_features)

    ### Selected Features
    feature_num = configs["feature-num"]
    calib_data.feature_selection(calib_data.dynamic_features, calib_data.dynamic_features_name, feature_num)
    calib_data = calib.dynamic_calib(calib_data, feature=calib_data.selected_features)
    calib.save_dynamic_model(calib_data)


    ### Total Power Calibration
    calib_data.compute_total_power()
    calib_data.save_modeling_results()

    ### AL Sampling
    print("\n[INFO]: Simulate AL sampling flow, please wait patiently")
    alr_model = calib.dynamic_model
    percentage = configs["sampling-percentage"]
    al_sampling_test(calib_data, alr_model, percentage)


if __name__ =='__main__':
    configs = get_configs(parse_args().configs)
    mode = configs["mode"]
    if mode == "estimation":
        print("[INFO]: Perform power estimation using the calibrated models.\n")
        parser = modeling_flow(configs)
        parser.gem5_to_mcpat()
        parser.mcpat_to_calib()
    elif mode == "train":
        print("[WARN]: Please replace the demo data with your own data.\n")
        calib_test(configs)
    else:
        raise NotImplementedError("Illegal operation!")
    sys.exit()


