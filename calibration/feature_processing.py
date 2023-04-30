import os
from select import select

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_features(feature_data_pd):
    feature_data = np.array(feature_data_pd)

    dynamic_features_name = []
    leakage_features_name = []

    mcapt_area = feature_data[:, 3:4]
    dynamic_features = mcapt_area
    dynamic_features_name.append(feature_data_pd.columns.values[3])
    leakage_features = mcapt_area
    leakage_features_name.append(feature_data_pd.columns.values[3])
    
    mcpat_leakage = feature_data[:, 4:5] * 1000
    dynamic_features = np.hstack((dynamic_features, mcpat_leakage))
    dynamic_features_name.append(feature_data_pd.columns.values[4])
    leakage_features = np.hstack((leakage_features, mcpat_leakage))
    leakage_features_name.append(feature_data_pd.columns.values[4])

    mcpat_dynamic_start = 5
    mcpat_dynamic_num = 36
    mcpat_dynamic = feature_data[:, mcpat_dynamic_start:mcpat_dynamic_start+mcpat_dynamic_num] * 1000
    dynamic_features = np.hstack((dynamic_features, mcpat_dynamic))
    dynamic_features_name = dynamic_features_name + feature_data_pd.columns.values[5:41].tolist()

    config_start = 41
    config_num = 18
    config = feature_data[:, config_start:config_start+config_num]
    dynamic_features = np.hstack((dynamic_features, config))
    dynamic_features_name = dynamic_features_name + feature_data_pd.columns.values[config_start:config_start+config_num].tolist()
    
    stats_start = 59
    stats = feature_data[:, stats_start:stats_start+3]
    stats = np.hstack((stats, 100 * feature_data[:, stats_start+3:] / feature_data[:, stats_start+2:stats_start+3]))
    dynamic_features = np.hstack((dynamic_features, stats))
    dynamic_features_name = dynamic_features_name + feature_data_pd.columns.values[stats_start:].tolist()

    return dynamic_features.astype(np.float), dynamic_features_name, leakage_features.astype(np.float), leakage_features_name



def feature_plot(dynamic_features, dynamic_features_name, dynamic_targets, dirname):
    r_list = [ pearsonr(dynamic_targets, dynamic_features[:, i])[0]  for i in range(0, dynamic_features.shape[1])]
    VIF_list = [variance_inflation_factor(dynamic_features, i) for i in range(dynamic_features.shape[1])]
    VIF_list = np.array(VIF_list).reshape(dynamic_features.shape[1],)
    VIF_list[np.where(VIF_list>100)] = 100

    index = np.arange(len(dynamic_features_name))
    bar_width = 0.5

    mcpat_index = range(0, 38)
    config_index = range(38, 56)
    stats_index = range(56, 146)
    # mcpat_index = range(0, 5)
    # stats_index = range(5, 17)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18,5)) 

    ax1 = ax[0]
    ax1.set_ylabel("Correlation", fontsize=12) 
    ax1.bar(index[mcpat_index], np.abs(r_list)[mcpat_index], width=bar_width, label='McPAT')
    ax1.bar(index[config_index], np.abs(r_list)[config_index], width=bar_width, label='Params')
    ax1.bar(index[stats_index], np.abs(r_list)[stats_index], width=bar_width, label='Stats')
    ax1.legend(fontsize=7)

    ax2 = ax[1]
    ax2.set_ylabel("VIF", fontsize=12) 
    ax2.bar(index[mcpat_index], np.abs(VIF_list)[mcpat_index], width=bar_width, label='McPAT')
    ax2.bar(index[config_index], np.abs(VIF_list)[config_index], width=bar_width, label='Params')
    ax2.bar(index[stats_index], np.abs(VIF_list)[stats_index], width=bar_width, label='Stats')
    ax2.legend(fontsize=7)
   

    selectd_features_name = ['Core.Leakage', 'Core.Dynamic', 'Free_List.Dynamic', 'MMU.Dynamic',
        'ROB.Dynamic', 'iew.exec_stores', 'int_alu_accesses', 'FU_FpMemRead',
        'FU_IntDiv', 'FU_FpMult', 'FU_FpDiv', 'mem.conflictStores', 'rename.Maps',
        'mem_ctrls.reads', 'icache.mshr_hits', 'dcache.accesses', 'dcache.mshr_hits']

    # selectd_idx = [x for x in range(0, len(dynamic_features_name)) if dynamic_features_name[x] in selectd_features_name]
    # other_idx = [x for x in index if x not in selectd_idx]

    for idx in range(0, len(dynamic_features_name)):
        if dynamic_features_name[idx] in selectd_features_name:
            dynamic_features_name[idx] = dynamic_features_name[idx].replace('_', '\_')
            dynamic_features_name[idx] = r"$\bf{" + str(dynamic_features_name[idx]) + "}$"


    plt.xticks(index, labels=dynamic_features_name, rotation=270, fontsize=6.2)
    plt.subplots_adjust(bottom=0.32)
    plt.savefig(os.path.join(dirname, "feature_corr_vif.pdf"), bbox_inches='tight')
    plt.close()


def feature_selection(total_features, targets, k_features, scoring, var_threshold=0):
    ### Feature Filtering
    var_sel = VarianceThreshold(threshold=var_threshold)
    var_sel.fit(total_features)
    slected_idx = var_sel.get_support()
    filter_features = total_features[:, slected_idx]

    ### Sequential Feature Selection
    ridge_model = Ridge(alpha=0.001)
    ridge_sfs = SFS(estimator=ridge_model, k_features=k_features, forward=True, scoring=scoring, cv=10)
    ridge_sfs.fit(filter_features, targets)

    slected_idx = [i for i, j in enumerate(slected_idx) if j == True]
    slected_idx = np.array(slected_idx)[list(ridge_sfs.k_feature_idx_)]
    select_fatures = filter_features[:, ridge_sfs.k_feature_idx_]

    return slected_idx, select_fatures

