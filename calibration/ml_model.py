from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, Ridge)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def init_dynamic_model(model_name):
    ### The parameters of the ML model need to be adjusted according to the data set.
    if model_name == "LinearRegression":
        model = LinearRegression()
        # model = LinearRegression(fit_intercept=True, normalize=False, n_jobs=12)
    elif model_name == "Lasso":
        model = Lasso()
        # model = Lasso(alpha=0.001, max_iter=10000)
    elif model_name == "Ridge":
        model = Ridge()
        # model = Ridge(alpha=0.001)
    elif model_name == "ElasticNet":
        model = ElasticNet()
        # model = ElasticNet(alpha=0.001, l1_ratio=0.001, max_iter=10000, 
        #     normalize=False, fit_intercept=True, positive=False)
    elif model_name == "BayesianRidge":
        model = BayesianRidge()
        # model = BayesianRidge(alpha_1=5000, alpha_2=8000, n_iter=50, verbose=True)
    elif model_name == "GaussianProcessRegressor":
        gp_kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=gp_kernel)
        # model = GaussianProcessRegressor(kernel=gp_kernel, alpha=0.05)
    elif model_name == "KNeighborsRegressor":
        model = KNeighborsRegressor()
        # model = KNeighborsRegressor(n_neighbors=3, leaf_size=3, p=1, 
        #     weights='distance',  algorithm='auto', n_jobs=12)
    elif model_name == "LinearSVR":
        model = LinearSVR()
    elif model_name == "PolySVR_model": 
        model = SVR(kernel='poly')
        # model = SVR(kernel='poly', C=1000, degree=1, tol=5e-3, max_iter=100000)
    elif model_name == "RBFSVR_model":
        model = SVR(kernel='rbf')
        # model = SVR(kernel='rbf', C=1000, degree=1, tol=5e-3, max_iter=100000)
    elif model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor()
        # model = DecisionTreeRegressor(criterion='squared_error', max_depth=15, 
        #     max_features='sqrt', splitter='best')
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor()
        # model = RandomForestRegressor(criterion='squared_error', max_depth=4, 
        #     max_features='log2', n_estimators=100, n_jobs=12)
    elif model_name == "AdaBoostRegressor":
        model = AdaBoostRegressor()
        # model = AdaBoostRegressor(loss='square', n_estimators=300)
    elif model_name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor()
        # model = GradientBoostingRegressor(n_estimators=400, ccp_alpha=0.001)
    elif model_name == "BaggingRegressor":
        model = BaggingRegressor()
        # model = BaggingRegressor(max_features=50, n_estimators=600)
    elif model_name == "XGBRegressor":
        model = XGBRegressor()
        # model = XGBRegressor(reg_alpha=5.25, reg_lambda=8, gamma=0, min_child_weight=1, colsample_bytree=1,
        #         learning_rate=0.02, max_depth=5, n_estimators=11000, subsample=0.035, n_jobs=8)
        
    return model


def init_leakage_model(model_name):
    if model_name == "LinearRegression":
        model = LinearRegression()
        # model = LinearRegression(fit_intercept=False, n_jobs=12)
    elif model_name == "Lasso":
        model = Lasso()
        # model = Lasso(alpha=0.01, max_iter=10000)
    elif model_name == "Ridge":
        odel = Ridge()
        # model = Ridge(alpha=0.01, fit_intercept=False)
    elif model_name == "ElasticNet":
        model = ElasticNet()
        # model = ElasticNet(alpha=0.005, l1_ratio=1, max_iter=10000, 
        #     fit_intercept=False, positive=True)
    elif model_name == "BayesianRidge":
        model = BayesianRidge()
        # model = BayesianRidge(alpha_1=0.0001, alpha_2=5, n_iter=20, verbose=True)
    elif model_name == "GaussianProcessRegressor":
        gp_kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=gp_kernel)
        # model = GaussianProcessRegressor(kernel=gp_kernel, alpha=0.1)
    elif model_name == "KNeighborsRegressor":
        model = KNeighborsRegressor()
        # model = KNeighborsRegressor(n_neighbors=4, leaf_size=2, p=1, 
        #     weights='distance',  algorithm='auto', n_jobs=12)
    elif model_name == "LinearSVR":
        model = LinearSVR()
        # model = LinearSVR(C=10, max_iter=100000)
    elif model_name == "PolySVR": 
        model = SVR(kernel='poly')
        # model = SVR(kernel='poly', C=2, degree=2)
    elif model_name == "RBFSVR":
        model = SVR(kernel='rbf')
        model = SVR(kernel='rbf', C=500, degree=1)
    elif model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor()
        # model = DecisionTreeRegressor(criterion='squared_error', max_depth=5, 
        #     max_features='sqrt', splitter='random')
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor()
        # model = RandomForestRegressor(criterion='squared_error', max_depth=20, 
        #     max_features='sqrt', n_estimators=300, n_jobs=12)
    elif model_name == "AdaBoostRegressor":
        model = AdaBoostRegressor()
        # model = AdaBoostRegressor(loss='square', n_estimators=600)
    elif model_name == "GradientBoostingRegressor":
        odel = GradientBoostingRegressor()
        # model = GradientBoostingRegressor(n_estimators=400, ccp_alpha=0.001)
    elif model_name == "BaggingRegressor":
        model = BaggingRegressor()
        # model = BaggingRegressor(max_features=2, n_estimators=1000)
    elif model_name == "XGBRegressor":
        model = XGBRegressor()
        # model = XGBRegressor(reg_alpha=0.5, reg_lambda=1.25, learning_rate=0.02, 
        #     max_depth=3, n_estimators=600, subsample=0.5, n_jobs=16)

    return model
