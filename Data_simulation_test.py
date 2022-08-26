import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.calibration import CalibratedClassifierCV
from sklearn import tree
import CalibrationM as calibm
import warnings
import Uncertainty as unc
import UncertaintyM as uncM
import pandas as pd

warnings.filterwarnings("ignore")

seed = 1
np.random.seed(seed)
calibration_method = "isotonic"
laplace = 1

# optimal prob
from math import nan

def opt_prob(threshold, x):
    p_a_0 = 1
    p_b_0 = 1/2
    p_c_0 = 0

    p_a_1 = 0
    p_b_1 = 1/2
    p_c_1 = 1

    # checking the un defined boundaries
    if threshold < 0: 
        return nan
    if threshold > 3:
        return nan

    # calculating the length of each section
    if threshold <= 1:
        a = threshold
        b = 0
        c = 0    
    if 1 < threshold <= 2:
        a = 1
        b = threshold - 1
        c = 0    
    if 2 < threshold <= 3:
        a = 1
        b = 1
        c = threshold - 2

    if x >= threshold:
        a = 1 - a
        b = 1 - b
        c = 1 - c

        threshold = 3 - threshold

    prob_1 = (a * p_a_1 + b * p_b_1 + c * p_c_1) / threshold # Probability of class one on the left side of the threshold ( here threshold is threshold)
    prob_0 = (a * p_a_0 + b * p_b_0 + c * p_c_0) / threshold # Probability of class zero on the left side of the threshold

    return prob_0, prob_1


for data_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:

    x = np.random.uniform(low=0, high=3, size=(data_size,))
    y = np.where(x<1, 0, np.where(x>2, 1, 1/2))
    mid_index = np.argwhere(y==1/2)
    mid_index = mid_index.reshape(-1)
    rand_y = np.random.binomial(size=len(mid_index), n=1, p= 0.5)
    y[mid_index] = rand_y
    x = x.reshape(-1,1)

    x_train, x_test_all, y_train, y_test_all = train_test_split(x, y, test_size=0.4, shuffle=True, random_state=seed)
    x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=seed) 

    model = RandomForestClassifier(max_depth=100, n_estimators=10, random_state=seed)
    model.fit(x_train, y_train)
    predictions_x_test = model.predict(x_test)
    acc = model.score(x_test, y_test)
    prob_x_test = model.predict_proba(x_test)
    prob_x_calib = model.predict_proba(x_calib)

    thresholds = []

    for est_index, estimator in enumerate(model.estimators_):
        tree_threshold = estimator.tree_.threshold[0]
        tree_prob_x_test = estimator.predict_proba(x_test)

        # Full Dirichlet
        if calibration_method == "isotonic":
            model_calib = CalibratedClassifierCV(estimator, cv="prefit", method=calibration_method) # cv=30
            model_calib.fit(x_calib , y_calib)
            tree_prob_x_test_calib = model_calib.predict_proba(x_test)


        if calibration_method == "Dir":
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
            dir_calib = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg, 'reg_mu': [None]}, cv=skf, scoring='neg_log_loss')
            dir_calib.fit(prob_x_calib , y_calib)
            tree_prob_x_test_calib = dir_calib.predict_proba(tree_prob_x_test)
            

        # optimal prob
        thresholds.append(tree_threshold)

    if calibration_method == "isotonic":
        model_calib = CalibratedClassifierCV(model, cv="prefit", method=calibration_method) # cv=30
        model_calib.fit(x_calib , y_calib)
        prob_x_test_calib = model_calib.predict_proba(x_test)

    # Full Dirichlet
    if calibration_method == "Dir":
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
        dir_calib = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg, 'reg_mu': [None]}, cv=skf, scoring='neg_log_loss')
        dir_calib.fit(prob_x_calib , y_calib)
        prob_x_test_calib = dir_calib.predict_proba(prob_x_test)

    # unc Q id
    tu, eu, au = unc.model_uncertainty(model, x_test, x_train, y_train, laplace_smoothing=laplace)
    tumc, eumc, aumc, porb_matrix = unc.calib_ens_member_uncertainty(model, x_test, y_test, x_train, y_train, x_calib, y_calib, calibration_method, seed)
    tuc = unc.calib_ens_total_uncertainty(prob_x_test_calib)

    # true prob calculation for x_test
    prob_x_test_true = []

    for x in x_test:
        for threshold in thresholds:
            p_0, p_1 = opt_prob(threshold, x)
            prob_x_test_true.append([p_0, p_1])

    prob_x_test_true = np.array(prob_x_test_true).reshape(len(x_test), len(thresholds), 2) # D1 data D2 ens member D3 class prob

    # uncertainty for true prob
    tuo, euo, auo = uncM.uncertainty_ent_bays(prob_x_test_true, np.full(len(thresholds), 1/len(thresholds)))

    opt_p = prob_x_test_true.mean(axis=1)

    data = np.concatenate((x_test, prob_x_test[:,0].reshape(-1,1), prob_x_test_calib[:,0].reshape(-1,1), opt_p[:,0].reshape(-1,1), au.reshape(-1,1), aumc.reshape(-1,1), auo.reshape(-1,1)), axis=1)
    data_df = pd.DataFrame(data, columns=["x", "n_prob", "c_prob", "o_prob", "a_unc", "a_unc_c", "a_unc_o"])
    print(data_df.head())


    print("------------------------------------")
    print("Data size: ", data_size)
    print("acc: ", acc)
    print("uncalibrated AU ", uncM.unc_auroc(predictions_x_test, y_test, au))
    print("calibrated   AU ", uncM.unc_auroc(predictions_x_test, y_test, aumc))
    print("optimal      AU ", uncM.unc_auroc(predictions_x_test, y_test, auo))
