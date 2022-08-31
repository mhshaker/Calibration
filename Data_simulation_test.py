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

warnings.filterwarnings("ignore")

seed = 1
np.random.seed(seed)
calibration_method = "isotonic"
# Data creation

data_size = 200
n_estimators = 10
max_depth = 1
laplace = 1

# for data_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
# for n_estimators in [2, 4, 6, 8, 10, 20, 50, 100]:
for max_depth in [1, 2, 4, 6, 8, 10]:
    x = np.random.uniform(low=0, high=3, size=(data_size,))
    y = np.where(x<1, 0, np.where(x>2, 1, 1/2))
    mid_index = np.argwhere(y==1/2)
    mid_index = mid_index.reshape(-1)
    rand_y = np.random.binomial(size=len(mid_index), n=1, p= 0.5)
    y[mid_index] = rand_y
    x = x.reshape(-1,1)

    x_train, x_test_all, y_train, y_test_all = train_test_split(x, y, test_size=0.4, shuffle=True, random_state=seed)
    x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=seed) 

    figure(figsize=(30, 3), dpi=80)
    # plt.plot(x, np.zeros_like(x), "x", color="y")
    colors = ['red','black']
    plt.scatter(x, np.full(len(x), "All Data"), c=y, cmap=matplotlib.colors.ListedColormap(colors))
    x1, y1 = [1, 1], [1, -1]
    x2, y2 = [2, 2], [1, -1]
    plt.plot(x1, y1, x2, y2, c="gray")
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
        # print("side ", x)
        # print("a ", a)
        # print("b ", b)
        # print("c ", c)
        prob_1 = (a * p_a_1 + b * p_b_1 + c * p_c_1) / threshold # Probability of class one on the left side of the threshold ( here threshold is threshold)
        prob_0 = (a * p_a_0 + b * p_b_0 + c * p_c_0) / threshold # Probability of class zero on the left side of the threshold

        return prob_0, prob_1
    opt_prob(2, 1.8) # It does not mater if x moves in a section, the prob will only change if x changes section. Prob will always change if the threshold changes.
    # Model: Decision stump ensembel
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=seed)
    model.fit(x_train, y_train)
    predictions_x_test = model.predict(x_test)
    prob_x_test = model.predict_proba(x_test)
    prob_x_calib = model.predict_proba(x_calib)
    # Member calibration
    thresholds = []
    figure(figsize=(30, 3), dpi=80)

    for est_index, estimator in enumerate(model.estimators_):
        # tree.plot_tree(estimator)
        tree_threshold = estimator.tree_.threshold[0]
        tree_prob_x_test = estimator.predict_proba(x_test)

        # print(tree_prob_x_test)
        
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
        o_p_pos_r, _ = opt_prob(tree_threshold, tree_threshold + 0.1)
        o_p_pos_l, _ = opt_prob(tree_threshold, tree_threshold - 0.1)

    # Ens Calibration

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


    # Uncertainty
    # unc Q id
    tu, eu, au = unc.model_uncertainty(model, x_test, x_train, y_train, laplace_smoothing=laplace)
    tumc, eumc, aumc, porb_matrix = unc.calib_ens_member_uncertainty(model, x_test, y_test, x_train, y_train, x_calib, y_calib, calibration_method, seed, laplace_smoothing=laplace)
    tuc = unc.calib_ens_total_uncertainty(prob_x_test_calib)
    # true tree prob for each x_test and true aleatoric uncertianty
    prob_x_test_true = []

    for x in x_test:
        for threshold in thresholds:
            p_0, p_1 = opt_prob(threshold, x)
            prob_x_test_true.append([p_0, p_1])

    prob_x_test_true = np.array(prob_x_test_true).reshape(len(x_test), len(thresholds), 2) # D1 data D2 ens member D3 class prob

    # uncertainty for true prob
    tuo, euo, auo = uncM.uncertainty_ent_bays(prob_x_test_true, np.full(len(thresholds), 1/len(thresholds)))
    # true ens prob for each x_test and uncertainty
    data_size_p_ens = 1000000

    x_p_ens = np.random.uniform(low=0, high=3, size=(data_size_p_ens,))
    y_p_ens = np.where(x_p_ens<1, 0, np.where(x_p_ens>2, 1, 1/2))
    mid_index_p_ens = np.argwhere(y_p_ens==1/2)
    mid_index_p_ens = mid_index_p_ens.reshape(-1)
    rand_y_p_ens = np.random.binomial(size=len(mid_index_p_ens), n=1, p= 0.5)
    y_p_ens[mid_index_p_ens] = rand_y_p_ens
    x_p_ens = x_p_ens.reshape(-1,1)

    x_p_ens_pred = model.predict(x_p_ens)
    u, count = np.unique(x_p_ens_pred, return_counts=True)

    ens_pred_0 = np.where(x_p_ens_pred==0)
    y_p_ens_0 = y_p_ens[ens_pred_0]
    ens_pred_0_corr = np.where(y_p_ens_0 == x_p_ens_pred[ens_pred_0])

    ens_pred_1 = np.where(x_p_ens_pred==1)
    y_p_ens_1 = y_p_ens[ens_pred_1]
    ens_pred_1_corr = np.where(y_p_ens_1 == x_p_ens_pred[ens_pred_1])


    p_0_ens = len(ens_pred_0_corr[0]) / count[0]
    p_1_ens = len(ens_pred_1_corr[0]) / count[1]

    ens_prob = []
    for x_pred in predictions_x_test:
        if x_pred == 0:
            ens_prob.append([p_0_ens, 1 - p_0_ens])
        elif x_pred == 1:
            ens_prob.append([1 - p_1_ens, p_1_ens])
    prob_x_test_ensopt = np.array(ens_prob)

    tueo = unc.calib_ens_total_uncertainty(prob_x_test_ensopt)


    print(f"------------------------------------ data_size {data_size} n_estimators {n_estimators} max_depth {max_depth}")
    print("Area under: ")
    print("uncalibrated AU ", uncM.unc_auroc(predictions_x_test, y_test, au))
    print("calibrated   AU ", uncM.unc_auroc(predictions_x_test, y_test, aumc))
    print("mem optimal  AU ", uncM.unc_auroc(predictions_x_test, y_test, auo))
    print("ens calib    TU ", uncM.unc_auroc(predictions_x_test, y_test, tuc))
    print("ens optimal  TU ", uncM.unc_auroc(predictions_x_test, y_test, tueo))

# # true uncertainty plot value
# figure(figsize=(30, 3), dpi=80)

# colors = ['red','black']
# plt.scatter(x_test, np.full(len(x_test), "Ens opt"), c=tueo, cmap="gist_yarg", edgecolors='black')
# plt.scatter(x_test, np.full(len(x_test), "Ens calib"), c=tuc, cmap="gist_yarg", edgecolors='black')
# plt.scatter(x_test, np.full(len(x_test), "Ale opt"), c=auo, cmap="gist_yarg", edgecolors='black')
# plt.scatter(x_test, np.full(len(x_test), "Ale calib"), c=aumc, cmap="gist_yarg", edgecolors='black')
# plt.scatter(x_test, np.full(len(x_test), "Ale normal"), c=au, cmap="gist_yarg", edgecolors='black')

# for est_index, threshold in enumerate(thresholds):
#     plt.plot([threshold,threshold], [5,-1], c="blue")
#     plt.annotate(est_index, (threshold+0.01, -1))

# x1, y1 = [1, 1], [5, -1]
# x2, y2 = [2, 2], [5, -1]
# plt.plot(x1, y1, x2, y2, c="gray")
