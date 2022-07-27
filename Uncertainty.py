import numpy as np
import UncertaintyM as unc
from sklearn.metrics import log_loss
from scipy.stats import entropy
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.isotonic import IsotonicRegression

def model_uncertainty(model, x_test, x_train, y_train, unc_method="bays", laplace_smoothing=1, log=False):
    
    if "bays" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, laplace_smoothing)
        porb_matrix = get_prob(model, x_test, laplace_smoothing)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for RF")

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty

def calib_ens_total_uncertainty(probs):
    total_uncertainty = entropy(probs, base=2,axis=1)
    return total_uncertainty

def calib_ens_member_uncertainty(model, x_test, x_train, y_train, X_calib, y_calib, calib_method="isotonic", unc_method="bays", laplace_smoothing=1, log=False):
    
    if "bays" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, laplace_smoothing)
        porb_matrix = get_member_calib_prob(model, x_test, X_calib, y_calib, calib_method=calib_method)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for RF")

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty


############################################################################################

def get_likelyhood(model_ens, x_train, y_train, laplace_smoothing, a=0, b=0, log=False):
    likelyhoods  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob_train = estimator.predict_proba(x_train) 
        else:
            tree_prob_train = tree_laplace_corr(estimator,x_train, laplace_smoothing, a, b)

        likelyhoods.append(log_loss(y_train,tree_prob_train))
    likelyhoods = np.array(likelyhoods)
    likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
    likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

    if log:
        print(f"<log>----------------------------------------[]")
        print(f"likelyhoods = {likelyhoods}")
    return np.array(likelyhoods)

def get_prob(model_ens, x_data, laplace_smoothing, a=0, b=0, log=False):
    prob_matrix  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob = estimator.predict_proba(x_data) 
        else:
            tree_prob = tree_laplace_corr(estimator,x_data, laplace_smoothing,a,b)
        prob_matrix.append(tree_prob)
    if log:
        print(f"<log>----------------------------------------[]")
        print(f"prob_matrix = {prob_matrix}")
    prob_matrix = np.array(prob_matrix)
    prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
    return prob_matrix

def get_member_calib_prob(model_ens, x_data, X_calib, y_calib, calib_method="isotonic"):
    prob_matrix  = []

    for estimator in model_ens.estimators_:
        if calib_method == "isotonic" or calib_method == "sigmoid":
            model_calib = CalibratedClassifierCV(estimator, cv="prefit", method=calib_method) # cv=30
            model_calib.fit(X_calib, y_calib)
            tree_prob_x_test_calib = model_calib.predict_proba(x_data)
        elif calib_method =="iso":
            iso = IsotonicRegression()
            iso.fit(estimator.predict_proba(X_calib)[:,0], y_calib)
            tree_prob_x_test_calib = iso.predict(estimator.predict_proba(x_data)[:,0])
            tree_prob_x_test_calib = np.nan_to_num(tree_prob_x_test_calib) # remove NAN values
            second_class_prob = np.ones(len(tree_prob_x_test_calib)) - tree_prob_x_test_calib
            tree_prob_x_test_calib = np.concatenate((tree_prob_x_test_calib.reshape(-1,1), second_class_prob.reshape(-1,1)), axis=1)

        elif calib_method == "Dir":
            # calib_model = FullDirichletCalibrator(reg_lambda=1e-1, reg_mu=None)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            # Full Dirichlet
            calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
            model_calib = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg, 'reg_mu': [None]}, cv=skf, scoring='neg_log_loss')
            model_calib.fit(estimator.predict_proba(X_calib) , y_calib)
            tree_prob_x_test_calib = model_calib.predict_proba(estimator.predict_proba(x_data))

        prob_matrix.append(tree_prob_x_test_calib)
    prob_matrix = np.array(prob_matrix)
    prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
    return prob_matrix

def tree_laplace_corr(tree, x_data, laplace_smoothing, a=0, b=0):
    tree_prob = tree.predict_proba(x_data)
    leaf_index_array = tree.apply(x_data)
    for data_index, leaf_index in enumerate(leaf_index_array):
        leaf_values = tree.tree_.value[leaf_index]
        leaf_samples = np.array(leaf_values).sum()
        for i,v in enumerate(leaf_values[0]):
            L = laplace_smoothing
            if a != 0 or b != 0:
                if i==0:
                    L = a
                else:
                    L = b
            # print(f"i {i} v {v} a {a} b {b} L {L} prob {(v + L) / (leaf_samples + (len(leaf_values[0]) * L))}")
            tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
    return tree_prob