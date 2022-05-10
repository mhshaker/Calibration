import numpy as np
import UncertaintyM as unc
from sklearn.metrics import log_loss
from scipy.stats import entropy
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

def model_uncertainty(model, x_test, x_train, y_train, unc_method="bays", laplace_smoothing=1, log=False):
    
    if "bays" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, laplace_smoothing)
        porb_matrix = get_prob(model, x_test, laplace_smoothing)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for RF")

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty

def calib_ens_total_uncertainty(model, x_test):
    
    probs = model.predict_proba(x_test)
    total_uncertainty = entropy(probs, base=2,axis=1)
    return total_uncertainty

def calib_ens_member_uncertainty(model, x_test, x_train, y_train, X_calib, y_calib, unc_method="bays", laplace_smoothing=0, log=False):
    
    if "bays" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, laplace_smoothing)
        porb_matrix = get_member_calib_prob(model, x_test, X_calib, y_calib)
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

def get_member_calib_prob(model_ens, x_data, X_calib, y_calib):
    prob_matrix  = []
    for estimator in model_ens.estimators_:
        model_calib = CalibratedClassifierCV(estimator, cv=20, method="isotonic")
        model_calib.fit(X_calib, y_calib)
        tree_prob = model_calib.predict_proba(x_data)
        prob_matrix.append(tree_prob)
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