from os import sep
import numpy as np
import Uncertainty as unc
import UncertaintyM as uncM
import Data.data_provider as dp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve


def expected_calibration_error(probs, predictions, y_true, bins=10):
    prob_max = np.max(probs, axis=1) # find the most probabil class
    correctness_map = np.where(predictions==y_true, 1, 0) # determine which predictions are correct

    bin_size = 1/bins
    ece = 0
    for bin in range(bins):
        bin_indexes = np.where((prob_max > bin * bin_size) & (prob_max <= (bin+1) * bin_size))[0]
        if len(bin_indexes) > 0:
            bin_conf = prob_max[bin_indexes].mean()
            bin_acc = correctness_map[bin_indexes].sum() / len(bin_indexes)
            dif = abs(bin_conf - bin_acc)
            ece += dif
    return ece 

# load data
dataset = "cifar10small"  
features, target = dp.load_data(dataset) # cifar10small

tu_auroc = []
eu_auroc = []
au_auroc = []
tumc_auroc = []
eumc_auroc = []
aumc_auroc = []
tuc_auroc = []

tu_ood_auroc = []
eu_ood_auroc = []
au_ood_auroc = []
tumc_ood_auroc = []
eumc_ood_auroc = []
aumc_ood_auroc = []
tuc_ood_auroc = []

for seed in range(10):
    print(seed)
    # seperate id from ood
    features, target = shuffle(features, target, random_state=seed)
    classes = np.unique(target)
    selected_id = np.random.choice(classes,int(len(classes)/2),replace=False) # select id classes
    id_index = np.argwhere(np.isin(target,selected_id)) # get index of all id instances
    ood_index = np.argwhere(np.isin(target,selected_id,invert=True)) # get index of all not selected classes (OOD)

    cv_outer = KFold(n_splits=5, shuffle=False)
    for train_ix, test_ix in cv_outer.split(features):
        # geting the train and test data with the Kfold indexes intersected with id data
        x_train, x_test_all = features[np.intersect1d(train_ix, id_index)], features[np.intersect1d(test_ix, id_index)]
        y_train, y_test_all = target[np.intersect1d(train_ix, id_index)], target[np.intersect1d(test_ix, id_index)]
        # spliting all test data to test and calibration
        test_len = len(x_test_all)
        x_test  = x_test_all[:int(test_len/2)]
        y_test  = y_test_all[:int(test_len/2)]
        x_calib = x_test_all[int(test_len/2):]
        y_calib = y_test_all[int(test_len/2):]
        
        # create ood test dataset with mix of id and ood
        x_ood,   y_ood  = features[np.intersect1d(test_ix, ood_index)],target[np.intersect1d(test_ix, ood_index)]
        minlen = len(x_test)
        if len(x_test) < minlen:
            minlen = len(x_ood)
        y_test_idoodmix = np.concatenate((np.zeros(minlen), np.ones(minlen)), axis=0)
        x_test_idoodmix = np.concatenate((x_test[:minlen], x_ood[:minlen]), axis=0)

        # train normal model
        model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
        model.fit(x_train, y_train)
        predictions_x_test = model.predict(x_test)
        # print("Ens model test score = ", model.score(x_test, y_test))
        
        # train calibrated model
        calib_method = "isotonic" # 
        model_calib = CalibratedClassifierCV(model, cv=5, method=calib_method)
        model_calib.fit(x_train, y_train)

        prob_x_calib = model.predict_proba(x_calib)
        lr = LogisticRegression(C=99999999999)
        lr.fit(prob_x_calib, y_calib)

        # iso = IsotonicRegression()
        # iso.fit(prob_x_calib, y_calib)

        prob_x_test = model.predict_proba(x_test)
        prob_x_test_calib = lr.predict_proba(prob_x_test)

        model_ece = expected_calibration_error(prob_x_test, model.predict(x_test), y_test, bins=10)
        # print("------------------------------------")
        modelcalib_ece = expected_calibration_error(prob_x_test_calib, lr.predict(prob_x_test), y_test, bins=10)
        
        # _,_, model_ece = calibration_curve(y_test, prob_x_test[:,0], n_bins=10)
        # _,_, modelcalib_ece = calibration_curve(y_test, prob_x_test_calib[:,0], n_bins=10)
        
        
        print(f"model_ece {model_ece}")
        print(f"modelcalib_ece {modelcalib_ece}")
        exit()

        prob_x_test_idoodmix = model.predict_proba(x_test_idoodmix)
        prob_x_test_calib_idoodmix = lr.predict_proba(prob_x_test_idoodmix)

        # print(prob_x_test[1])
        # print(prob_x_test_calib[1])

        # print(f"prob_x_test {prob_x_test.shape} prob_x_test_calib {prob_x_test_calib.shape}")
        # exit()

        # unc Q id
        tu, eu, au = unc.model_uncertainty(model, x_test, x_train, y_train)
        tumc, eumc, aumc = unc.calib_ens_member_uncertainty(model, x_test, x_train, y_train, x_calib, y_calib, calib_method=calib_method)
        tuc = unc.calib_ens_total_uncertainty(prob_x_test_calib)
        # unc Q id OOD 
        tu_ood, eu_ood, au_ood = unc.model_uncertainty(model, x_test_idoodmix, x_train, y_train)
        tumc_ood, eumc_ood, aumc_ood = unc.calib_ens_member_uncertainty(model, x_test_idoodmix, x_train, y_train, x_calib, y_calib)
        tuc_ood = unc.calib_ens_total_uncertainty(prob_x_test_calib_idoodmix)

        # acc-rej
        # ens
        tu_auroc.append(uncM.unc_auroc(predictions_x_test, y_test, tu))
        eu_auroc.append(uncM.unc_auroc(predictions_x_test, y_test, eu))
        au_auroc.append(uncM.unc_auroc(predictions_x_test, y_test, au))
        # ens member calib
        tumc_auroc.append(uncM.unc_auroc(predictions_x_test, y_test, tumc))
        eumc_auroc.append(uncM.unc_auroc(predictions_x_test, y_test, eumc))
        aumc_auroc.append(uncM.unc_auroc(predictions_x_test, y_test, aumc))
        # ens calib
        tuc_auroc.append(uncM.unc_auroc(predictions_x_test, y_test, tuc))

        # OOD test
        # ens
        tu_ood_auroc.append(roc_auc_score(y_test_idoodmix, tu_ood))
        eu_ood_auroc.append(roc_auc_score(y_test_idoodmix, eu_ood))
        au_ood_auroc.append(roc_auc_score(y_test_idoodmix, au_ood))
        # ens member calib
        tumc_ood_auroc.append(roc_auc_score(y_test_idoodmix, tumc_ood))
        eumc_ood_auroc.append(roc_auc_score(y_test_idoodmix, eumc_ood))
        aumc_ood_auroc.append(roc_auc_score(y_test_idoodmix, aumc_ood))
        # ens calib
        tuc_ood_auroc.append(roc_auc_score(y_test_idoodmix, tuc_ood))


tu_auroc_avg = np.array(tu_auroc).mean()
eu_auroc_avg = np.array(eu_auroc).mean()
au_auroc_avg = np.array(au_auroc).mean()
tumc_auroc_avg = np.array(tumc_auroc).mean()
eumc_auroc_avg = np.array(eumc_auroc).mean()
aumc_auroc_avg = np.array(aumc_auroc).mean()
tuc_auroc_avg = np.array(tuc_auroc).mean()

print(f"dataset {dataset}")
print("------------------------------------acc-rej")
print("Ens")
print(f"{tu_auroc_avg* 100:.2f} Total uncertainty") 
print(f"{eu_auroc_avg* 100:.2f} Epist uncertainty") 
print(f"{au_auroc_avg* 100:.2f} Aleat uncertainty") 
print("EnsMemberCalib")
print(f"{tumc_auroc_avg* 100:.2f} Total uncertainty") 
print(f"{eumc_auroc_avg* 100:.2f} Epist uncertainty") 
print(f"{aumc_auroc_avg* 100:.2f} Aleat uncertainty") 
print("EnsCalib")
print(f"{tuc_auroc_avg* 100:.2f} Total uncertainty") 

tu_ood_auroc_avg = np.array(tu_ood_auroc).mean()
eu_ood_auroc_avg = np.array(eu_ood_auroc).mean()
au_ood_auroc_avg = np.array(au_ood_auroc).mean()
tumc_ood_auroc_avg = np.array(tumc_ood_auroc).mean()
eumc_ood_auroc_avg = np.array(eumc_ood_auroc).mean()
aumc_ood_auroc_avg = np.array(aumc_ood_auroc).mean()
tuc_ood_auroc_avg = np.array(tuc_ood_auroc).mean()

print("------------------------------------OOD test")
print("Ens")
print(f"{tu_ood_auroc_avg* 100:.2f} Total uncertainty") 
print(f"{eu_ood_auroc_avg* 100:.2f} Epist uncertainty") 
print(f"{au_ood_auroc_avg* 100:.2f} Aleat uncertainty") 
print("EnsMemberCalib")
print(f"{tumc_ood_auroc_avg* 100:.2f} Total uncertainty") 
print(f"{eumc_ood_auroc_avg* 100:.2f} Epist uncertainty") 
print(f"{aumc_ood_auroc_avg* 100:.2f} Aleat uncertainty") 
print("EnsCalib")
print(f"{tuc_ood_auroc_avg* 100:.2f} Total uncertainty") 
