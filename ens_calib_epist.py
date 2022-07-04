import os
import numpy as np
import Uncertainty as unc
import Data.data_provider as dp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
import ray
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

dataset_list = ['CIFAR10'] # 'fashionMnist', 'CIFAR10',
run_name = "Results/uncCalib epist aleSort"

@ray.remote
def calib_ale_test(features, target, seed):

    features, target = shuffle(features, target, random_state=seed)
    # seperate to train test calibration
    classes = np.unique(target)
    selected_id = np.random.choice(classes,int(len(classes)/2),replace=False) # select id classes
    id_index = np.reshape(np.argwhere(np.isin(target,selected_id)), -1) # get index of all id instances
    ood_index = np.reshape(np.argwhere(np.isin(target,selected_id,invert=True)), -1) # get index of all not selected classes (OOD)

    x_train, x_test_all, y_train, y_test_all = train_test_split(features[id_index], target[id_index], test_size=0.4, shuffle=True, random_state=seed)
    x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=seed) 
    x_ood,   _  = features[ood_index], target[ood_index]
    
    # train normal model
    model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed) # 
    model.fit(x_train, y_train)
    
    # train calibrated model
    calib_method = "isotonic" # "sigmoid" # 
    model_calib = CalibratedClassifierCV(model, cv=30, method=calib_method)
    model_calib.fit(x_calib , y_calib)

    # aleatoric uncertainty on OOD data to sort based on it
    _, _, au = unc.model_uncertainty(model, x_ood, x_train, y_train)
    sorted_index = np.argsort(au, kind='stable')
    x_ood = x_ood[sorted_index]

    minlen = len(x_test)
    if len(x_ood) < minlen:
        minlen = len(x_ood)
    y_test_idoodmix = np.concatenate((np.zeros(minlen), np.ones(minlen)), axis=0)
    x_test_idoodmix = np.concatenate((x_test[:minlen], x_ood[:minlen]), axis=0)

    prob_x_test_calib_idoodmix = model_calib.predict_proba(x_test_idoodmix)

    # unc Q OOD
    tu_ood, eu_ood, au_ood = unc.model_uncertainty(model, x_test_idoodmix, x_train, y_train)
    tumc_ood, eumc_ood, aumc_ood = unc.calib_ens_member_uncertainty(model, x_test_idoodmix, x_train, y_train, x_calib, y_calib, calib_method)
    tuc_ood = unc.calib_ens_total_uncertainty(prob_x_test_calib_idoodmix)

    # acc-rej
    # ens
    tu_auroc = roc_auc_score(y_test_idoodmix, tu_ood)
    eu_auroc = roc_auc_score(y_test_idoodmix, eu_ood)
    au_auroc = roc_auc_score(y_test_idoodmix, au_ood)
    # ens member calib
    tumc_auroc = roc_auc_score(y_test_idoodmix, tumc_ood)
    eumc_auroc = roc_auc_score(y_test_idoodmix, eumc_ood)
    aumc_auroc = roc_auc_score(y_test_idoodmix, aumc_ood)
    # ens calib
    tuc_auroc = roc_auc_score(y_test_idoodmix, tuc_ood)
    return tu_auroc, eu_auroc, au_auroc, tumc_auroc, eumc_auroc, aumc_auroc, tuc_auroc

ray.init()
for dataset in dataset_list:
    # load data
    features, target = dp.load_data(dataset)

    ray_array = []
    for seed in range(10):
        ray_array.append(calib_ale_test.remote(features, target, seed))

    res_array = np.array(ray.get(ray_array)).mean(axis=0)

    res_txt = f"dataset {dataset}\n"
    res_txt += f"------------------------------------OOD test \n"
    res_txt += "Ens \n"
    res_txt += f"{res_array[0]* 100:.2f} Total uncertainty\n"
    res_txt += f"{res_array[1]* 100:.2f} Epist uncertainty\n"
    res_txt += f"{res_array[2]* 100:.2f} Aleat uncertainty\n"
    res_txt += "MemberCalib \n"
    res_txt += f"{res_array[3]* 100:.2f} Total uncertainty\n"
    res_txt += f"{res_array[4]* 100:.2f} Epist uncertainty\n"
    res_txt += f"{res_array[5]* 100:.2f} Aleat uncertainty\n"
    res_txt += "EnsCalib \n"
    res_txt += f"{res_array[6]* 100:.2f} Total uncertainty\n"

    if not os.path.exists(run_name):
        os.makedirs(run_name)
    with open(f"{run_name}/{dataset}_uncCalib.txt", "w") as text_file:
        text_file.write(res_txt)
    print(f"{dataset} done")
