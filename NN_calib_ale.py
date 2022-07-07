import os 
import numpy as np
import Uncertainty as unc
import UncertaintyM as uncM
import Data.data_provider as dp
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV
import ray

dataset_list = ['fashionMnist', 'CIFAR10', 'CIFAR100'] # 'fashionMnist', 'amazon_movie'
# dataset_list = ['amazon_movie'] 
run_name = "Results/uncCalib Ale NN"


@ray.remote
def calib_ale_test(features, target, seed):
    # seperate to train test calibration
    x_train, x_test_all, y_train, y_test_all = train_test_split(features, target, test_size=0.4, shuffle=True, random_state=seed)
    x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=seed) 
    
    # train normal model
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=seed) 
    model.fit(x_train, y_train)
    predictions_x_test = model.predict(x_test)

    
    # train calibrated model
    calib_method = "isotonic" # "sigmoid" # 
    model_calib = CalibratedClassifierCV(model, cv=30, method=calib_method)
    model_calib.fit(x_calib , y_calib)


    prob_x_test = model.predict_proba(x_test)
    prob_x_test_calib = model_calib.predict_proba(x_test)

    # unc Q id
    tu, eu, au = unc.model_uncertainty(model, x_test, x_train, y_train)
    tumc, eumc, aumc = unc.calib_ens_member_uncertainty(model, x_test, x_train, y_train, x_calib, y_calib, calib_method)
    tuc = unc.calib_ens_total_uncertainty(prob_x_test_calib)

    # acc-rej
    # ens
    tu_auroc = uncM.unc_auroc(predictions_x_test, y_test, tu)
    eu_auroc = uncM.unc_auroc(predictions_x_test, y_test, eu)
    au_auroc = uncM.unc_auroc(predictions_x_test, y_test, au)
    # ens member calib
    tumc_auroc = uncM.unc_auroc(predictions_x_test, y_test, tumc)
    eumc_auroc = uncM.unc_auroc(predictions_x_test, y_test, eumc)
    aumc_auroc = uncM.unc_auroc(predictions_x_test, y_test, aumc)
    # ens calib
    tuc_auroc = uncM.unc_auroc(predictions_x_test, y_test, tuc)

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
    res_txt += f"------------------------------------acc-rej\n"
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
