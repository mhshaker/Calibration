import os 
import numpy as np
import Uncertainty as unc
import UncertaintyM as uncM
import Data.data_provider as dp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import ray
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import CalibrationM as calibm

# from calmap import plot_calibration_map

calibration_method = "Dir" # isotonic "sigmoid"
dataset_list = ['fashionMnist', "CIFAR10", "CIFAR100"] # 'fashionMnist', 'amazon_movie'
run_name = "Results/Ale RF uncFix"
log_ECE = False
log_AUARC = False
log_prob_dist = False
plot_calib = False
all_methods_comp = False
laplace = 0

@ray.remote
def calib_ale_test(features, target, seed):
    # seperate to train test calibration
    x_train, x_test_all, y_train, y_test_all = train_test_split(features, target, test_size=0.4, shuffle=True, random_state=seed)
    x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=seed) 
    
    # train normal model
    model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
    model.fit(x_train, y_train)
    predictions_x_test = model.predict(x_test)
    prob_x_test = model.predict_proba(x_test)
    if log_prob_dist:
        plt.hist(prob_x_test)
        plt.savefig("Step_results/ens_normal_dist_l1.png")
    prob_x_calib = model.predict_proba(x_calib)
    
    normal_cw_ece = calibm.classwise_ECE(prob_x_test, y_test)
    normal_conf_ece = calibm.confidance_ECE(prob_x_test, y_test)

    if log_ECE:
        print(f"seed {seed} normal cw_ece = ", normal_cw_ece)
        print(f"seed {seed} normal cw_ece sum ", sum(normal_cw_ece))

    # train calibrated model
    if calibration_method == "isotonic" or calibration_method == "sigmoid" or all_methods_comp:
        model_calib = CalibratedClassifierCV(model, cv="prefit", method="isotonic") # cv=30
        model_calib.fit(x_calib , y_calib)
        prob_x_test_calib = model_calib.predict_proba(x_test)
        if plot_calib:
            # sk_prob_x_calib_calibrated = model_calib.predict_proba(x_calib)[:,1]
            pr = prob_x_test_calib
            prob_x_calib_sk = prob_x_calib[:,1]
            prob_x_calib_sk = prob_x_calib_sk[0:-1]

            idx = prob_x_calib_sk.argsort()
            scores = prob_x_calib_sk[idx]
            pr = pr[idx]
            plt.plot(pr, scores, label='sk_iso')
            plt.plot([0,1], [0,1], ':', c="black")
            plt.title(f"Isotonic regression fit on x_calib - seed {seed}")
            plt.xlabel('score')
            plt.ylabel('prediction')
            plt.legend()
            plt.savefig(f"Step_results/iso_plot_test_run{seed}_sk.png")
            plt.close()

            sk_iso_cw_ece = calibm.classwise_ECE(prob_x_test_calib, y_test)
            sk_iso_conf_ece = calibm.confidance_ECE(prob_x_test_calib, y_test)
            if log_ECE:
                print(f"seed {seed} sk_iso cw_ece = ", sk_iso_cw_ece)
                print(f"seed {seed} sk_iso cw_ece sum ", sum(sk_iso_cw_ece))
            
        
    if (calibration_method =="iso" or all_methods_comp) and len(np.unique(y_train)) <= 2:
        iso_calib = IsotonicRegression()
        iso_x_calib = prob_x_calib[:,1]

        iso_calib.fit(iso_x_calib, y_calib)

        prob_x_test_calib = iso_calib.predict(prob_x_test[:,1])
        prob_x_test_calib = np.nan_to_num(prob_x_test_calib) # remove NAN values
        second_class_prob = np.ones(len(prob_x_test_calib)) - prob_x_test_calib
        prob_x_test_calib = np.concatenate((prob_x_test_calib.reshape(-1,1), second_class_prob.reshape(-1,1)), axis=1)

        if plot_calib:
            linspace = np.linspace(0, 1, len(x_calib))
            pr = iso_calib.predict(linspace)
            idx = iso_x_calib.argsort() 
            scores = iso_x_calib[idx]
            plt.plot(pr, scores, label='iso')
            # plt.plot([0,1], [0,1], ':', c="black")
            # plt.title(f"Isotonic regression fit on x_calib - seed {seed}")
            # plt.xlabel('score')
            # plt.ylabel('prediction')
            # plt.legend()
            # plt.savefig(f"Step_results/iso_plot_test{seed}.png")
            # plt.close()
            if log_ECE:
                print("iso cw_ece = ", calibm.classwise_ECE(prob_x_test_calib, y_test))



    if calibration_method == "Dir" or all_methods_comp:
        # calib_model = FullDirichletCalibrator(reg_lambda=1e-1, reg_mu=None)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        # Full Dirichlet
        calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
        dir_calib = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg, 'reg_mu': [None]}, cv=skf, scoring='neg_log_loss')
        dir_calib.fit(prob_x_calib , y_calib)
        prob_x_test_calib = dir_calib.predict_proba(prob_x_test)
        if plot_calib:
            if len(np.unique(y_train)) <= 2:
                linspace = np.linspace(0, 1, len(x_calib))
                second_class_prob_l = np.ones(len(x_calib)) - linspace
                linspace = np.concatenate((second_class_prob_l.reshape(-1,1), linspace.reshape(-1,1)), axis=1)

                pr = dir_calib.predict_proba(linspace)
                dir_x_calib = prob_x_calib[:,1]
                idx = dir_x_calib.argsort()
                scores = dir_x_calib[idx]
                plt.plot(pr[:,1], scores, label='Dirichlet')
                plt.plot([0,1], [0,1], ':', c="black")
                plt.title(f"Isotonic regression fit on x_calib - seed {seed}")
                plt.xlabel('score')
                plt.ylabel('prediction')
                plt.legend()
                plt.savefig(f"Step_results/iso_plot_test_run{seed}.png")
                plt.close()

            dir_cw_ece = calibm.classwise_ECE(prob_x_test_calib, y_test)
            dir_conf_ece = calibm.confidance_ECE(prob_x_test_calib, y_test)
            if log_ECE:
                print(f"seed {seed} dir cw_ece = ", dir_cw_ece)
                print(f"seed {seed} dir cw_ece sum ", sum(dir_cw_ece))

    if log_ECE:
        print("ens_loss ", log_loss(y_test, prob_x_test))
        print("enscalib_loss ", log_loss(y_test, prob_x_test_calib))

    if log_prob_dist:
        plt.hist(prob_x_test_calib)
        plt.savefig("Step_results/ens_calib_dist_l1.png")
        exit()

    # unc Q id
    tu, eu, au = unc.model_uncertainty(model, x_test, x_train, y_train, laplace_smoothing=laplace) # have to turn off laplace_smoothing because member calibrated uncertainty does not use laplace_smoothing
    tumc, eumc, aumc, porb_matrix = unc.calib_ens_member_uncertainty(model, x_test, y_test, x_train, y_train, x_calib, y_calib, calibration_method, seed, laplace_smoothing=laplace)
    tuc = unc.calib_ens_total_uncertainty(prob_x_test_calib)
    ctp_cs_ece = calibm.classwise_ECE(np.mean(porb_matrix, axis=1), y_test)
    ctp_conf_ece = calibm.confidance_ECE(np.mean(porb_matrix, axis=1), y_test)


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

    if log_AUARC:
        print(tu_auroc, eu_auroc, au_auroc, tumc_auroc, eumc_auroc, aumc_auroc, tuc_auroc)
    # exit()
    if log_ECE == False:
        normal_cw_ece, sk_iso_cw_ece, dir_cw_ece, normal_conf_ece, sk_iso_conf_ece, dir_conf_ece, ctp_cs_ece, ctp_conf_ece = 0,0,0,0,0,0,0,0

    return tu_auroc, eu_auroc, au_auroc, tumc_auroc, eumc_auroc, aumc_auroc, tuc_auroc, normal_cw_ece, sk_iso_cw_ece, dir_cw_ece, normal_conf_ece, sk_iso_conf_ece, dir_conf_ece, ctp_cs_ece, ctp_conf_ece

ray.init()
for dataset in dataset_list:
    # load data
    features, target = dp.load_data(dataset)

    ray_array = []
    for seed in range(10):
        ray_array.append(calib_ale_test.remote(features, target, seed))
        # ray_array.append(calib_ale_test(features, target, seed))

    res_array_all = np.array(ray.get(ray_array))
    with open('Step_results/res_array_all.npy', 'wb') as f:
        np.save(f, res_array_all)
    print("------------------------------------")
    print("------------------------------------")


    res_array = res_array_all[:,0:-8].mean(axis=0)
    print(res_array)

    res_txt = f"dataset {dataset}  - calib: {calibration_method}\n"
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

    with open(f"{run_name}/{dataset}_calib_{calibration_method}.txt", "w") as text_file:
        text_file.write(res_txt)
    print(f"{dataset} done")


    # normal_cw_ece = np.array(res_array_all[:,-3])
    # sk_iso_cw_ece = np.array(res_array_all[:,-2])
    # dir_cw_ece    = np.array(res_array_all[:,-1])

    # with open('Step_results/ens_normal_cw_ece.npy', 'wb') as f:
    #     np.save(f, normal_cw_ece)
    # with open('Step_results/ens_sk_iso_cw_ece.npy', 'wb') as f:
    #     np.save(f, sk_iso_cw_ece)
    # with open('Step_results/ens_dir_cw_ece.npy', 'wb') as f:
    #     np.save(f, dir_cw_ece)

    # print("------------------------------------")

    # print("normal_cw_ece ", normal_cw_ece.mean(axis=0))
    # print("sk_iso_cw_ece ", sk_iso_cw_ece.mean(axis=0))
    # print("dir_cw_ece ", dir_cw_ece.mean(axis=0))
