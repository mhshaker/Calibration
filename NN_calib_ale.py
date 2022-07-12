import os
from matplotlib.pyplot import axes 
import numpy as np
import Uncertainty as unc
import UncertaintyM as uncM
import Data.data_provider as dp
from sklearn.model_selection import train_test_split
import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow import keras
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration

import ray

Train_new = True

# dataset_list = ['fashionMnist', 'CIFAR10', 'CIFAR100'] # 'fashionMnist', 'amazon_movie'
dataset_list = ['fashionMnist'] 
# dataset_list = ['amazon_movie'] 
run_name = "Results/uncCalib Ale NN"

def expected_calibration_error(probs, predictions, y_true, bins=10, equal_bin_size=True):
    prob_max = np.max(probs, axis=1) # find the most probabil class
    correctness_map = np.where(predictions==y_true, 1, 0) # determine which predictions are correct
    ece = 0

    if equal_bin_size == False:
        bin_size = 1/bins
        for bin in range(bins):
            bin_indexes = np.where((prob_max > bin * bin_size) & (prob_max <= (bin+1) * bin_size))[0]
            if len(bin_indexes) > 0:
                bin_conf = prob_max[bin_indexes].mean()
                bin_acc = correctness_map[bin_indexes].sum() / len(bin_indexes)
                dif = abs(bin_conf - bin_acc)
                ece += dif * len(bin_indexes)
        ece = ece / len(predictions)
    else: # equal_bin_size
        sorted_index = np.argsort(prob_max, kind='stable') # sort probs
        prob_max = prob_max[sorted_index]
        correctness_map = correctness_map[sorted_index] 

        bin_size = int(len(predictions) / bins)
        for bin in range(bins):
            bin_conf = prob_max[bin*bin_size:(bin+1)*bin_size].mean()
            bin_acc = correctness_map[bin*bin_size:(bin+1)*bin_size].sum() / bin_size
            dif = abs(bin_conf - bin_acc)
            ece += dif 
        ece = ece / bins

    return ece 


@ray.remote
def calib_ale_test(features, target, model_path, seed):
    
    print("------------------------------------ ", seed)
    model_path_seed = model_path + str(seed)
    # seperate to train test calibration
    x_train, x_test_all, y_train, y_test_all = train_test_split(features, target, test_size=0.4, shuffle=True, random_state=seed)
    x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=seed) 
    print("train data shape >>>>>>> ", x_train.shape)
    # train normal model or load
    if not os.path.exists(model_path_seed) or Train_new==True:
        os.makedirs(model_path_seed)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train.shape[1],)),
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=15, batch_size=8) # keras.utils.to_categorical(y_train)
        model.save(model_path_seed)
    else:
        print("loading model path: ", model_path_seed)
        model = tf.keras.models.load_model(model_path_seed)
    
    print("modle Eval ", model.evaluate(x_test, y_test))

    prob_x_test = model.predict(x_test)
    predictions_x_test = prob_x_test.argmax(axis=1)

    prob_x_calib = model.predict(x_calib)
    predictions_x_calib = prob_x_calib.argmax(axis=1)

    # train calibrated model

    # Temperature Scaling

    # temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
    # def compute_loss():
    #     y_pred_model_w_temp = tf.math.divide(prob_x_calib, temp)
    #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
    #                                 tf.convert_to_tensor(keras.utils.to_categorical(Y)), y_pred_model_w_temp))
    #     return loss

    # optimizer = tf.optimizers.Adam(learning_rate=0.01)
    # print(f"Temperature Initial value: {temp.numpy()}")
    # for i in range(300):
    #     opts = optimizer.minimize(compute_loss, var_list=[temp])
    # print(f"Temperature Final value: {temp.numpy()}")


    model_calib = BetaCalibration(parameters="abm")
    # model_calib = LogisticRegression(C=99999999999)
    # model_calib = IsotonicRegression()
    model_calib.fit(prob_x_calib[:,0].reshape(-1, 1), y_calib)

    prob_x_test_calib = model_calib.predict(prob_x_test[:,0].reshape(-1, 1))
    second_class_probs = np.ones(len(prob_x_test_calib)) - prob_x_test_calib
    prob_x_test_calib = np.concatenate((prob_x_test_calib.reshape(-1,1), second_class_probs.reshape(-1,1)), axis=1)

    # check ECE value for normal and calib model
    model_ece = expected_calibration_error(prob_x_test, predictions_x_test, y_test, bins=10, equal_bin_size=True)
    modelcalib_ece = expected_calibration_error(prob_x_test_calib, predictions_x_test, y_test, bins=10, equal_bin_size=True)
    print("Normal ECE ", model_ece)
    print("Calib  ECE ", modelcalib_ece)

    # # unc Q id
    # tu, eu, au = unc.model_uncertainty(model, x_test, x_train, y_train)
    # tumc, eumc, aumc = unc.calib_ens_member_uncertainty(model, x_test, x_train, y_train, x_calib, y_calib, calib_method)

    tu = unc.calib_ens_total_uncertainty(prob_x_test)
    tuc = unc.calib_ens_total_uncertainty(prob_x_test_calib)


    # # acc-rej
    # # ens
    tu_auroc = uncM.unc_auroc(predictions_x_test, y_test, tu)
    # eu_auroc = uncM.unc_auroc(predictions_x_test, y_test, eu)
    # au_auroc = uncM.unc_auroc(predictions_x_test, y_test, au)
    # # ens member calib
    # tumc_auroc = uncM.unc_auroc(predictions_x_test, y_test, tumc) # predictions_x_test might change after calibration
    # eumc_auroc = uncM.unc_auroc(predictions_x_test, y_test, eumc)
    # aumc_auroc = uncM.unc_auroc(predictions_x_test, y_test, aumc)
    # # ens calib
    tuc_auroc = uncM.unc_auroc(predictions_x_test, y_test, tuc)


    print("total uncertainty normal auc ", tu_auroc)
    print("total uncertainty Calib  auc ", tuc_auroc)
    
    # return tu_auroc, eu_auroc, au_auroc, tumc_auroc, eumc_auroc, aumc_auroc, tuc_auroc
    return tu_auroc, tuc_auroc

ray.init()
for dataset in dataset_list:
    # load data
    features, target = dp.load_data(dataset)
    model_path = f"Models/NN_{dataset}"
    ray_array = []
    for seed in range(1): # 10 
        ray_array.append(calib_ale_test.remote(features, target, model_path, seed))
        # calib_ale_test(features, target, model_path, seed)

    res_array = np.array(ray.get(ray_array)).mean(axis=0)

    res_txt = f"dataset {dataset}\n"
    res_txt += f"------------------------------------acc-rej\n"
    res_txt += "Ens \n"
    res_txt += f"{res_array[0]* 100:.2f} Total uncertainty\n"
    # res_txt += f"{res_array[1]* 100:.2f} Epist uncertainty\n"
    # res_txt += f"{res_array[2]* 100:.2f} Aleat uncertainty\n"
    # res_txt += "MemberCalib \n"
    # res_txt += f"{res_array[3]* 100:.2f} Total uncertainty\n"
    # res_txt += f"{res_array[4]* 100:.2f} Epist uncertainty\n"
    # res_txt += f"{res_array[5]* 100:.2f} Aleat uncertainty\n"
    res_txt += "EnsCalib \n"
    res_txt += f"{res_array[1]* 100:.2f} Total uncertainty\n"

    if not os.path.exists(run_name):
        os.makedirs(run_name)

    with open(f"{run_name}/{dataset}_uncCalib.txt", "w") as text_file:
        text_file.write(res_txt)
    print(f"{dataset} done")
