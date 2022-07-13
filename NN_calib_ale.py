import os
from matplotlib.pyplot import axes 
import numpy as np
from sklearn import ensemble
import Uncertainty as unc
import UncertaintyM as uncM
import Data.data_provider as dp
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration

import ray

Train_new = False

# dataset_list = ['CIFAR100', 'CIFAR10'] # 'fashionMnist', 'amazon_movie'
# dataset_list = ['fashionMnist'] 
dataset_list = ['CIFAR100'] 
run_name = "Results/Ale NN 2"

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


# @ray.remote
def calib_ale_test(ens_size, dataname, features, target, model_path, seed):

    # reshape the data to have proper images for CIFAR10
    if dataname == "CIFAR10" or dataname == "CIFAR100":
        features = np.reshape(features, (-1,3,32,32))
        features = features.transpose([0,2,3,1])
        input_shape = (32, 32, 3)
    else:
        input_shape = (features.shape[1],)

    print("------------------------------------ ", seed)
    model_path_seed = model_path + "_run" + str(seed)
    # seperate to train test calibration
    x_train, x_test_all, y_train, y_test_all = train_test_split(features, target, test_size=0.4, shuffle=True, random_state=seed)
    x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=seed) 
    print("x_train shape ", x_train.shape)
    # train normal model or load
    ensemble = []
    if not os.path.exists(model_path_seed) or Train_new==True:
        if Train_new==False:
            os.makedirs(model_path_seed)
        for i in range(ens_size):
            if Train_new==False:
                os.makedirs(model_path_seed + "_member"  + str(i))

            # Create model
            model = tf.keras.models.Sequential()

            # Add layers
            model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Dropout(0.25))

            # Layer
            model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Dropout(0.25))

            # Layer
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(1024, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax'))            
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=128, shuffle=True, epochs=50) # keras.utils.to_categorical(y_train)
            model.save(model_path_seed + "_member"  + str(i))
            ensemble.append(model)
    else:
        print("loading model path: ", model_path_seed)
        for i in range(ens_size):
            model = tf.keras.models.load_model(model_path_seed + "_member"  + str(i))
            ensemble.append(model)


    ens_x_test_prob = []
    ens_x_test_prob_calib = []
    ens_x_calib_prob = []
    
    for i in range(ens_size): 
        # print(f"member {i} Eval ", ensemble[i].evaluate(x_test, y_test))

        prob_x_test = ensemble[i].predict(x_test)
        # predictions_x_test = prob_x_test.argmax(axis=1)

        prob_x_calib = ensemble[i].predict(x_calib)
        ens_x_calib_prob.append(prob_x_calib)

        # train calibrated model: Temperature Scaling
        temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
        def compute_loss():
            y_pred_model_w_temp = tf.math.divide(prob_x_calib, temp)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                        tf.convert_to_tensor(keras.utils.to_categorical(y_calib)), y_pred_model_w_temp))
            return loss

        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        # print(f"Temperature Initial value: {temp.numpy()}")
        for i in range(300):
            opts = optimizer.minimize(compute_loss, var_list=[temp])
        # print(f"Temperature Final value: {temp.numpy()}")
        prob_x_test_calib = tf.math.divide(prob_x_test, temp).numpy()
        # predictions_x_test_calib = prob_x_test_calib.argmax(axis=1)

        ens_x_test_prob.append(prob_x_test)
        ens_x_test_prob_calib.append(prob_x_test_calib)

        # ECE result before calibration (TF code)
        # num_bins = 10
        # labels_true = tf.convert_to_tensor(y_test, dtype=tf.int32, name='labels_true')
        # logits = tf.convert_to_tensor(prob_x_test, dtype=tf.float32, name='logits')
        # print("Normal ECE TF ", tfp.stats.expected_calibration_error(num_bins=num_bins, logits=logits, labels_true=labels_true))
        # # ECE result after calibration
        # logits = tf.convert_to_tensor(prob_x_test_calib, dtype=tf.float32, name='logits')
        # print("Calib ECE TF ", tfp.stats.expected_calibration_error(num_bins=num_bins, logits=logits, labels_true=labels_true))

        # check ECE value for normal and calib model (my code)
        # model_ece = expected_calibration_error(prob_x_test, predictions_x_test, y_test, bins=10, equal_bin_size=True)
        # modelcalib_ece = expected_calibration_error(prob_x_test_calib, predictions_x_test, y_test, bins=10, equal_bin_size=True)
        # print("Normal ECE ", model_ece)
        # print("Calib  ECE ", modelcalib_ece)

    # convert ens probs to np and transpose the dimentions for uncQ
    ens_x_test_prob = np.array(ens_x_test_prob)
    ens_x_test_prob_calib = np.array(ens_x_test_prob_calib)
    ens_x_calib_prob = np.array(ens_x_calib_prob)

    ens_x_test_prob = ens_x_test_prob.transpose([1,0,2])
    ens_x_test_prob_calib = ens_x_test_prob_calib.transpose([1,0,2])
    ens_x_calib_prob = ens_x_calib_prob.transpose([1,0,2])

    # print("------------------------------------>>>>")
    # print("ens_x_test_prob shape ", ens_x_test_prob.shape)
    # print("ens_x_test_prob_calib shape ", ens_x_test_prob_calib.shape)

    ens_x_calib_prob_avg = ens_x_calib_prob.mean(axis=1)
    ens_x_test_prob_avg = ens_x_test_prob.mean(axis=1)
    ens_x_test_prob_memcalib_avg = ens_x_test_prob_calib.mean(axis=1)

    # train calibrated for ensemble avg prob
    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
    def compute_loss():
        y_pred_model_w_temp = tf.math.divide(ens_x_calib_prob_avg, temp)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                    tf.convert_to_tensor(keras.utils.to_categorical(y_calib)), y_pred_model_w_temp))
        return loss

    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    # print(f"Temperature Initial value: {temp.numpy()}")
    for i in range(300):
        opts = optimizer.minimize(compute_loss, var_list=[temp])
    # print(f"Temperature Final value: {temp.numpy()}")
    ens_x_test_prob_avg_enscalib = tf.math.divide(ens_x_test_prob_avg, temp).numpy()


    ens_x_test_predict = ens_x_calib_prob_avg.argmax(axis=1)
    ens_x_test_predict_memcalib = ens_x_test_prob_memcalib_avg.argmax(axis=1)
    ens_x_test_predict_enscalib = ens_x_test_prob_avg_enscalib.argmax(axis=1)

    # unc Q
    tu, eu, au = uncM.uncertainty_ent_bays(ens_x_test_prob, np.ones(ens_size))
    tumc, eumc, aumc = uncM.uncertainty_ent_bays(ens_x_test_prob_calib, np.ones(ens_size))

    tu = unc.calib_ens_total_uncertainty(ens_x_test_prob_avg)
    tuc = unc.calib_ens_total_uncertainty(ens_x_test_prob_avg_enscalib)


    # # acc-rej
    # # ens
    tu_auroc = uncM.unc_auroc(ens_x_test_predict, y_test, tu)
    eu_auroc = uncM.unc_auroc(ens_x_test_predict, y_test, eu)
    au_auroc = uncM.unc_auroc(ens_x_test_predict, y_test, au)
    # # ens member calib
    tumc_auroc = uncM.unc_auroc(ens_x_test_predict_memcalib, y_test, tumc) # ens_x_test_predict_memcalib might change after calibration
    eumc_auroc = uncM.unc_auroc(ens_x_test_predict_memcalib, y_test, eumc)
    aumc_auroc = uncM.unc_auroc(ens_x_test_predict_memcalib, y_test, aumc)
    # # ens calib
    tuc_auroc = uncM.unc_auroc(ens_x_test_predict_enscalib, y_test, tuc)


    # print("total uncertainty normal auc ", tu_auroc)
    # print("total uncertainty Calib  auc ", tuc_auroc)
    # exit()
    return tu_auroc, eu_auroc, au_auroc, tumc_auroc, eumc_auroc, aumc_auroc, tuc_auroc
    # return tu_auroc, tuc_auroc

# ray.init()
for dataset in dataset_list:
    # load data
    features, target = dp.load_data(dataset)
    ens_size = 10
    model_path = f"Models/NN_{dataset}"
    ray_array = []
    for seed in range(10): # 10 
        # ray_array.append(calib_ale_test.remote(ens_size, features, target, model_path, seed))
        calib_ale_test(ens_size, features, target, model_path, seed)

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
    res_txt += f"{res_array[1]* 100:.2f} Total uncertainty\n"

    if not os.path.exists(run_name):
        os.makedirs(run_name)

    with open(f"{run_name}/{dataset}_uncCalib.txt", "w") as text_file:
        text_file.write(res_txt)
    print(f"{dataset} done")
