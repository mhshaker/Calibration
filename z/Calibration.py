from os import sep
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import Uncertainty as unc
import UncertaintyM as uncM
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# Class for loading data
import classes.io as iox
io = iox.Io('./')

# Identifiers of dataset
dataset_id = 'amazon-movie-reviews-10000'
descriptor = io.DESCRIPTOR_DOC_TO_VEC
details = 'dim50-epochs50'

# [My note] a -> 1 star. b -> 5 star. First dimention is the key [0] and the text [1]. Second dimention is the index of documents

# Load data text
texts = io.load_data_pair(dataset_id, io.DATATYPE_TEXT)

# Load data embeddings
embeddings = io.load_data_pair(dataset_id, io.DATATYPE_EMBEDDINGS, descriptor, details)

# create the dataset (with targets including the keys)
class_1 = np.array(embeddings.get_a_dict_as_lists()[1]) # get the embeddings (features) for class 1 star
class_5 = np.array(embeddings.get_b_dict_as_lists()[1]) # get the embeddings (features) for class 5 star

target_1_label = np.zeros(len(class_1)).reshape((-1,1))
target_5_label = np.ones(len(class_5)).reshape((-1,1))
target_1_key = np.array(embeddings.get_a_dict_as_lists()[0]).reshape((-1,1)) # get the keys (part of target but not the label) for class 1 star
target_5_key = np.array(embeddings.get_b_dict_as_lists()[0]).reshape((-1,1)) # get the keys for class 5 star

target_1 = np.concatenate((target_1_label,target_1_key), axis=1)
target_5 = np.concatenate((target_5_label,target_5_key), axis=1)

features = np.concatenate((class_1,class_5))
targets = np.concatenate((target_1,target_5))

# split and shuffel the data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.4, shuffle=True, random_state=1)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=1)

# train the model
model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=1)
model.fit(X_train, y_train[:,0]) # remove keys when fiting the model

predictions = model.predict(X_test)
print("model test score = ", model.score(X_test, y_test[:,0]))

# Aleatoric uncertianty for X_test
total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.model_uncertainty(model, X_test, X_train, y_train[:,0])

# AR plot
avg_acc, avg_min, avg_max, avg_random ,steps = uncM.accuracy_rejection2(predictions.reshape((1,-1)), y_test[:,0].reshape((1,-1)), total_uncertainty.reshape((1,-1)))
plt.plot(steps, avg_acc*100)
plt.savefig(f"./AR_plot.png",bbox_inches='tight')
plt.close()


gnb_isotonic = CalibratedClassifierCV(model, cv=2, method="isotonic")
gnb_sigmoid = CalibratedClassifierCV(model, cv=2, method="sigmoid")

clf_list = [
    (model, "RF"),
    (gnb_isotonic, "RF + Isotonic"),
    (gnb_sigmoid, "RF + Sigmoid"),
]

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train[:,0])
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test[:,0],
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (RF)")

# # Add histogram
# grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
# for i, (_, name) in enumerate(clf_list):
#     row, col = grid_positions[i]
#     ax = fig.add_subplot(gs[row, col])

#     ax.hist(
#         calibration_displays[name].y_prob,
#         range=(0, 1),
#         bins=10,
#         label=name,
#         color=colors(i),
#     )
#     ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.savefig(f"./Calib_plot2.png")
plt.close()


# fig = plt.figure(figsize=(10, 10))
# gs = GridSpec(4, 2)
# colors = plt.cm.get_cmap("Dark2")

# ax_calibration_curve = fig.add_subplot(gs[:2, :2])
# calibration_displays = {}

# display = CalibrationDisplay.from_estimator(
#     model,
#     X_test,
#     y_test[:,0],
#     n_bins=10,
#     name="RF",
#     ax=ax_calibration_curve,
#     color="red",
# )
# calibration_displays["RF"] = display

# ax_calibration_curve.grid()
# ax_calibration_curve.set_title("Calibration plots (RF)")

# plt.savefig(f"./Calib_plot.png",bbox_inches='tight')
# plt.close()



















# model_to_probs = {'train': pred_probs_train, 'test': pred_probs_test, 'valid': pred_probs_valid}

# plt.figure(figsize=(20,4))

# plt.subplot(1,2,1)
# sns.displot(pred_probs_train)
# plt.title(f"RF - train", fontsize=20)

# plt.subplot(1,2,2)
# sns.displot(pred_probs_test)
# plt.title(f"RF - test", fontsize=20)
# plt.savefig(f"./prob.png",bbox_inches='tight')
# plt.close()


# pred_probs = pred_probs_test
# pred_probs_space = np.linspace(pred_probs.min(), pred_probs.max(), 10)


# empirical_probs = []
# pred_probs_midpoints = []

# for i in range(len(pred_probs_space)-1):
#     empirical_probs.append(np.mean(test_y[(pred_probs > pred_probs_space[i]) & (pred_probs < pred_probs_space[i+1])]))
#     pred_probs_midpoints.append((pred_probs_space[i] + pred_probs_space[i+1])/2)

# print(pred_probs_midpoints)
# print(empirical_probs)

# plt.figure(figsize=(10,4))
# plt.plot(pred_probs_midpoints, empirical_probs, linewidth=2, marker='o')
# plt.title(f"RF", fontsize=20)
# plt.xlabel('predicted prob', fontsize=14)
# plt.ylabel('empirical prob', fontsize=14)

# plt.plot([0,1],[0,1],linestyle='--',color='gray')

# plt.legend(['original', 'ideal'], fontsize=20)
# plt.savefig(f"./Calib_plot.png",bbox_inches='tight')
# plt.close()

