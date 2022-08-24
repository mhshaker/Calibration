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
calibration_method = "iso"


data_size = 100

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
plt.scatter(x, np.full(len(x), "Add Data"), c=y, cmap=matplotlib.colors.ListedColormap(colors))
x1, y1 = [1, 1], [1, -1]
x2, y2 = [2, 2], [1, -1]
plt.plot(x1, y1, x2, y2, c="gray")

model = RandomForestClassifier(max_depth=1, n_estimators=3, random_state=seed)
model.fit(x_train, y_train)
predictions_x_test = model.predict(x_test)
prob_x_test = model.predict_proba(x_test)
prob_x_calib = model.predict_proba(x_calib)

figure(figsize=(30, 3), dpi=80)

for est_index, estimator in enumerate(model.estimators_):
    # tree.plot_tree(estimator)
    tree_threshold = estimator.tree_.threshold[0]
    tree_prob_x_test = estimator.predict_proba(x_test)

    # print(tree_prob_x_test)
    
    print(f"Tree [{est_index}] threshold {tree_threshold}")
    print(f"Normal          ECE {calibm.confidance_ECE(tree_prob_x_test, y_test)} ", np.unique(tree_prob_x_test[:,0]))

    # Full Dirichlet
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    model_calib = CalibratedClassifierCV(estimator, cv="prefit", method="isotonic") # cv=30
    model_calib.fit(x_calib , y_calib)
    tree_prob_x_test_calib = model_calib.predict_proba(x_test)

    print(f"calibration iso ECE {calibm.confidance_ECE(tree_prob_x_test_calib, y_test)} ", np.unique(tree_prob_x_test_calib[:,0]))

    calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
    dir_calib = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg, 'reg_mu': [None]}, cv=skf, scoring='neg_log_loss')
    dir_calib.fit(prob_x_calib , y_calib)
    tree_prob_x_test_calib = dir_calib.predict_proba(tree_prob_x_test)
    
    print(f"calibration dir ECE {calibm.confidance_ECE(tree_prob_x_test_calib, y_test)} ", np.unique(tree_prob_x_test_calib[:,0]))

    # optimal prob
    a = 1
    b = tree_threshold - 1
    c = 2 - tree_threshold
    o_p_neg = a/(a+b) + (b * (1/2)) /(a+b)
    o_p_pos = a/(a+c) + (c * (1/2)) /(a+c)
    # o_p_pos = (c * (1/2))
    print(f"optimal prob ", [1 - o_p_pos, o_p_neg])
    print("------------------------------------")

    plt.plot([tree_threshold,tree_threshold], [3,-1], c="blue")
    plt.annotate(est_index, (tree_threshold+0.01, -1))

plt.scatter(x_train, np.full(len(x_train), "Train"), c=y_train, cmap=matplotlib.colors.ListedColormap(colors))
# for i, txt in enumerate(np.ones(len(x_train))): # attempt to label the training data of each tree in the RF (plot works but cannot access the training subsets)
#     plt.annotate(int(txt), (x_train[i], "Train"))

plt.scatter(x_calib, np.full(len(x_calib), "Calibration"), c=y_calib, cmap=matplotlib.colors.ListedColormap(colors))
plt.scatter(x_test, np.full(len(x_test), "Test"), c=y_test, cmap=matplotlib.colors.ListedColormap(colors))
x1, y1 = [1, 1], [3, -1]
x2, y2 = [2, 2], [3, -1]
plt.plot(x1, y1, x2, y2, c="gray")


model_calib = CalibratedClassifierCV(model, cv="prefit", method="isotonic") # cv=30
model_calib.fit(x_calib , y_calib)
prob_x_test_calib = model_calib.predict_proba(x_test)

print(f"Normal          ECE {calibm.confidance_ECE(prob_x_test, y_test)} {np.unique(prob_x_test[:,0])}")
print(f"calibration iso ECE {calibm.confidance_ECE(prob_x_test_calib, y_test)} ", np.unique(prob_x_test_calib[:,0]))

# Full Dirichlet
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
reg = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
calibrator = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
dir_calib = GridSearchCV(calibrator, param_grid={'reg_lambda':  reg, 'reg_mu': [None]}, cv=skf, scoring='neg_log_loss')
dir_calib.fit(prob_x_calib , y_calib)
prob_x_test_calib = dir_calib.predict_proba(prob_x_test)

print(f"calibration dir ECE {calibm.confidance_ECE(prob_x_test_calib, y_test)} ", np.unique(prob_x_test_calib[:,0]))

normal_cw_ece = calibm.classwise_ECE(prob_x_test, y_test)
normal_conf_ece = calibm.confidance_ECE(prob_x_test, y_test)

print(normal_cw_ece)
print(normal_conf_ece)


# unc Q id
tu, eu, au = unc.model_uncertainty(model, x_test, x_train, y_train)
tumc, eumc, aumc, porb_matrix = unc.calib_ens_member_uncertainty(model, x_test, y_test, x_train, y_train, x_calib, y_calib, calibration_method, seed)
tuc = unc.calib_ens_total_uncertainty(prob_x_test_calib)