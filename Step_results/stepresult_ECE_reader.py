import numpy as np


with open('Step_results/res_array_all.npy', 'rb') as f:
    res_array_all = np.load(f, allow_pickle=True) 
    # print(res_array_all)
    # print("------------------------------------")

normal_cw_ece = np.array(res_array_all[:,-3])
a_normal_cw_ece = []
for i in normal_cw_ece:
    a_normal_cw_ece.append(i)
normal_cw_ece = np.array(a_normal_cw_ece)

sk_iso_cw_ece = np.array(res_array_all[:,-2])
a_sk_iso_cw_ece = []
for i in sk_iso_cw_ece:
    a_sk_iso_cw_ece.append(i)
sk_iso_cw_ece = np.array(a_sk_iso_cw_ece)

dir_cw_ece    = np.array(res_array_all[:,-1])
a_dir_cw_ece = []
for i in dir_cw_ece:
    a_dir_cw_ece.append(i)
dir_cw_ece = np.array(a_dir_cw_ece)


print("------------------------------------")

print("normal_cw_ece \n", normal_cw_ece.mean(axis=0).sum())
print("sk_iso_cw_ece \n", sk_iso_cw_ece.mean(axis=0).sum())
print("dir_cw_ece \n", dir_cw_ece.mean(axis=0).sum())
