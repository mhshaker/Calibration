# import ray
# from time import sleep

# @ray.remote
# def task(index, run):
#     sleep(0.5)
#     print(f"task {index} done")
#     return f"run {run} task {index}"

# ray.init()


# for run in ["a", "b", "c"]:
#     ray_array = []
#     for i in range(10):
#         ray_array.append(task.remote(i, run))
#     res_array = ray.get(ray_array)
#     print(f"run {run} done")
#     print(res_array)
#     print("------------------------------------")

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.io import arff
# dataset = arff.load(open(f'./Data/MagicTelescope.arff', 'r'))
# df = pd.DataFrame(dataset['data'])
# df = df.sample(frac=1).reset_index(drop=True)
# 

data = arff.loadarff('./Data/KDDCup09_upselling.arff')
df = pd.DataFrame(data[0])
df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
print(df.head(20))

features = df.drop("target", axis=1)
target = df.target

le = preprocessing.LabelEncoder()
le.fit(target)
target = le.transform(target)

print(features.shape)