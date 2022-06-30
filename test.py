import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

features = []
targets = []
for i in ["train", "test"]:
    d = unpickle(f"Data/cifar-100-python/{i}")
    f = d[b'data']
    # print(d.keys())
    features.append(np.array(d[b'data']))
    targets.append(np.array(d[b'fine_labels']))

features = np.concatenate((features[0], features[1]))
targets = np.concatenate((targets[0], targets[1]))


# print(d[b'labels'])
print(features.shape)
print(targets.shape)