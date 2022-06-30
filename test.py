import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

features = []
targets = []
for i in ["1", "2", "3", "4", "5", "test"]:
    d = unpickle(f"Data/cifar-10-batches-py/data_batch_{i}")
    features.append(d[b'data'])
    targets.append(d[b'labels'])

features = np.array(features)
features = np.reshape(features, (-1, features.shape[-1]))
targets = np.reshape(np.array(targets), (-1,))
# print(d.keys())

# print(d[b'labels'])
print(features.shape)
print(targets.shape)