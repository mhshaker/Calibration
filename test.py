def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

d = unpickle("Data/cifar-10-batches-py/data_batch_1")

print(d["b'labels'"])