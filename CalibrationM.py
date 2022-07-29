import numpy as np

def classwise_ECE(probs, y_true, bins=10, equal_bin_size=True, full_ece=False):
    
    class_ece = []

    for class_index in range(len(np.unique(y_true))):

        class_prob = probs[:,class_index] # select probs form i th class
        correctness_map = np.where(class_index==y_true, 1, 0) # determine which predictions are correct

        if equal_bin_size == False:
            bin_size = 1/bins
            for bin in range(bins):
                bin_indexes = np.where((class_prob > bin * bin_size) & (class_prob <= (bin+1) * bin_size))[0]
                if len(bin_indexes) > 0:
                    bin_conf = class_prob[bin_indexes].mean()
                    bin_class_friq = correctness_map[bin_indexes].sum() / len(bin_indexes)
                    dif = abs(bin_conf - bin_class_friq)
                    class_ece.append(dif * len(bin_indexes) / len(probs))  # difference times the number of instances in the bin divided by the total number of instances in the dataset
        else: # equal_bin_size
            sorted_index = np.argsort(class_prob, kind='stable') # sort probs
            class_prob = class_prob[sorted_index]
            correctness_map = correctness_map[sorted_index]

            bin_size = int(len(probs) / bins)
            for bin in range(bins):
                bin_conf = class_prob[bin*bin_size:(bin+1)*bin_size].mean()
                bin_class_friq = correctness_map[bin*bin_size:(bin+1)*bin_size].sum() / bin_size
                dif = abs(bin_conf - bin_class_friq)
                class_ece.append(dif * bin_size / len(probs))

    if full_ece:
        return sum(class_ece)
    else:
        return class_ece 

def confidance_ECE(probs, predictions, y_true, bins=10, equal_bin_size=True):
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
                ece += dif * len(bin_indexes) / len(probs)  # difference times the number of instances in the bin divided by the total number of instances in the dataset
    else: # equal_bin_size
        sorted_index = np.argsort(prob_max, kind='stable') # sort probs
        prob_max = prob_max[sorted_index]
        correctness_map = correctness_map[sorted_index]

        bin_size = int(len(predictions) / bins)
        for bin in range(bins):
            bin_conf = prob_max[bin*bin_size:(bin+1)*bin_size].mean()
            bin_acc = correctness_map[bin*bin_size:(bin+1)*bin_size].sum() / bin_size
            dif = abs(bin_conf - bin_acc)
            ece += dif * bin_size / len(probs)

    return ece 

