
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import resample
from sklearn import preprocessing
from scipy.io import arff

def load_data(data_name):   

	if data_name == "Jdata/dbpedia":
		features, target = datasets.load_svmlight_file("Data/Jdata/dbpedia_train.svm")
	if data_name == 'amazon_movie':
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
		targets = targets[:,0]
		return features, targets


	df = pd.read_csv(f'./Data/{data_name}.csv')
	features = np.array(df.drop("Class", axis=1))
	target = np.array(df["Class"])

	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	if "cifar10" == data_name or "mnist" in data_name:
		features = features.astype('float32')
		features /= 255

	if "digits" == data_name:
		features = features.astype('float32')
		features /= 16
	if "cifar10small" == data_name:
		features = features.astype('float32')
		features /= 256

	return features, target

def load_ood(data_name, split=0.3, calibration_set=True, seed=1):
	features, target = load_data(data_name)
	np.random.seed(seed)
	classes = np.unique(target)
	selected_id = np.random.choice(classes,int(len(classes)/2),replace=False) # select id classes
	selected_id_index = np.argwhere(np.isin(target,selected_id)) # get index of all id instances
	selected_ood_index = np.argwhere(np.isin(target,selected_id,invert=True)) # get index of all not selected classes (OOD)

	target_id    = target[selected_id_index].reshape(-1)
	features_id  = features[selected_id_index].reshape(-1, features.shape[1])
	target_ood   = target[selected_ood_index].reshape(-1)
	features_ood = features[selected_ood_index].reshape(-1, features.shape[1])    

	x_train, x_test_id, y_train, y_test_id  = split_data(features_id, target_id,   split=split, seed=seed)
	_      , x_test_ood, _     , y_test_ood = split_data(features_ood, target_ood, split=split, seed=seed)

	if calibration_set:
		x_test_id, x_calib, y_test_id, y_calib = train_test_split(x_test_id, y_test_id, test_size=0.5, shuffle=True, random_state=1)
	else:
		x_calib, y_calib = 0, 0

	minlen = len(x_test_id)
	if len(x_test_ood) < minlen:
		minlen = len(x_test_ood)

	y_test_idoodmix = np.concatenate((np.zeros(minlen), np.ones(minlen)), axis=0)
	x_test_idoodmix = np.concatenate((x_test_id[:minlen], x_test_ood[:minlen]), axis=0)

	return x_train, y_train, x_test_id, y_test_id, x_test_ood, y_test_ood, x_test_idoodmix, y_test_idoodmix, x_calib, y_calib

def split_data(features, target, split, seed=1):
   x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=split, shuffle=True, random_state=seed, stratify=target)
   return x_train, x_test, y_train, y_test

def balance_dataset(df):
	# Separate majority and minority classes
	y = df.Class.unique()
	df_loss = df[df.Class==y[0]]
	df_win  = df[df.Class==y[1]]
	
	max_len = len(df_loss)
	if len(df_win) > max_len:
		max_len = len(df_win)
	# Upsample minority class
	df_upsampled_win = resample(df_win, 
                                replace=True,           # sample with replacement
                                n_samples=max_len,      # to match majority class
                                random_state=123)       # reproducible results
	 
	# Combine majority class with upsampled minority class
	df_balance = pd.concat([df_loss, df_upsampled_win])
	return df_balance

def load_arff_data(name="adult", convert_to_int=True, type="binary", log=True):
	dataset = arff.load(open(f'/home/mhshaker/projects/uncertainty/Data/{name}.arff', 'r'))
	df = pd.DataFrame(dataset['data'])
	df = df.sample(frac=1).reset_index(drop=True)
	df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
	if log:
		print(f"Data name = {name}")
		print(df.head())
		print(df['target'].value_counts())

	if convert_to_int:
		for column in df:
			df[column] = df[column].astype("category").cat.codes
	# print(df.head())
	# exit()
	features = df.drop("target", axis=1)
	# print(features)
	features = preprocessing.scale(features)
	# print(features)

	# features_names = list(features.columns)
	# target_names = ["class1","class2"]

	return features, df.target #features_names, target_names

def load_arff_2(data_name):
	data = arff.loadarff(f"./Data/{data_name}.arff")
	df = pd.DataFrame(data[0])
	df.rename(columns={ df.columns[-1]: "target" }, inplace = True)
	if data_name == "MagicTelescope":
		df.drop("ID", axis=1, inplace=True)

	features = df.drop("target", axis=1)
	target = df.target

	le = preprocessing.LabelEncoder()
	le.fit(target)
	target = le.transform(target)

	# print(features.head())

	return np.array(features), np.array(target)
