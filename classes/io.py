import os
import pickle
import shutil
from .data_pair import DataPair
from os import path

class Io:
    
    # Schema: [descriptor.[details.]]datatype.pickle
    # E.g. text.pickle
    # E.g. bert.embeddings.pickle
    # E.g. doc2vec.dim50-epochs50.embeddings.pickle
    
    # Datatypes
    DATATYPE_TEXT = "text"
    DATATYPE_EMBEDDINGS = "embeddings"
    DATATYPE_META_MODEL = "meta-model"
    
    # Descriptors
    DESCRIPTOR_BERT = "bert"
    DESCRIPTOR_DOC_TO_VEC = "doc2vec"
    
    # Filetype extensions
    FILETYPE_EXTENSION_PICKLE = "pickle"
    FILETYPE_EXTENSION_MODEL = "model"
    

    def __init__(self, data_directory):
        self.data_directory = data_directory
    
    
    def get_datasets(self):
        return self.get_subdirectories(self.data_directory)
    
    def get_dataset_files(self, dataset_id):
        return self.get_files(self.get_path_directory(dataset_id))
    
    
    def delete_dataset(self, dataset_id):
        path = self.get_path_directory(dataset_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            print('Deleted:', dataset_id)
        else:
            print('Dataset does not exist:', dataset_id)
    
    def delete_dataset_file(self, dataset_id, filename):
        path = self.get_path_filename(dataset_id, filename)
        if os.path.isfile(path):
            os.remove(path)
            print('Deleted:', dataset_id, filename)
        else:
            print('Dataset file does not exist:', dataset_id, filename)
           
        
    def save_pickle(self, data, dataset_id, datatype_id, descriptor=None, details=None):
        directory = self.get_path_directory(dataset_id)
        file = self.get_path(dataset_id, datatype_id, descriptor, details)
        
        if(data is None):
            print('No data given, will not save:', file)
            return
        
        if not os.path.isfile(file):
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file, 'wb') as handle:
                pickle.dump(data.get_data(), handle)
                print('Wrote:', file)
        else:
            print('File already exists. Will not overwrite:', dataset_id, datatype_id)
        
    def load_pickle(self, dataset_id, datatype_id, descriptor=None, details=None):
        file = self.get_path(dataset_id, datatype_id, descriptor, details)
        if os.path.isfile(file):
            with open(self.get_path(dataset_id, datatype_id, descriptor, details), 'rb') as handle:
                print('Loaded', file)
                return pickle.load(handle)
        else:
            print('Dataset file does not exist:', file)

    def load_data_pair(self, dataset_id, datatype_id, descriptor=None, details=None):
        return DataPair(self.load_pickle(dataset_id, datatype_id, descriptor, details))

    
    def get_path_data_directory(self):
        return self.data_directory
    
    def get_path_directory(self, dataset_id):
        return self.data_directory + '/' + dataset_id
    
    def get_path_filename(self, dataset_id, filename):
        return self.get_path_directory(dataset_id) + '/' +  filename
    
    def get_path(self, dataset_id, datatype_id=None, descriptor=None, details=None, filetype_extension=FILETYPE_EXTENSION_PICKLE):
        directory_path = self.get_path_directory(dataset_id)
        if (datatype_id is None):
            path = directory_path
        else: 
            suffix = datatype_id + '.' + filetype_extension
            if (descriptor is None):
                path = directory_path + '/' + suffix
            else:
                if (details is None):
                    path = directory_path + '/' + descriptor + '.' + suffix
                else:
                    path = directory_path + '/' + descriptor + '.' + details + '.' + suffix
        return path
    
    
    def exists(self, dataset_id, datatype_id=None, descriptor=None, details=None, filetype_extension=FILETYPE_EXTENSION_PICKLE):
        return path.exists(self.get_path(dataset_id, datatype_id, descriptor, details, filetype_extension))
    
    
    def get_subdirectories(self, directory):
        if os.path.exists(directory):
            return [ f.name for f in os.scandir(directory) if f.is_dir() and not f.name.startswith('.') ]
        else:
            return []
    
    def get_files(self, directory):
        if os.path.exists(directory):
            return [ f.name for f in os.scandir(directory) if f.is_file() ]
        else:
            return []