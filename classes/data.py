from datetime import datetime

class Data:

    KEY_META = "meta"
    
    KEY_META_CREATION_DATE = "creation-date"
    KEY_META_CREATION_SECONDS = "creation-seconds"
    KEY_META_DESCRIPTION = "description"
    
    KEY_META_MODEL_FILENAME = "model-filename"
    
    def __init__(self, data=None, storage_ids=None):
        if not (data is None):
            self.data = data
        else:
            self.data = {}
            self.data[self.KEY_META] = {}
            if not (storage_ids is None):
                for storage_id in storage_ids:
                    self.data[storage_id] = {}

            self.add_meta(self.KEY_META_CREATION_DATE, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
    def add(self, storage_id, element_id, element_data):
        self.data[storage_id][element_id] = element_data
        
    def add_meta(self, key, value):
        self.add(self.KEY_META, key, value)
    
    
    def get_data(self):
        return self.data
    
    def get(self, storage_id):
        return self.data[storage_id]
    
    def get_dict_as_lists(self, storage_id):
        return (list(self.get(storage_id).keys()), list(self.get(storage_id).values()))
    
    def get_meta(self):
        if (self.KEY_META in self.data):
            return self.data[self.KEY_META]
        else:
            return {}

        
    def set_runtime(self):
        if (self.KEY_META_CREATION_DATE in self.data[self.KEY_META]):
            creation = datetime.strptime(self.data[self.KEY_META][self.KEY_META_CREATION_DATE], '%Y-%m-%d %H:%M:%S')
            self.add_meta(self.KEY_META_CREATION_SECONDS, (datetime.now() - creation).total_seconds())