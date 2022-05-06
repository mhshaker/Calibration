from .data import Data

class DataPair(Data):
    
    KEY_A = "a"
    KEY_B = "b"
    
    KEY_META_TITLE_A = "title-a"
    KEY_META_TITLE_B = "title-b"
    
    def __init__(self, data=None):
        if not (data is None):
            super().__init__(data=data)
        else:
            super().__init__(storage_ids=[self.KEY_A, self.KEY_B])
                    
        
    def add_a(self, element_id, element_data):
        self.add(self.KEY_A, element_id, element_data)
        
    def add_b(self, element_id, element_data):
        self.add(self.KEY_B, element_id, element_data)
    
    
    def get_a(self):
        return self.get(self.KEY_A)
    
    def get_b(self):
        return self.get(self.KEY_B)
    
    
    def get_a_dict_as_lists(self):
        return self.get_dict_as_lists(self.KEY_A)
    
    def get_b_dict_as_lists(self):
        return self.get_dict_as_lists(self.KEY_B)
    
        
    def get_sizes(self):
        return (len(self.get_a()), len(self.get_b()))