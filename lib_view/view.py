import numpy as np
import pandas as pd
from lib_view.non_negativity import NonNegativity


class View:
    def __init__(self, view_domain, dataset_domain):
        self.indicator = np.zeros(len(dataset_domain.attrs), dtype=np.uint8)
        self.indicator[[dataset_domain.attr_index_mapping[attr] for attr in view_domain.attrs]] = 1
        self.num_categories = np.array(dataset_domain.shape)

        self.attributes_index = np.nonzero(self.indicator)[0]
        self.attr_set = set(view_domain.attrs)
        self.attr_names = [dataset_domain.attrs[i] for i in self.attributes_index]

        self.num_key = np.product(self.num_categories[self.attributes_index])
        self.num_attributes = self.indicator.shape[0]
        self.ways = np.count_nonzero(self.indicator)
        
        self.encode_num = np.zeros(self.ways, dtype=np.uint32)
        self.cum_mul = np.zeros(self.ways, dtype=np.uint32)

        self.count = np.zeros(self.num_key)
        self.rho = 0.0
        
        self.calculate_encode_num(self.num_categories)
    
    ########################################### general functions ####################################
    def calculate_encode_num(self, num_categories):
        if self.ways != 0:
            categories_index = self.attributes_index
            
            categories_num = num_categories[categories_index]
            categories_num = np.roll(categories_num, 1)
            categories_num[0] = 1
            self.cum_mul = np.cumprod(categories_num)
            
            categories_num = num_categories[categories_index]
            categories_num = np.roll(categories_num, self.ways - 1)
            categories_num[-1] = 1
            categories_num = np.flip(categories_num)
            self.encode_num = np.flip(np.cumprod(categories_num))
    
    def calculate_tuple_key(self):
        self.tuple_key = np.zeros([self.num_key, self.ways], dtype=np.uint32)
        
        if self.ways != 0:
            for i in range(self.attributes_index.shape[0]):
                index = self.attributes_index[i]
                categories = np.arange(self.num_categories[index])
                column_key = np.tile(np.repeat(categories, self.encode_num[i]), self.cum_mul[i])
                
                self.tuple_key[:, i] = column_key
        else:
            self.tuple_key = np.array([0], dtype=np.uint32)
            self.num_key = 1
    
    def count_records(self, records):        
        encode_records = np.matmul(records[:, self.attributes_index], self.encode_num)
        encode_key, count = np.unique(encode_records, return_counts=True)
        
        indices = np.where(np.isin(np.arange(self.num_key), encode_key))[0]
        try:
            self.count[indices] = count
        except:
            raise Exception("num_key is smaller than the largest value, check domain shape")

    def decode_records(self, attr_recode = None):
        decoded_records = []
        self.calculate_tuple_key()
        # Iterate over all indices in self.count
        for encoded_value, count in zip(np.arange(self.num_key), self.count):
            decoded_value = self.tuple_key[encoded_value,:]
            decoded_value_with_count = list(decoded_value) + [count]
            decoded_records.append(decoded_value_with_count)

        # Create a DataFrame from the decoded records
        columns = self.attr_names + ['count']
        decoded_df = pd.DataFrame(decoded_records, columns=columns)

        if attr_recode is not None:
            #replacing the grouped attribute values
            decoded_df = attr_recode.decode_marginal(decoded_df, self.attr_names)

        return decoded_df

    def filter_count(self, threshold):
        self.count = self.count[self.count >= threshold]
        self.num_key = len(self.count)

    def calculate_normalize_count(self):
        self.normalize_count = self.count / np.sum(self.count)
        return self.normalize_count
    
    def calculate_count_matrix(self):
        shape = []
        
        for attri in self.attributes_index:
            shape.append(self.num_categories[attri])
        
        self.count_matrix = np.copy(self.count).reshape(tuple(shape))
        
        return self.count_matrix
    
    def calculate_count_matrix_2d(self, row_attr_index):
        attr_index = np.where(self.attributes_index == row_attr_index)[0]
        shape = [1, 1]
        
        for attri in self.attributes_index:
            if attri == row_attr_index:
                shape[0] *= self.num_categories[attri]
            else:
                shape[1] *= self.num_categories[attri]
                
        self.count_matrix_2d = np.zeros(shape)
        
        for value in range(shape[0]):
            indices = np.where(self.tuple_key[:, attr_index] == value)[0]
            self.count_matrix_2d[value] = self.count[indices]
            
        return self.count_matrix_2d
    
    def reserve_original_count(self):
        self.original_count = self.count
    
    def get_sum(self):
        self.sum = np.sum(self.count)
    
    def generate_attributes_index_set(self):
        self.attributes_set = set(self.attributes_index)
    
    ################################### functions for outside invoke #########################
    def calculate_encode_num_general(self, attributes_index):
        categories_index = attributes_index
        
        categories_num = self.num_categories[categories_index]
        categories_num = np.roll(categories_num, attributes_index.size - 1)
        categories_num[-1] = 1
        categories_num = np.flip(categories_num)
        encode_num = np.flip(np.cumprod(categories_num))
        
        return encode_num
    
    def count_records_general(self, records):
        count = np.zeros(self.num_key)
        
        encode_records = np.matmul(records[:, self.attributes_index], self.encode_num)
        encode_key, value_count = np.unique(encode_records, return_counts=True)
        
        indices = np.where(np.isin(np.arange(self.num_key), encode_key))[0]
        count[indices] = value_count
        
        return count
    
    def calculate_normalize_count_general(self, count):
        return count / np.sum(count)
    
    def calculate_count_matrix_general(self, count):
        shape = []
        
        for attri in self.attributes_index:
            shape.append(self.num_categories[attri])
        
        return np.copy(count).reshape(tuple(shape))
    
    def calculate_tuple_key_general(self, unique_value_list):
        self.tuple_key = np.zeros([self.num_key, self.ways], dtype=np.uint32)
        
        if self.ways != 0:
            for i in range(self.attributes_index.shape[0]):
                categories = unique_value_list[i]
                column_key = np.tile(np.repeat(categories, self.encode_num[i]), self.cum_mul[i])
                
                self.tuple_key[:, i] = column_key
        else:
            self.tuple_key = np.array([0], dtype=np.uint32)
            self.num_key = 1
    
    def project_from_bigger_view_general(self, bigger_view):
        encode_num = np.zeros(self.num_attributes, dtype=np.uint32)
        encode_num[self.attributes_index] = self.encode_num
        encode_num = encode_num[bigger_view.attributes_index]
        
        encode_tuple_key = np.matmul(bigger_view.tuple_key, encode_num)
        self.count = np.bincount(encode_tuple_key, weights=bigger_view.count, minlength=self.num_key)
    
    ######################### functions for consistency #######################
    def init_consist_parameters(self, num_target_views):
        self.summations = np.zeros([self.num_key, num_target_views])
        self.weights = np.zeros(num_target_views)
        self.rhos = np.zeros(num_target_views)

    def calculate_delta(self):
        weights = self.rhos * self.weights
        target = np.matmul(self.summations, weights) / np.sum(weights)
        self.delta = - (self.summations.T - target).T * weights
    
    def project_from_bigger_view(self, bigger_view, index):
        encode_num = np.zeros(self.num_attributes, dtype=np.uint32)
        encode_num[self.attributes_index] = self.encode_num
        encode_num = encode_num[bigger_view.attributes_index]
        
        encode_tuple_key = np.matmul(bigger_view.tuple_key, encode_num)
        
        self.weights[index] = 1.0 / np.product(
            self.num_categories[np.setdiff1d(bigger_view.attributes_index, self.attributes_index)])
        self.rhos[index] = bigger_view.rho

        self.summations[:, index] = np.bincount(encode_tuple_key, weights=bigger_view.count, minlength=self.num_key)

    ############### used in views to be consisted ###############
    def update_view(self, common_view, index):
        encode_num = np.zeros(self.num_attributes, dtype=np.uint32)
        encode_num[common_view.attributes_index] = common_view.encode_num
        encode_num = encode_num[self.attributes_index]
        
        encode_tuple_key = np.matmul(self.tuple_key, encode_num)

        sort_indices = np.argsort(encode_tuple_key)
        _, count = np.unique(encode_tuple_key, return_counts=True)
        np.add.at(self.count, sort_indices, np.repeat(common_view.delta[:, index], count))

    ######################################### non-negative functions ####################################
    def non_negativity(self, method, iteration=-1):
        if method == "N3":
            assert iteration != -1
        
        non_negativity = NonNegativity(self.count)
        
        if method == "N1":
            self.count = non_negativity.norm_cut()
        elif method == "N2":
            self.count = non_negativity.norm_sub()
        elif method == "N3":
            if iteration < 400:
                self.count = non_negativity.norm_sub()
            else:
                self.count = non_negativity.norm_cut()
        else:
            raise Exception("non_negativity method is invalid")


if __name__ == "__main__":
    pass
