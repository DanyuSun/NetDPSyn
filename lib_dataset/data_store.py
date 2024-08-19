import pickle
import os
import sys
sys.path.append('/Users/sdy/Desktop/dpsyn_clean_newest/')
#sys.path.append('/home/dsun/dpsyn_clean_newest/')

import config_dpsyn

from lib_view.view import View
from lib_dataset.domain import Domain
from pandas import DataFrame

import csv

class DataStore:
    def __init__(self, args):
        self.args = args
        
        self.determine_data_path()
        self.generate_folder()
    
    def determine_data_path(self):
        synthesized_records_name = '_'.join((self.args['dataset_name'], str(self.args['epsilon']), str(self.args['update_iterations']), self.args['initialize_method']))
        marginal_name = '_'.join((self.args['dataset_name'], str(self.args['epsilon']), str(self.args['update_iterations']),  self.args['initialize_method']))
        
        self.synthesized_records_file = config_dpsyn.SYNTHESIZED_RECORDS_PATH + synthesized_records_name
        self.marginal_file = config_dpsyn.MARGINAL_PATH + marginal_name
        
    def generate_folder(self):
        for path in config_dpsyn.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)
    
    def load_processed_data(self):
        return pickle.load(open(config_dpsyn.PROCESSED_DATA_PATH + self.args['dataset_name'], 'rb'))
    
    def save_synthesized_records(self, records):
        pickle.dump(records, open(self.synthesized_records_file, 'wb'))
        
    def save_marginal(self, marginals):
        pickle.dump(marginals, open(self.marginal_file, 'wb'))
    
    def load_marginal(self):
        return pickle.load(open(self.marginal_file, 'rb'))

    def save_one_marginal(self, step, data, attr_recode = None):
        #ZL: added to debug the marginal
        if self.args['dump_marginal'] == 'None':
            return
        #parse marginal_name      
        marginal_name = self.args['dump_marginal']

        if type(data) is DataFrame:
            df = data
            selected_attrs = list(marginal_name.split("_"))
            if not set(selected_attrs).issubset(df.columns):
                print(marginal_name + "isn't in dataset domain")
                return
            # Compute the marginal and ungroup the zipped attributes 
            marginal = df[selected_attrs].value_counts().reset_index()
            marginal.columns = selected_attrs + ['count']
            if attr_recode is not None:
                marginal =attr_recode.decode_marginal(marginal, selected_attrs)
        elif type(data) is View:
            view = data
            #view is one marginal selected from the whole dataset
            selected_attrs = tuple(marginal_name.split("_"))
            if not set(selected_attrs) == view.attr_set:
                #print(marginal_name + "isn't in dataset domain")
                return
            marginal = view.decode_records(attr_recode)        
        sorted_marginal = marginal.sort_values(by='count', ascending=False)
        #TBD: reverse mapping on the marginal
        csv_file = config_dpsyn.MARGINAL_PATH + self.args["dataset_name"] + "_" + str(self.args["epsilon"]) + "_" + marginal_name + "_" + step + ".csv"
        sorted_marginal.to_csv(csv_file, index=False)
           