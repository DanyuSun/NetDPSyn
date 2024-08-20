import os
import logging
import pickle
import json
import random
import ssl
import zipfile
import os.path as osp
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

import sys
 
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(parent_dir)

import config_dpsyn
from lib_dataset.dataset import Dataset
from lib_dataset.domain import Domain



import socket
import struct
from scipy.stats import norm
from parameter_parser import parameter_parser

class PreprocessNetwork:
    def __init__(self):
        self.logger = logging.getLogger("preprocess a network dataset")

        self.shape = []
       
        for path in config_dpsyn.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Load the Field Types from JSON
        with open(config_dpsyn.TYPE_CONIFG_PATH, 'r') as file:
            field_config = json.load(file)
            self.field_types = field_config["field_types"]
            self.bin_sizes = field_config["bin_sizes"]
            self.field_key = field_config["flow_key"]

    def load_data(self, csv_filename):
        self.logger.info("loading data")
        with open(config_dpsyn.RAW_DATA_PATH + csv_filename, 'r') as file:
            self.df = pd.read_csv(config_dpsyn.RAW_DATA_PATH + csv_filename, low_memory=False)
            for column in self.df.columns:
                if self.field_types[column] == 'timestamp':
                    self.df.sort_values(by=column, inplace=True, ascending=True)
                    break
            self.set_data_types()
        
    def set_data_types(self):
        # deal with large timestamp, pkt and byt
        for column in self.df.columns:
            if column in self.field_types and self.field_types[column] in ['timestamp', 'binned_integer', 'int-exponential']:
                self.df[column] = self.df[column].astype('uint64')


    # Function to bin IP addresses by subnet
    def bin_ip(self, ip_series, subnet_size):
        factor = 32 - subnet_size
        return ip_series.apply(lambda ip: int(ip) >> factor)

    # Exponential binning function
    def bin_exponential(self, value, base):
        if value <= 0:
            return 0
        return int(np.ceil(np.log(value) / np.log(base)))

    def print_bin_stats(self, column):
        self.logger.info(f"Encoded Column: {column}")
        self.logger.info(f"Number of Bins: {self.shape[-1]}")
        self.logger.info(f"Min Bin Value: {self.df[column].min()}")
        self.logger.info(f"Max Bin Value: {self.df[column].max()}")
        self.logger.info(f"Average Bin Value: {self.df[column].mean()}")

    def build_mapping(self):        
        self.logger.info("build mapping")
        # Encode Categorical Fields and Handle Timestamp
        self.mappings = {}  # To store mappings of categorical fields
        self.shape = []
        timestamp_diff = None

        for column in self.df.columns:
            if self.field_types[column] == 'categorical':
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                self.mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
                self.shape.append(len(le.classes_))
            elif self.field_types[column] == 'binned-ip':
                binned_values= self.bin_ip(self.df[column], self.bin_sizes[column])
                le = LabelEncoder()
                self.df[column] = le.fit_transform(binned_values)
                self.mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
                self.shape.append(len(le.classes_))
            elif self.field_types[column] == 'binned-port':
                threshold = self.bin_sizes[column]
                bin_size = self.bin_sizes["port_bin_size"]
                binned_values= self.df[column].apply(lambda x: x if x < threshold else threshold + ((x - threshold) // bin_size))
                le = LabelEncoder()
                self.df[column] = le.fit_transform(binned_values)
                self.mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
                self.shape.append(len(le.classes_))
            elif self.field_types[column] == 'binned_integer':
                bin_size = self.bin_sizes.get(column, 1)
                bins = np.arange(0, self.df[column].max() + bin_size, bin_size)
                self.df[column] = np.digitize(self.df[column], bins, right=False)
                self.shape.append(len(bins))
            elif self.field_types[column] == 'timestamp':
                initial_timestamp = self.df[column].min()
                self.mappings['initial_timestamp'] = initial_timestamp
                self.df[column] -= initial_timestamp
                bin_size = self.bin_sizes.get(column, 1)
                bins = np.arange(0, self.df[column].max() + bin_size, bin_size)
                binned_timestamps = np.digitize(self.df[column], bins, right=False)
           
                bin_starts = ((binned_timestamps - 1) * bin_size).astype('uint64')
                # Compute the difference
                timestamp_diff = self.df[column] - bin_starts
                self.df[column] = binned_timestamps
                self.shape.append(len(bins))

            elif self.field_types[column] in ['float-exponential', 'int-exponential']:
                # First, apply exponential binning
                base = self.bin_sizes[column]
                encoded_values = self.df[column].apply(lambda x: self.bin_exponential(x, base))
                # Find the minimum encoded value
                min_encoded_val = encoded_values.min()
                self.mappings[column + "_min_encoded_val"] = min_encoded_val
                # Shift the encoded values to ensure they are >= 0
                self.df[column] = encoded_values - min_encoded_val
                self.shape.append(self.df[column].max() + 1)
            self.print_bin_stats(column)

        # Bin the difference
        if timestamp_diff is not None:
            # ts_diff is set to 1/10 of the ts bin window
            ts_diff_size = self.bin_sizes.get("ts_diff", 1)
            ts_diff_bins = np.arange(0, timestamp_diff.max() + ts_diff_size, ts_diff_size)
            self.df['ts_diff'] = np.digitize(timestamp_diff, ts_diff_bins, right=False)
            self.shape.append(len(ts_diff_bins))
            self.print_bin_stats('ts_diff')

    def save_data(self, pickle_filename, mapping_filename):
        self.logger.info("saving data")

        domain = Domain(self.df.columns, self.shape)
        dataset = Dataset(self.df, domain)

        #Save the Encoded Dataset to a Pickle File
        with open(config_dpsyn.PROCESSED_DATA_PATH + pickle_filename, 'wb') as file:
            pickle.dump(dataset, open(config_dpsyn.PROCESSED_DATA_PATH + pickle_filename, 'wb'))
        #Save the Mappings to another Pickle File
        with open(config_dpsyn.PROCESSED_DATA_PATH + mapping_filename, 'wb') as file:
            pickle.dump(self.mappings, file)

    def reverse_mapping(self):
        self.logger.info("reverse mapping")
        for column, mapping in self.mappings.items():
            if column in self.field_types and self.field_types[column] in ['binned-ip', 'binned-port', 'categorical']:
                inv_map = {v: k for k, v in mapping.items()}
                self.df[column] = self.df[column].map(inv_map)

        # Reversing the binning for IP addresses
        for column in self.df.columns:
            if column not in self.field_types:
                #for added column like ts_diff
                continue

            if self.field_types[column] == 'binned-ip':
                subnet_size = self.bin_sizes[column]
                factor = 32 - subnet_size
                self.df[column] = self.df[column].apply(
                    lambda x: (int(x) << factor) + random.randint(0, (1 << factor) - 1))

            # Reversing the binning for Ports
            elif self.field_types[column] == 'binned-port':
                threshold = self.bin_sizes[column]
                bin_size = self.bin_sizes["port_bin_size"]

                if column == 'srcport':

                    self.df[column] = self.df[column].apply(
                        lambda x: x if x < threshold else (threshold + (x - threshold) * bin_size) + random.randint(0,bin_size - 1))
                    max_binned_value = threshold + ((65535 - threshold) // bin_size)
                    max_decode_value = threshold + (max_binned_value - threshold) * bin_size
                    self.df[column] = self.df[column].apply(
                        lambda x: x if x <= 65535 else (random.randint(max_decode_value ,65535))
                    )

                if column == 'dstport':
                    self.df[column] = self.df[column].apply(
                        lambda x: x if x < threshold else (threshold + (x - threshold) * bin_size) + random.randint(
                            0, bin_size - 1))
                    max_binned_value = threshold + ((65535 - threshold) // bin_size)
                    max_decode_value = threshold + (max_binned_value - threshold) * bin_size
                    self.df[column] = self.df[column].apply(
                        lambda x: x if x <= 65535 else (random.randint(max_decode_value, 65535))
                    )

            # Reversing the binning for integer
            elif self.field_types[column] == 'binned_integer': 
                bin_size = self.bin_sizes[column]
                # Randomly sample within each bin for other binned_integer fields
                self.df[column] = self.df[column].apply(lambda x: (x - 1) * bin_size + random.randint(0, bin_size - 1))

            # Reversing the binning for timestamp
            elif self.field_types[column] == 'timestamp':
                bin_size = self.bin_sizes[column]
                ts_diff_bin_size = self.bin_sizes.get("ts_diff", 1)
                # For timestamp fields, reconstruct timestamps from bins
                initial_timestamp = self.mappings.get('initial_timestamp', 0)
                # Calculate the bin start for each timestamp
                bin_starts = ((self.df[column] - 1) * bin_size).astype('uint64')
                mean = ts_diff_bin_size / 2
                std_dev = ts_diff_bin_size / self.bin_sizes['ts_diff_std']   
                gaussian_diff_within_bin = np.random.normal(mean, std_dev, size=len(self.df))
                # Clip the values to ensure they fall within the bin and round to nearest integer
                gaussian_diff_within_bin_clipped = np.clip(gaussian_diff_within_bin, 0,
                                                           ts_diff_bin_size - 1).round().astype('uint64')
                self.df[column] = initial_timestamp + bin_starts + gaussian_diff_within_bin_clipped

            # Reversing the exponential binning
            elif self.field_types[column] in ['float-exponential', 'int-exponential']:
                min_encoded_val = self.mappings.get(column + "_min_encoded_val", 0)
                base = self.bin_sizes[column]

                if self.field_types[column] == 'float-exponential':
                    # Sample a floating point value for float-exponential fields
                    self.df[column] = (self.df[column] + min_encoded_val).apply(
                        lambda x: (base ** x) + random.uniform(0, (base ** (x + 1)) - (base ** x)))
                else:
                    # Sample an integer value for int-exponential fields
                    self.df[column] = (self.df[column] + min_encoded_val).apply(
                        lambda x: int(np.power(base, x)) + random.randint(0, int(np.power(base, x + 1)) - int(np.power(base, x)) - 1))

        #remove ts_diff after decoding
        self.df.drop('ts_diff', axis=1, inplace=True)
        #We don't syntheszing floating point
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(0, inplace=True)

    def reverse_mapping_from_files(self, pickle_filename, mapping_filename):
        with open(config_dpsyn.SYNTHESIZED_RECORDS_PATH + pickle_filename, 'rb') as file:
            ds = pickle.load(file)
            self.df = ds.df
        with open(config_dpsyn.PROCESSED_DATA_PATH + mapping_filename, 'rb') as file:
            self.mappings = pickle.load(file)
        '''
        if gaussian_filename is not None:
            with open(config.PROCESSED_DATA_PATH + gaussian_filename, 'rb') as file:
                self.gaussian_params = pickle.load(file) 
        '''           
        self.set_data_types()
        self.reverse_mapping()
          
    def save_data_csv(self, csv_filename):
        self.logger.info("save df to csv file")
        # Save the Decoded Dataset to a CSV File
        with open(config_dpsyn.SYNTHESIZED_RECORDS_PATH + csv_filename, 'wb') as file:
            self.df.to_csv(config_dpsyn.SYNTHESIZED_RECORDS_PATH + csv_filename, index=False)

def main(args):
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)


    preprocess = PreprocessNetwork()
    file_prefix = args['dataset_name']
    preprocess.load_data(file_prefix + '.csv')
    preprocess.build_mapping()
    preprocess.save_data(file_prefix, file_prefix + '_mapping')
    preprocess.reverse_mapping()
    preprocess.save_data_csv(file_prefix + '_syn_trivial.csv')

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
