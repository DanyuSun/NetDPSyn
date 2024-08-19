import os
import logging
import pickle
import json
import ssl
import zipfile
from six.moves import urllib
import os.path as osp

import numpy as np
import pandas as pd

import config
from lib_dataset.dataset import Dataset
from lib_dataset.domain import Domain


class PreprocessColorado:
    def __init__(self):
        self.logger = logging.getLogger("preprocess colorado")

        self.shape = []
        
        for path in config.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)
        
    def load_data(self):
        self.logger.info("loading data")

        self._download_url('https://drive.google.com/u/0/uc?export=download&id=1pFRjwGoe1nh5yI4Wf6-v-Ana02RSUJ5o',
                           config.RAW_DATA_PATH + 'colorado.zip')
        self._download_url('https://drive.google.com/uc?export=download&id=1clFGr8rXyHLgSnZ4srt4L32V9Xwg5wKp',
                           config.RAW_DATA_PATH + 'colorado-specs.json')
        self._download_url('https://drive.google.com/uc?export=download&id=1lIsaHlvp7COX0cH6beymzcYE31c-9pSz',
                           config.RAW_DATA_PATH + 'colorado-code-mapping.pkl')

        self.df = pd.read_csv(config.RAW_DATA_PATH + "colorado.csv", low_memory=False)
        self.specs = json.load(open(config.RAW_DATA_PATH + "colorado-specs.json", 'r'))
        self.code_mapping = pickle.load(open(config.RAW_DATA_PATH + "colorado-code-mapping.pkl", "rb"))

    def calculate_num_categories(self):
        self.logger.info("calculating num_categories")

        for index, col in enumerate(self.df.columns):
            maxval = self.specs[col]["maxval"]

            if col in self.code_mapping:
                maxval_index = np.where(self.code_mapping[col] == maxval)[0]

                if maxval_index.size == 0:
                    self.shape.append(maxval + 1)
                    self.code_mapping[col] = np.arange(maxval + 1, dtype=np.uint32)
                else:
                    self.shape.append(int(maxval_index[0] + 1))
            else:
                self.shape.append(maxval + 1)

        self.logger.info("calculated num_categories")

    def transform_to_num(self):
        for index, col in enumerate(self.df.columns):
            self.logger.info("transforming attribute %s: %s" % (col, index))

            # if attribute_name in self.code_mapping and attribute_name not in self.inconsistent_attributes:
            if col in self.code_mapping:
                records = np.copy(self.df[col])
                unique_value, count = np.unique(records, return_counts=True)

                for value in unique_value:
                    code_mapping_index = np.where(self.code_mapping[col] == value)[0]
                    self.df.loc[records == value, col] = code_mapping_index

    def save_data(self):
        self.logger.info("saving data")

        domain = Domain(self.df.columns, self.shape)
        dataset = Dataset(self.df, domain)

        dataset.df.INCWAGE = np.digitize(dataset.df.INCWAGE, np.array([5e4]))
        dataset.domain.change_shape("INCWAGE", 2)
        # dataset = dataset.drop(("INCWAGE",))
        dataset = dataset.drop(("VALUEH",))

        pickle.dump(dataset, open(config.PROCESSED_DATA_PATH + "colorado", 'wb'))

        self.logger.info("saved data")
        
    def _download_url(self, url, path):
        if osp.exists(path):
            return
        
        context = ssl._create_unverified_context()
        data = urllib.request.urlopen(url, context=context)
    
        with open(path, 'wb') as f:
            f.write(data.read())
            
        if path.split('.')[1] == 'zip':
            with zipfile.ZipFile(path, 'r') as f:
                for file in f.namelist():
                    f.extract(file, config.RAW_DATA_PATH)


def main():
    os.chdir("../")

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    preprocess = PreprocessColorado()
    preprocess.load_data()
    preprocess.calculate_num_categories()
    preprocess.transform_to_num()
    preprocess.save_data()


if __name__ == "__main__":
    main()
