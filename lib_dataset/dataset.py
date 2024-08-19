import numpy as np
import pandas as pd
import json
from lib_dataset.domain import Domain


class Dataset:
    def __init__(self, df, domain):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        """

        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        self.domain = domain
        self.df = df.loc[:, domain.attrs]

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)

        return Dataset(df, domain)

    @staticmethod
    def load(path, domain):
        """ Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())

        return Dataset(df, domain)

    def change_column(self, attr_name, new_records, new_shape):
        self.df.loc[:, attr_name] = new_records
        self.domain.change_shape(attr_name, new_shape)

    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]

        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)

        return Dataset(data, domain)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]

        return self.project(proj)

    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n + 1) for n in self.domain.shape]
        try:
            ans = np.histogramdd(self.df.values, bins)[0]
        except:
            a = 1

        return ans.flatten() if flatten else ans

    def to_csv(self, save_path, index=False):
        pass
