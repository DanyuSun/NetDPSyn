import logging
import pickle
import copy
import math

import numpy as np
import pandas as pd

from lib_view.view import View
from lib_composition.advanced_composition import AdvancedComposition
import config_dpsyn


class AttrDepend:
    def __init__(self, dataset, dataset_name, args):
        self.logger = logging.getLogger("determine_marginal")

        self.df = copy.deepcopy(dataset.df)
        self.dataset_domain = copy.deepcopy(dataset.domain)
        self.original_domain = copy.deepcopy(dataset.domain)
        self.dataset_name = dataset_name
        self.args = args

    def transform_records_distinct_value(self):
        self.logger.info("transforming records")

        distinct_shape = []

        for attr_index, attr in enumerate(self.dataset_domain.attrs):
            record = np.copy(self.df.loc[:, attr])
            unique_value = np.unique(record)
            distinct_shape.append(unique_value.size)

            for index, value in enumerate(unique_value):
                indices = np.where(record == value)[0]
                # self.df.loc[indices, attr] = index
                self.df.values[indices, attr_index] = index

        self.dataset_domain.shape = tuple(distinct_shape)
        self.dataset_domain.config = dict(zip(self.dataset_domain.attrs, distinct_shape))

        self.logger.info("transformed records")

    def calculate_pair_indif(self):
        self.logger.info("calculating pair indif")

        dependency_df = pd.DataFrame(columns=["first_attr", "second_attr", "num_cells", "error"])
        dependency_index = 0

        for first_index, first_attr in enumerate(self.dataset_domain.attrs[:-1]):
            first_view = View(self.dataset_domain.project(first_attr), self.dataset_domain)
            first_view.count_records(self.df.values)
            first_histogram = first_view.calculate_normalize_count()

            for second_attr in self.dataset_domain.attrs[first_index + 1:]:
                self.logger.info("calculating [%s, %s]" % (first_attr, second_attr))                

                second_view = View(self.dataset_domain.project(second_attr), self.dataset_domain)
                second_view.count_records(self.df.values)
                second_histogram = second_view.calculate_normalize_count()

                # calculate real 2-way marginal
                pair_view = View(self.dataset_domain.project((first_attr, second_attr)), self.dataset_domain)
                pair_view.count_records(self.df.values)
                pair_view.calculate_count_matrix()
                # pair_distribution = pair_view.calculate_normalize_count()

                # calculate 2-way marginal assuming independent
                independent_pair_distribution = np.outer(first_histogram, second_histogram)

                # calculate the errors
                normalize_pair_view_count = pair_view.count_matrix / np.sum(pair_view.count_matrix)
                error = np.sum(np.absolute(normalize_pair_view_count - independent_pair_distribution))

                num_cells = self.original_domain.config[first_attr] * self.original_domain.config[second_attr]
                dependency_df.loc[dependency_index] = [first_attr, second_attr, num_cells, error]

                dependency_index += 1

        pickle.dump(dependency_df, open(config_dpsyn.DEPENDENCY_PATH + self.dataset_name, "wb"))

        self.dependency_df = dependency_df
        self.logger.info("calculated pair dependency")

    def solve_score_function(self, dataset_name, epsilon, rho, marg_add_sensitivity, marg_select_sensitivity, noise_add_method):
        self.logger.info("choosing marginals")

        self.dependency_df = pickle.load(open(config_dpsyn.DEPENDENCY_PATH + self.dataset_name, "rb"))

        if epsilon != 0.0:
            composition = AdvancedComposition()
            sigma = composition.gauss_zcdp(epsilon, 1.0 / (self.df.shape[0]), marg_select_sensitivity, self.dependency_df.shape[0])
            self.dependency_df.error += np.random.normal(scale=sigma / (2.0 * self.df.shape[0]), size=self.dependency_df.shape[0])

        gap = 1e10

        self.marginals = []
        self.selected_attrs = set()

        error = self.dependency_df["error"].to_numpy() * self.df.shape[0]
        num_cells = self.dependency_df["num_cells"].to_numpy().astype(np.float64)
        overall_error = np.sum(error)
        selected = set()
        unselected = set(self.dependency_df.index)

        gauss_error_normalizer = 1.0

        threshold = self.args['threshold']

        while gap > threshold:
            error_new = np.sum(error)
            selected_index = None

            for j in unselected:
                select_candidate = selected.union({j})

                if noise_add_method == "A3":
                    cells_square_sum = np.sum(np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
                    gauss_constant = np.sqrt(cells_square_sum / (math.pi * rho))
                    gauss_error = np.sum(gauss_constant * np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
                elif noise_add_method == "A1" or noise_add_method == "A2":
                    gauss_constant = np.sqrt((marg_add_sensitivity ** 2 * len(select_candidate)) / (2.0 * rho))
                    gauss_error = np.sum(gauss_constant * num_cells[list(select_candidate)])
                else:
                    raise Exception("invalid noise add method")

                gauss_error *= gauss_error_normalizer

                pairwise_error = np.sum(error[list(unselected.difference(select_candidate))])
                error_temp = gauss_error + pairwise_error

                if error_temp < error_new:
                    selected_index = j
                    error_new = error_temp

            gap = overall_error - error_new
            overall_error = error_new
            selected.add(selected_index)
            unselected.remove(selected_index)

            first_attr, second_attr = self.dependency_df.loc[selected_index, "first_attr"], self.dependency_df.loc[
                selected_index, "second_attr"]
            self.marginals.append((first_attr, second_attr))
            self.selected_attrs.update((first_attr, second_attr))

            self.logger.info("select %s marginal: %s | gap: %s" % (len(self.marginals), (first_attr, second_attr), gap))

    def handle_isolated_attrs(self, method="isolate", sort=False):
        # find attrs that does not appear in any of the pairwise marginals
        missing_attrs = set(self.dataset_domain.attrs) - self.selected_attrs

        if sort:
            # self.dependency_df["error"] /= np.sqrt(self.dependency_df["num_cells"].astype("float"))
            self.dependency_df.sort_values(by="error", ascending=False, inplace=True)
            self.dependency_df.reset_index(drop=True, inplace=True)

        for attr in missing_attrs:
            if method == "isolate":
                self.marginals.append((attr,))

            elif method == "connect":
                match_missing_df = self.dependency_df.loc[
                    (self.dependency_df["first_attr"] == attr) | (self.dependency_df["second_attr"] == attr)]
                match_df = match_missing_df.loc[(match_missing_df["first_attr"].isin(self.selected_attrs)) | (
                    match_missing_df["second_attr"].isin(self.selected_attrs))]
                match_df.reset_index(drop=True, inplace=True)
                self.marginals.append((match_df.loc[0, "first_attr"], match_df.loc[0, "second_attr"]))

        return self.marginals
