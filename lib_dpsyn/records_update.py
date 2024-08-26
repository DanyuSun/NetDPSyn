import logging

import numpy as np
import pandas as pd


class RecordUpdate:
    def __init__(self, domain, num_records):
        self.logger = logging.getLogger("record_update")
        
        self.domain = domain
        self.num_records = num_records


    def initialize_records(self, iterate_keys, method="random", singletons=None, marginals=None, pred_attr=None):
        self.logger.info("initializing synthesized dataframe")
        self.records = np.zeros([self.num_records, len(self.domain.attrs)], dtype=np.uint32)
        self.df = pd.DataFrame(self.records, columns=self.domain.attrs, copy=False)

        if method in ["random", "singleton"]:
            for attr in self.domain.attrs:
                if method == "random":
                    self.df[attr] = np.random.randint(0, self.domain.config[attr], size=self.num_records)

                elif method == "singleton":
                    self.df[attr] = self.generate_singleton_records(singletons[attr])

        elif method in ["marginal_manual", "marginal_auto"]:
            # initialization from selected marginals
            init_attrs = set()
            for index, marginal in enumerate(marginals):
                self.logger.info("initializing with marginal %s", (str(marginal.attr_names),))
                if index == 0:
                    # first marginal, self.df tries to be fully consistent with it
                    new_df = self.generate_marginal_records(marginal, pred_attr)
                else:
                    # other marginals, update the overlapping attributes 
                    new_df = self.update_marginal_records(marginal, init_attrs, pred_attr)
                for col in set(new_df.columns) - init_attrs:
                    self.df[col] = new_df[col]
                init_attrs.update(marginal.attr_names)
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            for attr in set(self.domain.attrs) - init_attrs:
                self.df[attr] = self.generate_singleton_records(singletons[attr])
        self.error_tracker = pd.DataFrame(index=iterate_keys)
        self.records = self.df.to_numpy()


    def generate_singleton_records(self, singleton):
        record = np.empty(self.num_records, dtype=np.uint32)
        dist_cumsum = np.cumsum(singleton.normalize_count)
        start = 0

        for index, value in enumerate(dist_cumsum):
            end = int(round(value * self.num_records))
            record[start: end] = index
            start = end

        np.random.shuffle(record)

        return record
    
    def generate_marginal_records(self, marginal, pred_attr):
        #update df by one marginal, marginal is a view object
        records = np.zeros([self.num_records, len(marginal.attr_names)], dtype=np.uint32)
        df_generated = pd.DataFrame(records, columns=marginal.attr_names)
        marginal_df = marginal.decode_records()
        #sort by pred_attr so different marginal syn records align
        marginal_df = marginal_df.sort_values(by=pred_attr, ascending=True)
        dist_cumsum = np.cumsum(marginal_df['count'] / marginal_df['count'].sum())
        #dist_cumsum = np.cumsum(marginal.normalize_count)

        start = 0      
        for index, row in marginal_df.iterrows():
            end = int(round(dist_cumsum[index] * self.num_records))
            # Fill the DataFrame rows with the attribute values
            df_generated.iloc[start:end] = row[:-1].values
            start = end

        # Shuffle the DataFrame rows
        #df_generated = df_generated.sample(frac=1).reset_index(drop=True)
        
        return df_generated


    def update_marginal_records(self, marginal, init_attrs, pred_attr):
        df_generated = self.df[marginal.attr_names].copy()
        marginal_df = marginal.decode_records()
        # Identify overlapping attributes (already synthesized) in the new marginal
        overlapping_attrs = [attr for attr in marginal.attr_names if attr in init_attrs]
        new_attrs = [attr for attr in marginal.attr_names if attr not in init_attrs]

        # Normalize counts to match the distribution of overlapping attributes
        if overlapping_attrs:
            # Calculate group counts and normalize
            original_ratio = self.df.groupby(overlapping_attrs).size()
            original_ratio /= original_ratio.sum()

            new_ratio = marginal_df.groupby(overlapping_attrs).sum()
            new_ratio = new_ratio["count"]
            new_ratio /= new_ratio.sum()

            original_ratio_dict = original_ratio.to_dict()
            new_ratio_dict = new_ratio.to_dict()

            # Calculate the ratio of new to original for each type
            ratio_changes = {k: original_ratio_dict[k] / new_ratio_dict[k] for k in new_ratio_dict}

            # Update the counts in the marginal_df
            marginal_df['adjusted_count'] = marginal_df.apply(
                lambda row: row['count'] * ratio_changes.get(tuple(row[attr] for attr in overlapping_attrs), 1), axis=1)

        # Update dist_cumsum after adjusting counts
        marginal_df = marginal_df.sort_values(by=pred_attr, ascending=True)
        dist_cumsum = np.cumsum(marginal_df['adjusted_count'] / marginal_df['adjusted_count'].sum())

        # Synthesize new attributes
        start = 0
        for index, row in marginal_df.iterrows():
            end = int(round(dist_cumsum[index] * self.num_records))
            # Only update new attributes in df_generated
            df_generated.loc[start:end, new_attrs] = row[new_attrs].values
            start = end

        # Shuffle the DataFrame rows
        #df_generated = df_generated.sample(frac=1).reset_index(drop=True)

        return df_generated


    def update_records_main(self, view, alpha):
        ##################################### deal with cells to be boosted #######################################
        # deal with the cell that synthesize_marginal != 0 and synthesize_marginal < actual_marginal
        self.cell_under_indices = \
            np.where((self.synthesize_marginal < self.actual_marginal) & (self.synthesize_marginal != 0))[0]
        
        ratio_add = np.minimum((self.actual_marginal[self.cell_under_indices] - self.synthesize_marginal[
            self.cell_under_indices]) / self.synthesize_marginal[self.cell_under_indices],
            np.full(self.cell_under_indices.shape[0], alpha))
        self.num_add = self.normal_rounding(ratio_add * self.synthesize_marginal[self.cell_under_indices] * self.num_records)
        
        # deal with the case synthesize_marginal == 0 and actual_marginal != 0
        self.cell_zero_indices = np.where((self.synthesize_marginal == 0) & (self.actual_marginal != 0))[0]
        self.num_add_zero = self.normal_rounding(alpha * self.actual_marginal[self.cell_zero_indices] * self.num_records)

        #################################### deal with cells to be reduced ########################################
        # determine the number of records to be removed
        self.cell_over_indices = np.where(self.synthesize_marginal > self.actual_marginal)[0]
        num_add_total = np.sum(self.num_add) + np.sum(self.num_add_zero)

        beta = self.find_optimal_beta(num_add_total, self.cell_over_indices)
        ratio_reduce = np.minimum(
            (self.synthesize_marginal[self.cell_over_indices] - self.actual_marginal[self.cell_over_indices])
            / self.synthesize_marginal[self.cell_over_indices], np.full(self.cell_over_indices.shape[0], beta))
        self.num_reduce = self.normal_rounding(
            ratio_reduce * self.synthesize_marginal[self.cell_over_indices] * self.num_records).astype(int)

        self.logger.debug("alpha: %s | beta: %s" % (alpha, beta))
        self.logger.debug("num_boost: %s | num_reduce: %s" % (num_add_total, np.sum(self.num_reduce)))

        # calculate some general variables
        self.encode_records = np.matmul(self.records[:, view.attributes_index], view.encode_num)
        self.encode_records_sort_index = np.argsort(self.encode_records)
        self.encode_records = self.encode_records[self.encode_records_sort_index]

    def determine_throw_indices(self):
        valid_indices = np.nonzero(self.num_reduce)[0]
        valid_cell_over_indices = self.cell_over_indices[valid_indices]
        valid_cell_num_reduce = self.num_reduce[valid_indices]
        valid_data_over_index_left = np.searchsorted(self.encode_records, valid_cell_over_indices, side="left")
        valid_data_over_index_right = np.searchsorted(self.encode_records, valid_cell_over_indices, side="right")
    
        valid_num_reduce = np.sum(valid_cell_num_reduce)
        self.records_throw_indices = np.zeros(valid_num_reduce, dtype=np.uint32)
        throw_pointer = 0
        
        for i, cell_index in enumerate(valid_cell_over_indices):
            match_records_indices = self.encode_records_sort_index[
                                    valid_data_over_index_left[i]: valid_data_over_index_right[i]]
            throw_indices = np.random.choice(match_records_indices, valid_cell_num_reduce[i], replace=False)
            
            self.records_throw_indices[throw_pointer: throw_pointer + throw_indices.size] = throw_indices
            throw_pointer += throw_indices.size
        
        np.random.shuffle(self.records_throw_indices)
    
    def complete_partial_ratio(self, view, complete_ratio):
        num_complete = np.rint(complete_ratio * self.num_add).astype(int)
        num_partial = np.rint((1 - complete_ratio) * self.num_add).astype(int)

        valid_indices = np.nonzero(num_complete + num_partial)
        num_complete = num_complete[valid_indices]
        num_partial = num_partial[valid_indices]

        valid_cell_under_indices = self.cell_under_indices[valid_indices]
        valid_data_under_index_left = np.searchsorted(self.encode_records, valid_cell_under_indices, side="left")
        valid_data_under_index_right = np.searchsorted(self.encode_records, valid_cell_under_indices, side="right")
        
        for valid_index, cell_index in enumerate(valid_cell_under_indices):
            match_records_indices = self.encode_records_sort_index[
                                    valid_data_under_index_left[valid_index]: valid_data_under_index_right[valid_index]]

            np.random.shuffle(match_records_indices)
            
            if self.records_throw_indices.shape[0] >= (num_complete[valid_index] + num_partial[valid_index]):
                # complete update code
                if num_complete[valid_index] != 0:
                    self.records[self.records_throw_indices[: num_complete[valid_index]]] = self.records[
                        match_records_indices[: num_complete[valid_index]]]
                
                # partial update code
                if num_partial[valid_index] != 0:
                    self.records[np.ix_(
                        self.records_throw_indices[num_complete[valid_index]: (num_complete[valid_index] + num_partial[valid_index])],
                        view.attributes_index)] = view.tuple_key[cell_index]
                
                # update records_throw_indices
                self.records_throw_indices = self.records_throw_indices[num_complete[valid_index] + num_partial[valid_index]:]
            
            else:
                self.records[self.records_throw_indices] = self.records[
                    match_records_indices[: self.records_throw_indices.size]]
        self.df = pd.DataFrame(self.records, columns=self.domain.attrs, copy=False)
   
    def handle_zero_cells(self, view):
        # overwrite / partial when synthesize_marginal == 0
        if self.cell_zero_indices.size != 0:
            for index, cell_index in enumerate(self.cell_zero_indices):
                num_partial = int(self.num_add_zero[index])
                
                if num_partial != 0:
                    for i in range(view.ways):
                        self.records[self.records_throw_indices[: num_partial], view.attributes_index[i]] = \
                            view.tuple_key[cell_index, i]
                
                self.records_throw_indices = self.records_throw_indices[num_partial:]
    
    def find_optimal_beta(self, num_add_total, cell_over_indices):
        actual_marginal_under = self.actual_marginal[cell_over_indices]
        synthesize_marginal_under = self.synthesize_marginal[cell_over_indices]
        
        lower_bound = 0.0
        upper_bound = 1.0
        beta = 0.0
        current_num = 0.0
        iteration = 0
        
        while abs(num_add_total - current_num) >= 1.0:
            beta = (upper_bound + lower_bound) / 2.0
            current_num = np.sum(
                np.minimum((synthesize_marginal_under - actual_marginal_under) / synthesize_marginal_under,
                np.full(cell_over_indices.shape[0], beta)) * synthesize_marginal_under * self.records.shape[0])
            
            if current_num < num_add_total:
                lower_bound = beta
            elif current_num > num_add_total:
                upper_bound = beta
            else:
                return beta

            iteration += 1
            if iteration > 50:
                self.logger.warning("cannot find the optimal beta")
                break
        
        return beta
    

    def update_records_before(self, view, view_key, iteration, mute=False):
        self.actual_marginal = view.normalize_count
        self.synthesize_marginal = self.synthesize_marginal_distribution(view)
        l1_error = self._l1_distance(self.actual_marginal, self.synthesize_marginal)
        self.error_tracker.loc[[view_key], "%s-before" % (iteration,)] = l1_error

    def update_records_after(self, view, view_key, iteration):
        self.synthesize_marginal = self.synthesize_marginal_distribution(view)
        l1_error = self._l1_distance(self.actual_marginal, self.synthesize_marginal)
        self.error_tracker.loc[[view_key], "%s-after" % (iteration,)] = l1_error
    
    def synthesize_marginal_distribution(self, view):
        count = view.count_records_general(self.records)
        return view.calculate_normalize_count_general(count)

    def stochastic_rounding(self, vector):
        ret_vector = np.zeros(vector.size)
        rand = np.random.rand(vector.size)

        integer = np.floor(vector)
        decimal = vector - integer

        ret_vector[rand > decimal] = np.floor(decimal[rand > decimal])
        ret_vector[rand < decimal] = np.ceil(decimal[rand < decimal])
        ret_vector += integer

        return ret_vector
    
    def normal_rounding(self, vector):
        return np.round(vector)

    def _l1_distance(self, t1, t2):
        assert len(t1) == len(t2)
    
        return np.sum(np.absolute(t1 - t2))
