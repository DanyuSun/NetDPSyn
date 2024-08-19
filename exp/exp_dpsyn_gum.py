import datetime
import logging
import math
import time

import numpy as np
import pandas as pd

from exp.exp_dpsyn import ExpDPSyn
from lib_dpsyn.sep_graph import SepGraph
from lib_dpsyn.update_config import UpdateConfig
from lib_dataset.dataset import Dataset
from lib_dpsyn.attr_append import AttrAppend
import config_dpsyn
import ast
from scipy.spatial.distance import jensenshannon

class ExpDPSynGUM(ExpDPSyn):
    def __init__(self, args):
        super(ExpDPSynGUM, self).__init__(args)
        
        self.logger = logging.getLogger('exp_dpsyn_gum')

        self.selected_attrs = None
        if self.args['dump_marginal'] != 'None':
            marginal_name = self.args['dump_marginal']
            self.selected_attrs = tuple(marginal_name.split("_"))

        ################################## main procedure ##########################################
        ################################## main procedure ##########################################
        self.preprocessing()
        self.construct_views()
        for key, view in self.views_dict.items():
            print(key)

        for key, view in self.views_dict.items():
           self.data_store.save_one_marginal("construct", view, self.attr_recode)
        #ZL: dump the marginal for each phase
        self.anonymize_views()
        for key, view in self.views_dict.items():
            self.data_store.save_one_marginal("anonymize", view, self.attr_recode)
        self.consist_views(self.attr_recode.dataset_recode.domain, self.views_dict)
        for key, view in self.views_dict.items():
            self.data_store.save_one_marginal("consist", view, self.attr_recode)
        self.pred_attr = args["pred_attr"]
        print('----start gum -----')
        print(time.time())
        self.synthesize_records(args)
        self.data_store.save_one_marginal("synthesize", self.synthesized_df, self.attr_recode)

        self.postprocessing()
        #print('len(self.synthesized_df) after', len(self.synthesized_df))
        #postprocessing decoded the grouped values already
        self.data_store.save_one_marginal("postprocessing", self.synthesized_df)

    def preprocessing(self):
        #ZL: Section 5.3 Separate & Join. If graph is separated, GUM is done on several lists of marginals independently
        #ZL: for Attribute Append, it just concatenate the marginal with degree=1 to the dataset after GUM
        self.sep_graph = SepGraph(self.original_dataset.domain, self.marginals)
        self.sep_graph.cut_graph()

        self.attr_append = AttrAppend(self.attr_recode.dataset.domain, self.marginals)

        iterate_marginals = self.attr_append.clip_graph(enable=self.args['append'])
        #ZL: debugging
        self.logger.info("iterate_marginals after clip_graph is %s" % (iterate_marginals,))

        self.iterate_keys = self.sep_graph.find_sep_graph(iterate_marginals, enable=self.args['sep_syn'])

    def construct_views(self):
        self.logger.info("constructing views")

        for i, marginal in enumerate(self.marginals):
            self.logger.debug('%s th marginal' % (i,))
            self.views_dict[marginal] = self.construct_view(self.attr_recode.dataset_recode, marginal)

        # this part can be obtained directly from attrs recode part
        for singleton in self.original_dataset.domain.attrs:
            self.views_dict[(singleton,)] = self.construct_view(self.attr_recode.dataset_recode, (singleton,))
            self.singleton_key.append((singleton,))

    def anonymize_views(self):
        #ZL: add noise
        self.logger.info("anonymizing views")

        divider = 0.0

        for key, view in self.views_dict.items():
            divider += math.sqrt(view.num_key)

        for key, view in self.views_dict.items():
            if self.args['noise_add_method'] == "A1":
                view.rho = 1.0
                self.anonymize_view(view, epsilon=self.remain_epsilon / len(self.views_dict))
            elif self.args['noise_add_method'] == "A2":
                view.rho = self.remain_rho / len(self.views_dict)
                self.anonymize_view(view, rho=view.rho)
            elif self.args['noise_add_method'] == "A3":
                view.rho = self.remain_rho * math.sqrt(view.num_key) / divider
                self.anonymize_view(view, rho=view.rho)                
            else:
                raise Exception("invalid noise adding method")
        # for key, view in self.views_dict.items():
        #     print('key', key)
        #     print('view', len(view.count))
        #     print(view.attributes_index)
        #     print(view.num_key)
        #     print(view.num_categories)

    def synthesize_records(self, args):
        #ZL: GUM
        self.synthesized_df = pd.DataFrame(data=np.zeros([self.args['num_synthesize_records'], self.num_attributes], dtype=np.uint32),
                                           columns=self.original_dataset.domain.attrs)
        self.error_tracker = pd.DataFrame()

        # main procedure for synthesizing records
        for key, value in self.iterate_keys.items():           

            self.logger.info("synthesizing for %s" % (key,))

            #ZL: bootstrap dataframe with marginals, manual or auto
            self.marginal_init = None
            if args["initialize_method"] == "marginal_manual":
                self.marginal_init = ast.literal_eval(config_dpsyn.MARGINAL_INIT)
            elif args["initialize_method"] == "marginal_auto":
                #ZL: infer the bootstrapping marginals from data
                self.marginal_init = self.find_init_marginals(value)
                #ZL: order self.marginal_init by in_dif, so highly correlated marginals are less likely to be changed
                self.marginal_init = self.rank_marginals(self.marginal_init)

            synthesizer = self._update_records(value)
            self.synthesized_df.loc[:, key] = synthesizer.update.df.loc[:, key]

            #ZL: error because of old append is deprecated
            #self.error_tracker = self.error_tracker.append(synthesizer.update.error_tracker)
            self.error_tracker = pd.concat([self.error_tracker, synthesizer.update.error_tracker])

    def find_init_marginals(self, marginal_tuples):
        # Filter tuples that contain the key attribute
        key_tuples = [t for t in marginal_tuples if self.pred_attr in t]

        # Collect all unique attributes from tuples
        all_attrs = set(attr for tuple in marginal_tuples for attr in tuple)

        # Set to keep track of covered attributes
        covered_attrs = set([self.pred_attr])

        # Final set of tuples
        minimal_tuples = []

        while covered_attrs != all_attrs:
            # Find the tuple that adds the most new attributes
            max_new_attrs = 0
            best_tuple = None
            for t in key_tuples:
                new_attrs = len(set(t) - covered_attrs)
                if new_attrs > max_new_attrs:
                    max_new_attrs = new_attrs
                    best_tuple = t

            # Add the best tuple to the result and update covered attributes
            if best_tuple:
                minimal_tuples.append(best_tuple)
                covered_attrs.update(best_tuple)
                key_tuples.remove(best_tuple)
            else:
                # Break if no more tuples can add new attributes
                break

        return minimal_tuples

    def rank_marginals(self, marginal_tuples):
        def expand_marginal_df(marginal_df):
            #Expands the marginal DataFrame based on the count of each row.
            marginal_df_expanded = marginal_df.copy()
            rounded_counts = marginal_df_expanded['count'].round().astype(int)
            return marginal_df_expanded.loc[marginal_df_expanded.index.repeat(rounded_counts)].drop('count', axis=1)

        def calculate_correlation_with_type(df, pred_attr):
            #Calculates the correlation with the pred_attr attribute.
            correlation_matrix = df.corr()
            correlations = correlation_matrix[pred_attr].drop(pred_attr, errors='ignore')
            return correlations.mean()  # or max(type_correlations)
        
        #Ranks the marginals first by the number of attributes, then by correlation with pred_attr.
        marginal_info = []
        marginal_dfs = {key: self.views_dict[key].decode_records() for key in marginal_tuples}
        for key, df in marginal_dfs.items():
            expanded_df = expand_marginal_df(df)
            #ZL TBD: could think about another metric
            correlation = calculate_correlation_with_type(expanded_df, self.pred_attr)
            num_attributes = len(df.columns) - 1  # excluding count column
            marginal_info.append((key, num_attributes, correlation))

        # Sort by number of attributes (ascending), then by correlation (descending)
        ranked_marginals = sorted(marginal_info, key=lambda x: (x[1], -x[2]))

        return [key for key, _, _ in ranked_marginals]

    def postprocessing(self):
        self.logger.info("postprocessing dataset")

        # decode records
        self.sep_graph.join_records(self.synthesized_df)
        self.attr_append.append_attrs(self.synthesized_df, self.views_dict)
        self.attr_recode.decode(self.synthesized_df)

        self.synthesized_dataset = Dataset(self.synthesized_df, self.original_dataset.domain)
        self.end_time = datetime.datetime.now()

        self.data_store.save_synthesized_records(self.synthesized_dataset)

    def get_df_marginal(self, df, selected_attrs, attr_recode):
        marginal = df[selected_attrs].value_counts().reset_index()
        marginal.columns = selected_attrs + ['count']
        if attr_recode is not None:
            marginal =attr_recode.decode_marginal(marginal, selected_attrs)
        return marginal

    def diff_marginal(self, marginal1, marginal2):
        def normalize_counts(df):
            total = df['count'].sum()
            df['probability'] = df['count'] / total
            return df

        # Normalize the counts for both dataframes
        df1_normalized = normalize_counts(marginal1)
        df2_normalized = normalize_counts(marginal2)

        # Align the dataframes on 'dstport' and 'type', filling missing values with zeros
        merged_df = df1_normalized.merge(df2_normalized, on=list(self.selected_attrs), how='outer', suffixes=('_1', '_2'))
        merged_df.fillna(0, inplace=True)

        # Compute the Jensen-Shannon Divergence
        js_divergence = jensenshannon(merged_df['probability_1'], merged_df['probability_2'])
        return js_divergence

    def _update_records(self, views_iterate_key):
        update_config = {
            "alpha": self.args['update_rate_initial'],
            "alpha_update_method": self.args['update_rate_method'],
            "update_method": self.args['update_method'],
            "threshold": 0.0
        }

        singletons = {singleton: self.views_dict[(singleton,)] for singleton in self.original_dataset.domain.attrs}

        synthesizer = UpdateConfig(self.attr_recode.dataset_recode.domain, self.args['num_synthesize_records'], update_config)
        if self.marginal_init is not None:
            marginal_views = [self.views_dict[key] for key in self.marginal_init]
            synthesizer.update.initialize_records(views_iterate_key, method=self.args['initialize_method'], singletons=singletons, marginals=marginal_views, pred_attr=self.pred_attr)
        else:
            synthesizer.update.initialize_records(views_iterate_key, method=self.args['initialize_method'], singletons=singletons)

        print('self.selected_attrs', self.selected_attrs)
        jsd_value = []
        if self.selected_attrs is not None:
            marginal_consisted = self.views_dict[self.selected_attrs].decode_records(self.attr_recode) 
            marginal_before_update = self.get_df_marginal(synthesizer.update.df, list(self.selected_attrs), self.attr_recode)
            self.logger.info("synthesize %s, JSD after init, before update: %f" % (str(self.selected_attrs), self.diff_marginal(marginal_consisted, marginal_before_update)))
            jsd_value.append(self.diff_marginal(marginal_consisted, marginal_before_update))

        print('self.args[update_iterations]', self.args['update_iterations'])
        for update_iteration in range(self.args['update_iterations']):
            self.logger.info("update round: %d" % (update_iteration,))

            synthesizer.update_alpha(update_iteration)
            views_iterate_key = synthesizer.update_order(update_iteration, self.views_dict, views_iterate_key)


            for index, key in enumerate(views_iterate_key):
                #ZL: commented out, too many log messages
                #self.logger.info("updating %s view: %s, num_key: %s" % (index, key, self.views_dict[key].num_key))
                 # ZL: selecting the minimum set of marginals without breaking connectivity
                #if self.marginal_filter is not None and key not in self.marginal_filter:
                #    continue
                # ZL: measure the changes of interesting marginal to the original marginal
                synthesizer.update_records(self.views_dict[key], key, update_iteration)
                # if self.selected_attrs is not None:
                #     marginal_after_update = self.get_df_marginal(synthesizer.update.df, list(self.selected_attrs), self.attr_recode)
                #     self.logger.info("synthesize %s (iteration %d), JSD after update with key %s: %f" % (str(self.selected_attrs), update_iteration, str(key), self.diff_marginal(marginal_consisted, marginal_after_update)))
                #

            if self.selected_attrs is not None:
                marginal_after_update = self.get_df_marginal(synthesizer.update.df, list(self.selected_attrs),self.attr_recode)
                self.logger.info("synthesize %s (iteration %d), JSD after update: %f" % (str(self.selected_attrs), update_iteration, self.diff_marginal(marginal_consisted, marginal_after_update)))
                jsd_value.append(self.diff_marginal(marginal_consisted, marginal_after_update))

            #ZL: save the result of each iteration, problem starts from the first iteration
            # self.data_store.save_one_marginal("synthesize-iter-"+str(update_iteration), synthesizer.update.df, self.attr_recode)

        print('jsd_value')
        print(jsd_value)
        return synthesizer
