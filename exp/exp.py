import copy
import logging
import datetime

from lib_attrs.attr_recode import AttrRecord
from lib_view.marg_select import MarginalSelection
from lib_dataset.data_store import DataStore
from lib_composition.advanced_composition import AdvancedComposition


class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger("exp")
        
        self.start_time = datetime.datetime.now()
        self.args = args
        
        self.epsilon = self.args['epsilon']
        self.dataset_name = args["dataset_name"]

        ########################################### main proceedure ###########################################
        # load dataset
        self.data_store = DataStore(self.args)
        self.load_data()
        self.logger.info("original dataset domain: %e" % (self.original_dataset.domain.size(),))
        
        self.data_store.save_one_marginal("original", self.original_dataset.df)

        self.num_records = self.original_dataset.df.shape[0]
        self.num_attributes = self.original_dataset.df.shape[1]
        self.delta = 1.0 / self.num_records ** 2

        self.privacy_budget_allocation()
        self.remain_rho = self._calculate_rho(self.remain_epsilon)
        self.marginals = self.select_marginals(self.original_dataset)
        self.gauss_sigma = self._calculate_sigma(self.remain_epsilon, len(self.marginals) + self.num_attributes)
        # recode groups low-count values 
        self.attr_recode = self.recode_attrs(self.gauss_sigma)
        
        self.logger.info('rho: %s | sigma: %s' % (self.remain_rho, self.gauss_sigma))

    ############################## preprocess ###########################################
    def load_data(self):
        self.logger.info("loading dataset %s" % (self.dataset_name,))
        self.original_dataset = self.data_store.load_processed_data()

    def privacy_budget_allocation(self):
        self.depend_epsilon = self.epsilon * self.args['depend_epsilon_ratio']
        self.remain_epsilon = self.epsilon - self.depend_epsilon

        self.logger.info('privacy budget allocation: marginal %s | synthesize %s' % (self.depend_epsilon, self.remain_epsilon))

    def select_marginals(self, dataset):
        if self.args['is_cal_marginals']:
            self.logger.info("selecting marginals")
    
            select_args = copy.deepcopy(self.args)
            select_args['total_epsilon'] = self.epsilon
            select_args['depend_epsilon'] = self.depend_epsilon
            select_args['remain_rho'] = self.remain_rho
            select_args['threshold'] = 5000
            
            marginal_selection = MarginalSelection(dataset, select_args, self.args)
            marginals = marginal_selection.select_marginals()
            self.data_store.save_marginal(marginals)
        else:
            marginals = self.data_store.load_marginal()
        
        return marginals

    def recode_attrs(self, sigma):
        self.logger.info("recoding attrs")
    
        # sigma = self._calculate_sigma(self.recode_epsilon, self.num_attributes)
        attr_recode = AttrRecord(self.original_dataset)
        attr_recode.recode(sigma)
    
        return attr_recode

    def _calculate_rho(self, epsilon):
        composition = AdvancedComposition()
        sigma = composition.gauss_zcdp(epsilon, self.delta, self.args['marg_add_sensitivity'], 1)
        
        return (self.args['marg_add_sensitivity'] ** 2 / (2.0 * sigma ** 2))
    
    def _calculate_sigma(self, epsilon, num_views):
        composition = AdvancedComposition()
    
        return composition.gauss_zcdp(epsilon, self.delta, self.args['marg_add_sensitivity'], num_views)
