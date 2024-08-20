import logging

from lib_view.attr_depend import AttrDepend
from lib_view.marg_combine import MargCombine


class MarginalSelection:
    def __init__(self, dataset, select_config, args):
        self.logger = logging.getLogger('marginal_selection')
        
        self.dataset = dataset
        self.select_config = select_config
        self.args = args

    def select_marginals(self):
        self.logger.info('choosing pairs')
        
        marginals = self.choose_pairs_M3()
        marginals = self.combine_pairs(marginals)
    
        return marginals

    # MargDense
    def choose_pairs_M3(self):
        determine_marginal = AttrDepend(self.dataset, self.select_config['dataset_name'], self.args)

        if self.select_config['is_cal_depend'] is True:
            determine_marginal.transform_records_distinct_value()
            determine_marginal.calculate_pair_indif()

        determine_marginal.solve_score_function(self.select_config['dataset_name'], self.select_config['depend_epsilon'], self.select_config['remain_rho'],
                                                self.select_config['marg_add_sensitivity'], self.select_config['marg_select_sensitivity'],
                                                self.select_config['noise_add_method'])
        marginals = determine_marginal.handle_isolated_attrs(method="connect", sort=True)

        return marginals
    
    def combine_pairs(self, marginals):
        self.logger.info("%s marginals before combining" % (len(marginals)))

        marg_combine = MargCombine(self.dataset.domain, marginals)
        marginals = marg_combine.determine_marginals(self.select_config['threshold'], enable=self.select_config['is_combine'])

        self.logger.info("%s marginals after combining" % (len(marginals)))
        
        return marginals
