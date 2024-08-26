from functools import reduce
import math

import numpy as np

from exp.exp import Exp
from lib_view.view import View
from lib_view.consistent import Consistenter


class ExpDPSyn(Exp):
    def __init__(self, args):
        super(ExpDPSyn, self).__init__(args)
        
        self.views_dict = {}
        self.singleton_key = []

    def construct_view(self, dataset, marginal):
        num_keys = reduce(lambda x, y: x * y, [dataset.domain.config[m] for m in marginal])
        self.logger.info("constructing %s views, num_keys: %s" % (marginal, num_keys))

        view = View(dataset.domain.project(marginal), dataset.domain)
        view.count_records(dataset.df.values)
        
        return view
    
    def anonymize_view(self, view, rho=0.0, epsilon=0.0):
        if self.epsilon != -1.0:
            if rho != 0.0 and epsilon == 0.0:
                sigma = math.sqrt(self.args['marg_add_sensitivity'] ** 2 / (2.0 * rho))
                noise = np.random.normal(scale=sigma, size=view.num_key)
                view.count += noise
            else:
                view.count += np.random.laplace(scale=self.args['marg_add_sensitivity'] / epsilon, size=view.num_key)
        self.logger.info("rho and epsilon for %s is %f and %f" % (str(view.attr_set), rho,epsilon))        

        return view
    
    def consist_views(self, recode_domain, views):
        self.logger.info("consisting views")
        
        consist_parameters = {
            "consist_iterations": self.args['consist_iterations'],
            "non_negativity": self.args['non_negativity'],
        }
        
        consistenter = Consistenter(views, recode_domain, consist_parameters)
        consistenter.consist_views()
        
        self.logger.info("consisted views")
        

        return views
