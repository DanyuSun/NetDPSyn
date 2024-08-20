import logging
import copy

import numpy as np

from lib_view.view import View


class Consistenter:
    class SubsetWithDependency:
        def __init__(self, attr_set):
            # a list of categories
            self.attr_set = attr_set
            # a set of tuples this object depends on
            self.dependency = set()
    
    def __init__(self, views, dataset_domain, consist_parameters):
        self.logger = logging.getLogger("Consistenter")
        
        self.views = views
        self.dataset_domain = dataset_domain
        self.num_categories = np.array(dataset_domain.shape)
        self.iterations = consist_parameters["consist_iterations"]
        self.non_negativity = consist_parameters["non_negativity"]

    def _compute_dependency(self):
        subsets_with_dependency = {}

        for view_key in self.views:
            # create a dependency subset for each view
            new_subset = self.SubsetWithDependency(set(view_key))
            subsets_temp = copy.deepcopy(subsets_with_dependency)

            for subset_key, subset_value in subsets_temp.items():
                # sort here to avoid producing multiple keys with the same set
                attr_intersection = sorted(subset_value.attr_set & set(view_key))

                if attr_intersection:
                    # add interacted attrs as dependency subset
                    if tuple(attr_intersection) not in subsets_with_dependency:
                        intersection_subset = self.SubsetWithDependency(set(attr_intersection))
                        subsets_with_dependency[tuple(attr_intersection)] = intersection_subset

                    # add dependency to subset_key, and avoid regarding self as dependency
                    if not set(attr_intersection) == set(subset_key):
                        subsets_with_dependency[subset_key].dependency.add(tuple(attr_intersection))

                    # add dependency to the new view

                    if not set(attr_intersection) == set(view_key):
                        new_subset.dependency.add(tuple(attr_intersection))

                    # add dependency to other subsets
                    for sub_key, sub_value in subsets_with_dependency.items():
                        if set(attr_intersection) < sub_value.attr_set:
                            subsets_with_dependency[sub_key].dependency.add(tuple(attr_intersection))

            subsets_with_dependency[view_key] = new_subset

        return subsets_with_dependency
    
    def consist_views(self):
        def find_subset_without_dependency():
            for key, subset in subsets_with_dependency_temp.items():
                if not subset.dependency:
                    return key, subset
            
            return None, None
        
        def find_views_containing_target(target):
            result = []
            
            for _, view in self.views.items():
                if target <= view.attr_set:
                    result.append(view)
            
            return result

        def remove_subset_from_dependency(target):
            for _, subset in subsets_with_dependency_temp.items():
                if target in subset.dependency:
                    subset.dependency.remove(target)
        
        def consist_on_subset(target, target_views):
            common_view = View(self.dataset_domain.project(target), self.dataset_domain)
            common_view.init_consist_parameters(len(target_views))

            for index, view in enumerate(target_views):
                common_view.project_from_bigger_view(view, index)

            common_view.calculate_delta()

            for index, view in enumerate(target_views):
                view.update_view(common_view, index)

        # calculate necessary variables
        for key, view in self.views.items():
            assert np.array_equal(view.num_categories, self.num_categories)
            
            view.calculate_tuple_key()
            # view.generate_attributes_index_set()
            # view.get_sum()

        # print('in consistent part:')
        # for key, view in self.views.items():
        #     print(key)
        #     if key == (('pkt', 'byt')) or key == (('byt', 'pkt')):
        #         print(view)
        #
        # exit()


        # calculate the dependency relationship
        subsets_with_dependency = self._compute_dependency()
        self.logger.debug("dependency computed")
        
        # ripple steps needs several iterations
        # for i in range(self.iterations):
        non_negativity = True
        iterations = 0

        while non_negativity and iterations < self.iterations:
            self.logger.info("consist for round %s" % (iterations,))

            # first make sure summation are the same
            consist_on_subset(set(), [view for _, view in self.views.items()])
            subsets_with_dependency_temp = copy.deepcopy(subsets_with_dependency)
            
            # consist views in the dependency tree
            while len(subsets_with_dependency_temp) > 0:
                key, subset = find_subset_without_dependency()
                target_views = find_views_containing_target(subset.attr_set)

                # only if the number of target views larger than 1, one need to consist views
                if len(target_views) > 1:
                    consist_on_subset(subset.attr_set, target_views)
                    remove_subset_from_dependency(key)

                subsets_with_dependency_temp.pop(key, None)

            self.logger.debug("consist finish")

            # if iterations % 10 == 0:
            #     self.check_whole_consistency()

            # ensure all cells in all views are non-negative
            views_count = 0
            
            for key, view in self.views.items():
                if (view.count < 0.0).any():
                    view.non_negativity(self.non_negativity, iterations)
                    view.get_sum()
                else:
                    views_count += 1
                
                if views_count == len(self.views):
                    self.logger.info("finish in %s round" % (iterations,))
                    non_negativity = False

            self.logger.debug("non-negativity finish")
            
            iterations += 1

        # calculate normalized count
        for key, view in self.views.items():
            view.calculate_normalize_count()
            # view.get_sum()

        # for key, view in self.views.items():
        #     print('view.count')
        #     print(view.count)
        #
        # exit()


