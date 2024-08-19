import itertools
import logging

import networkx as nx


class SepGraph:
    def __init__(self, domain, marginals):
        self.logger = logging.getLogger("sep_graph")

        self.domain = domain
        self.marginals = marginals

        self.graph = self._construct_graph()

    def _construct_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.domain.attrs)

        for m in self.marginals:
            graph.add_edges_from(itertools.combinations(m, 2))

        return graph

    def cut_graph(self):
        pass

    def find_sep_graph(self, iterate_marginals, enable=True):
        iterate_keys = {}


        if enable is False:
            keys = []

            for marginal in self.marginals:
                if marginal in iterate_marginals:
                    keys.append(marginal)

            iterate_keys[self._find_keys_set(self.marginals)] = keys

        else:
            for component in nx.connected_components(self.graph):
                keys = []

                for marginal in self.marginals:
                    if set(marginal) < component and marginal in iterate_marginals:
                        keys.append(marginal)

                iterate_keys[self._find_keys_set(keys)] = keys

        return iterate_keys

    def _find_keys_set(self, keys):
        keys_set = set()

        for key in keys:
            keys_set.update(key)

        return tuple(keys_set)

    def join_records(self, df):
        pass


class SubGraph:
    def __init__(self):
        self.attrs = set()
        self.iterate_keys = []
