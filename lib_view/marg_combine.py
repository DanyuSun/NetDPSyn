import logging
import itertools
import copy
from collections import defaultdict

import networkx as nx


class MargCombine:
    def __init__(self, domain, marginals):
        self.logger = logging.getLogger("marg_combine")

        self.domain = domain
        self.marginals = marginals
        self.clip_layers = []

        self.graph = self._construct_graph()

    def determine_marginals(self, threshold, enable=True):
        if not enable:
            return self.marginals

        self.identify_cliques()

        self.combine_marginals_c1(threshold)

        return self.combined_marginals + self.identify_missing_depend()

    def _construct_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.domain.attrs)

        for m in self.marginals:
            graph.add_edges_from(itertools.combinations(m, 2))

        return graph

    def identify_cliques(self):
        all_cliques = nx.enumerate_all_cliques(self.graph)
        self.size_cliques = defaultdict(list)

        for clique in all_cliques:
            self.size_cliques[len(clique)].append(clique)

    def identify_missing_depend(self):
        missing_depend = copy.deepcopy(self.marginals)

        for marginals in self.combined_marginals:
            for marg in itertools.combinations(marginals, 2):
                if marg in missing_depend:
                    missing_depend.remove(marg)
                elif (marg[1], marg[0]) in missing_depend:
                    missing_depend.remove((marg[1], marg[0]))

        return missing_depend

    def combine_marginals_c1(self, threshold):
        self.logger.info("combining marginals")

        self.combined_marginals = []
        selected_attrs = set()

        for size in range(len(self.size_cliques), 2, -1):
            for clique in self.size_cliques[size]:
                if len(set(clique) & selected_attrs) <= 2 and self.domain.size(clique) < threshold:
                    self.combined_marginals.append(tuple(clique))
                    selected_attrs.update(clique)
