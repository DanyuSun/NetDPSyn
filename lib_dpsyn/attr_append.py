import itertools
import logging
import copy

import networkx as nx
import numpy as np


class AttrAppend:
    def __init__(self, domain, marginals):
        self.logger = logging.getLogger("attr_append")

        self.domain = domain
        self.marginals = marginals
        self.clip_layers = []

        self.graph = self._construct_graph()

    def _construct_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.domain.attrs)

        for m in self.marginals:
            graph.add_edges_from(itertools.combinations(m, 2))

        return graph

    def clip_graph(self, enable=True):
        self.logger.info("clipping graph for appending")

        clip_marginals = copy.deepcopy(self.marginals)

        while enable:
            layer = ClipLayer()
            num_nodes = 0

            for node, degree in self.graph.degree():
                if degree == 1:
                    neighbor = next(self.graph.neighbors(node))

                    layer.attrs.add(node)
                    layer.attrs_ancestor[node] = neighbor

                    if (node, neighbor) in self.marginals:
                        layer.attrs_marginal[node] = (node, neighbor)
                    else:
                        layer.attrs_marginal[node] = (neighbor, node)

                    num_nodes += 1

            if num_nodes != 0:
                self.clip_layers.append(layer)
                isolated_attr = []

                for attr in layer.attrs_marginal:
                    try:
                        self.graph.remove_edge(layer.attrs_marginal[attr][0], layer.attrs_marginal[attr][1])
                        clip_marginals.remove(layer.attrs_marginal[attr])
                    except:
                        self.logger.info("isolated attr: %s" % (attr,))
                        isolated_attr.append(attr)

                for attr in isolated_attr:
                    layer.attrs.remove(attr)
                    layer.attrs_ancestor.pop(attr)
                    layer.attrs_marginal.pop(attr)
                    clip_marginals.append((attr,))
            else:
                break

        self.logger.info("totally %s layers" % (len(self.clip_layers)))

        return clip_marginals

    def append_attrs(self, df, views):
        for index, layer in enumerate(self.clip_layers[::-1]):
            self.logger.info("appending %s layer" % (index,))

            for append_attr in layer.attrs:
                anchor_attr = layer.attrs_ancestor[append_attr]
                anchor_record = np.copy(df[anchor_attr])
                unique_value = np.unique(anchor_record)
                append_record = np.zeros(anchor_record.size, dtype=np.uint32)

                marginal = views[layer.attrs_marginal[append_attr]].calculate_count_matrix()

                for value in unique_value:
                    indices = np.where(anchor_record == value)[0]

                    if self.domain.attr_index_mapping[anchor_attr] < self.domain.attr_index_mapping[append_attr]:
                        if np.sum(marginal[value, :]) != 0:
                            dist = marginal[value, :] / np.sum(marginal[value, :])
                        else:
                            dist = np.full(marginal.shape[1], 1.0 / marginal.shape[1])
                    else:
                        if np.sum(marginal[:, value]) != 0:
                            dist = marginal[:, value] / np.sum(marginal[:, value])
                        else:
                            dist = np.full(marginal.shape[0], 1.0 / marginal.shape[0])

                    cumsum = np.cumsum(dist)
                    start = 0

                    for i, v in enumerate(cumsum):
                        end = int(round(v * indices.size))
                        append_record[indices[start: end]] = i
                        start = end

                df[append_attr] = append_record
                np.random.shuffle(df.values)


class ClipLayer:
    def __init__(self):
        self.attrs = set()
        self.attrs_ancestor = {}
        self.attrs_marginal = {}


if __name__ == "__main__":
    from lib_dataset.domain import Domain

    attrs = ["A", "B", "C", "D", "E"]
    shape = [2, 2, 2, 2, 2]
    domain = Domain(attrs, shape)
    marginals = [("A", "B"), ("B", "C"), ("C", "A"), ("B", "D"), ("C", "E")]

    attr_append = AttrAppend(domain, marginals)
    clip_marginals = attr_append.clip_graph()
    a = 1
