import math


class AdvancedComposition:
    def __init__(self):
        pass

    def gauss_zcdp(self, epsilon, delta, sensitivity, k):
        tmp_var = 2 * k * sensitivity ** 2 * math.log(1 / delta)

        sigma = (math.sqrt(tmp_var) + math.sqrt(tmp_var + 2 * k * sensitivity ** 2 * epsilon)) / (2 * epsilon)

        return sigma
