import logging

import numpy as np


class NonNegativity:
    def __init__(self, count):
        self.count = np.copy(count)
    
    def norm_sub(self):
        summation = np.sum(self.count)
        lower_bound = 0.0
        upper_bound = - np.sum(self.count[self.count < 0.0])
        current_summation = 0.0
        delta = 0.0
        
        while abs(summation - current_summation) > 1.0:
            delta = (lower_bound + upper_bound) / 2.0
            new_count = self.count - delta
            new_count[new_count < 0.0] = 0.0
            current_summation = np.sum(new_count)
            
            if current_summation < summation:
                upper_bound = delta
            elif current_summation > summation:
                lower_bound = delta
            else:
                break
        
        self.count = self.count - delta
        self.count[self.count < 0.0] = 0.0
        
        return self.count

    def norm_cut(self):
        # set all negative value to 0.0
        negative_indices = np.where(self.count < 0.0)[0]
        negative_total = abs(np.sum(self.count[negative_indices]))
        self.count[negative_indices] = 0.0
    
        # find all positive value and sort them in ascending order
        positive_indices = np.where(self.count > 0.0)[0]
    
        if positive_indices.size != 0:
            positive_sort_indices = np.argsort(self.count[positive_indices])
            sort_cumsum = np.cumsum(self.count[positive_indices[positive_sort_indices]])
        
            # set the smallest positive value to 0.0 to preserve the total density
            threshold_indices = np.where(sort_cumsum <= negative_total)[0]
        
            if threshold_indices.size == 0:
                self.count[positive_indices[positive_sort_indices[0]]] = sort_cumsum[0] - negative_total
            else:
                self.count[positive_indices[positive_sort_indices[threshold_indices]]] = 0.0
                next_index = threshold_indices[-1] + 1
            
                if next_index < positive_sort_indices.size:
                    self.count[positive_indices[positive_sort_indices[next_index]]] = sort_cumsum[
                                                                                          next_index] - negative_total
        else:
            self.count[:] = 0.0
    
        return self.count


def main():
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
    
    count_before = np.array([-1.5, -2.5, 2, 1, 10])
    
    non_negativity = NonNegativity(count_before)
    
    count_after = non_negativity.norm_cut()
    print(count_after)


if __name__ == "__main__":
    main()
