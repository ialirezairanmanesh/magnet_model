import numpy as np

"""
Author: Milad Abolhassani

"""

class roullet_wheel:

    @classmethod
    def select(cls, costs, num, replace=False):

        cls.fitness = np.max(costs) - costs
        s = np.sum(cls.fitness)

        # All costs are same return first one
        if s == 0 or len(list((filter(None, cls.fitness)))) < 2:
            selection = [0]*num
        else:
            selection_probs = cls.fitness.astype(np.float64) / s
            selection = np.random.choice(len(costs), num, replace=replace, p=selection_probs)
        
        return selection[0] if num == 1 else selection

#  class other_selection_types:
#   # def select()
#               ...
