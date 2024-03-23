import numpy as np
from utils import *


class PageRank:

    def __init__(self, alpha=0.2) -> None:

        self.edge_list = get_matrix_edge_list(DATA_PATH)
        self.max_index = get_matrix_max_size(DATA_PATH)
        self.alpha = alpha

    def power_iteration(self):
        rank = np.zeros(self.max_index) + 1 / (self.max_index)
        rank /= rank.sum()
        prev_rank = np.zeros(self.max_index)

        while np.linalg.norm(rank - prev_rank) > 10**-8:
            prev_rank = rank.copy()
            rank = np.zeros(self.max_index)
            for rank_idx in range(len(rank)):
                rank += (
                    get_adjacency_matrix_row(rank_idx, self.edge_list, self.max_index)
                    * prev_rank[rank_idx]
                )

            rank = ((1 - self.alpha) * rank) + (self.alpha * (1 / rank.shape[0]))
            rank /= rank.sum()
        return rank


# adj_matrix = np.array([
#     [0, 1, 0, 0],
#     [1, 0, 0, 0],
#     [1, 1, 0, 0],
#     [1, 1, 0, 0]
# ])
page_rank = PageRank()
# print(page_rank.power_iteration())
