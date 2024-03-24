import numpy as np
from utils import *


class PageRank:

    def __init__(self, alpha=0.2) -> None:

        self.edge_list = get_matrix_edge_list(DATA_PATH)
        self.max_index = get_matrix_max_size(DATA_PATH)
        self.alpha = alpha

    def power_iteration(self):
        rank = np.zeros((self.max_index, 1)) + 1 / (self.max_index)
        rank /= rank.sum()
        prev_rank = np.zeros((self.max_index, 1))
        counter = 0

        while np.linalg.norm(rank - prev_rank) > 10**-8:

            counter += 1
            prev_rank = rank.copy()
            rank = np.zeros((self.max_index, 1))
            for rank_idx in range(0, len(rank), 4000):
                print(counter, rank_idx)
                mat = get_adjacency_matrix_row(
                    [i for i in range(rank_idx, min(rank_idx + 4000, self.max_index))],
                    self.edge_list,
                    self.max_index,
                )

                rank += np.expand_dims(
                    np.sum(prev_rank[rank_idx : rank_idx + 4000] * mat, axis=0), axis=1
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
