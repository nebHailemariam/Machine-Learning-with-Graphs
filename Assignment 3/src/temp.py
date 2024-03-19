import numpy as np


class PageRank:

    def __init__(self, transition_matrix, alpha=0.2) -> None:

        self.transition_matrix = transition_matrix
        self.alpha = alpha

    def power_iteration(self):
        rank = np.random.rand(len(self.transition_matrix))
        rank /= rank.sum()
        counter = 0

        while counter < 1:
            prev_rank = rank.copy()
            for rank_idx in range(len(rank)):
                rank[rank_idx] = np.dot(self.get_transition_column(rank_idx), prev_rank)
            rank = (1 - self.alpha) * rank + self.alpha * (1 / rank.shape[0])
            counter += 1

        return rank

    def get_transition_column(self, column_index):
        row = np.zeros(len(self.transition_matrix))
        idx = 0
        for transition in self.transition_matrix:
            row[idx] = transition.todense()[0][column_index]
            idx += 1

        return row


# adj_matrix = np.array([
#     [0, 1, 0, 0],
#     [1, 0, 0, 0],
#     [1, 1, 0, 0],
#     [1, 1, 0, 0]
# ])
# page_rank = PageRank(adj_matrix)
# print(page_rank.power_iteration())
