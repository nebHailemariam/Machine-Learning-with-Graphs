import numpy as np
from utils import *


class TopicSensitivePageRank:

    def __init__(self, topic_index, alpha=0.2) -> None:

        self.edge_list = get_matrix_edge_list(DATA_PATH)
        self.max_index = get_matrix_max_size(DATA_PATH)
        self.topic_sensitive = get_doc_topic(DATA_PATH, topic_index)
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
            rank = (
                ((1 - self.alpha) * rank)
                + ((self.alpha / 0.8) * self.topic_sensitive)
                + ((self.alpha / 0.2) * (1 / rank.shape[0]))
            )
            rank /= rank.sum()
        print(rank)
        return rank
