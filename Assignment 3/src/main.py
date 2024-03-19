import numpy as np
from utils import *
from page_rank import PageRank

page_rank = PageRank()
rank = page_rank.power_iteration()
print(len(rank))
print(sum(rank))
print(rank)
np.save("rank.npy", rank)
# adj_mat_row_sum = adjacency_matrix.sum(axis=1)
# adj_mat_row_sum[adj_mat_row_sum == 0] = 1
# adjacency_matrix = adjacency_matrix/adj_mat_row_sum[:,None]
# print(adjacency_matrix.shape)
# page_rank = PageRank(adjacency_matrix)
# print(page_rank.power_iteration.shape)
