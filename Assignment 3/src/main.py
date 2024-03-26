import numpy as np
from utils import *
from page_rank import PageRank
from topic_sensitive_page_rank import TopicSensitivePageRank
import time

# Uncomment the following block if you want to compute PageRank without topic sensitivity
# page_rank = PageRank()
# start_time = time.time()
# rank = page_rank.power_iteration()
# end_time = time.time()
# print(rank)
# np.save("rank.npy", rank)
# print(f"PageRank computation time: {end_time - start_time} seconds")

total_time = 0
num_topics = 0

for topic_index in range(10, 13):
    topic_sensitive_page_rank = TopicSensitivePageRank(topic_index)
    start_time = time.time()
    rank = topic_sensitive_page_rank.power_iteration()
    end_time = time.time()
    np.save(f"topic_rank/rank_{topic_index}.npy", rank)
    computation_time = end_time - start_time
    total_time += computation_time
    num_topics += 1
    print(f"Topic {topic_index} computation time: {computation_time} seconds")

average_time = total_time / num_topics
print(f"Average computation time: {average_time} seconds")
