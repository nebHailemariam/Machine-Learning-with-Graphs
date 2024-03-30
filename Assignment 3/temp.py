import numpy as np
from src.utils import *


def save_array_with_index(arr, file_name):
    with open(file_name, "w") as file:
        for i, value in enumerate(arr, start=1):
            file.write(f"{i} {value}\n")
    print(f"Values with index written to '{file_name}' successfully.")


arr = np.load(
    r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\src\topic_rank\rank.npy"
).reshape(-1)
file_name = "GPR.txt"
save_array_with_index(arr, file_name)


def save_weighted_sum_rank(file_path, weights, output_file):
    page_ranks = get_probability_weighted_sum_rank(file_path, weights)
    save_array_with_index(page_ranks, output_file)


user_topic_distro = load_user_topic_distro(DATA_PATH)
topic_distro = load_query_topic_distro(DATA_PATH)

save_weighted_sum_rank(
    r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\src\topic_rank\\",
    user_topic_distro[(2, 1)],
    "PTSPR-U2Q1.txt",
)

save_weighted_sum_rank(
    r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\src\topic_rank\\",
    topic_distro[(2, 1)],
    "QTSPR-U2Q1.txt",
)
