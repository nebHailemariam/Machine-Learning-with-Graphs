import numpy as np

DATA_PATH = r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\data\\"


def get_adjacency_matrix_row(index, edge_list, max_index):
    edge_list = [edge for edge in edge_list if edge[0] == index]
    row = np.zeros((max_index))

    for connection in edge_list:
        row[connection[1]] = 1

    row /= max(1, sum(row))

    return row


def get_matrix_max_size(file_path):
    edge_list = []

    with open(file_path + r"\transition.txt") as file:
        transitions = file.readlines()

        for transition in transitions:
            a, b, _ = transition.split(" ")
            edge_list.append((int(a) - 1, int(b) - 1))

    max_index = (
        max(
            max(edge_list, key=lambda x: x[0])[0], max(edge_list, key=lambda x: x[1])[1]
        )
        + 1
    )
    return max_index


def get_matrix_edge_list(file_path):
    edge_list = []

    with open(file_path + r"\transition.txt") as file:
        transitions = file.readlines()

        for transition in transitions:
            a, b, _ = transition.split(" ")
            edge_list.append((int(a) - 1, int(b) - 1))
    return edge_list
