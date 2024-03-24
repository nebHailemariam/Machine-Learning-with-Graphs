import numpy as np

DATA_PATH = r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\data\\"


def get_adjacency_matrix_row(indices: list, edge_list, max_index):
    edge_list = [edge for edge in edge_list if edge[0] in indices]
    row = np.zeros((len(indices), max_index))

    for connection in edge_list:
        index = indices.index(connection[0])
        row[index][connection[1]] = 1

    row_sum = np.expand_dims(np.sum(row, axis=1), axis=1)
    row_sum_copy = row_sum.copy()
    row_sum_copy[row_sum_copy == 0] = 1
    row /= row_sum_copy
    zeros = np.where(row_sum == 0)

    for idx in zeros[0]:
        row[idx] += 1 / max_index
        row[idx][indices[idx]] = 0

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


def get_query_topic_distro(file_path):
    max_index = get_matrix_max_size()
    row = np.zeros((max_index))

    with open(file_path + r"\transition.txt") as file:
        transitions = file.readlines()

        for transition in transitions:
            doc, topic = transition.split(" ")
            row[doc - 1] = topic

    return row


def get_doc_topic(file_path, topic_index):
    max_index = get_matrix_max_size(file_path)
    row = np.zeros((max_index))
    topics = {i: [] for i in range(1, 13)}

    with open(file_path + r"\doc_topics.txt") as file:
        transitions = file.readlines()

        for transition in transitions:
            node, topic = transition.split(" ")
            topics[int(topic)].append(int(node) - 1)

    if topics[topic_index] == []:
        row += 1 / len(row)
    else:
        for doc in topics[topic_index]:
            row[doc] += 1 / (len(topics[topic_index]))
    print(row)
    print(topics)
    return row


def get_matrix_edge_list(file_path):
    edge_list = []

    with open(file_path + r"\transition.txt") as file:
        transitions = file.readlines()

        for transition in transitions:
            a, b, _ = transition.split(" ")
            edge_list.append((int(a) - 1, int(b) - 1))
    return edge_list
