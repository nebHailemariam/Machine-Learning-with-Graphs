import numpy as np

DATA_PATH = r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\data\\"


def get_adjacency_matrix_row(index, edge_list, max_index):
    edge_list = [edge for edge in edge_list if edge[0] == index]
    row = np.zeros((max_index))
    checked = False

    for connection in edge_list:
        row[connection[1]] = 1
        checked = True

    if checked:
        row /= max(1, sum(row))
    else:
        row += 1 / (len(row) - 1)
        row[index] = 0

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
            row[doc] += 1 / (len(topics[topic_index]) - 1)
    return row


def get_matrix_edge_list(file_path):
    edge_list = []

    with open(file_path + r"\transition.txt") as file:
        transitions = file.readlines()

        for transition in transitions:
            a, b, _ = transition.split(" ")
            edge_list.append((int(a) - 1, int(b) - 1))
    return edge_list
