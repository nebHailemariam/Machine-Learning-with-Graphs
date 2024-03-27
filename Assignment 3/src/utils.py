import numpy as np

DATA_PATH = r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\data\\"
RANK_PATH = r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 3\src\\"


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
        row += 1 / len(row) - 1
    else:
        for doc in topics[topic_index]:
            row[doc] += 1 / (len(topics[topic_index]))
    return row


def get_matrix_edge_list(file_path):
    edge_list = []

    with open(file_path + r"\transition.txt") as file:
        transitions = file.readlines()

        for transition in transitions:
            a, b, _ = transition.split(" ")
            edge_list.append((int(a) - 1, int(b) - 1))
    return edge_list


def load_user_topic_distro(file_path):

    with open(file_path + r"user-topic-distro.txt") as file:
        lines = file.readlines()
        distributions = {}
        for line in lines:
            line = line.strip().split()
            user_id, query_id = int(line[0]), int(line[1])
            distribution = [float(entry.split(":")[1]) for entry in line[2:]]
            distributions[(user_id, query_id)] = distribution
            distributions[(user_id, query_id)] = {
                index + 1: value for index, value in enumerate(distribution)
            }

    return distributions


def load_query_topic_distro(file_path):

    with open(file_path + r"query-topic-distro.txt") as file:
        lines = file.readlines()
        distributions = {}
        for line in lines:
            line = line.strip().split()
            user_id, query_id = int(line[0]), int(line[1])
            distribution = [float(entry.split(":")[1]) for entry in line[2:]]
            distributions[(user_id, query_id)] = distribution
            distributions[(user_id, query_id)] = {
                index + 1: value for index, value in enumerate(distribution)
            }

    return distributions


NS = "No_Search"
WS = "Weighted_Sum"
CM = "Custom_Method"


def get_rank_with_no_search_relevance(file_path, document_id):
    ranks = np.load(file_path).reshape(-1)
    score = ranks[document_id - 1]
    return score


def get_rank_with_weighted_sum(file_path, user_id, query_id):
    doc_scores = {}
    with open(file_path + f"\indri-lists\{user_id}-{query_id}.results.txt") as file:
        lines = file.readlines()
        for line in lines:
            QueryID, Q0, DocID, Rank, Score, RunID = line.split(" ")
            doc_scores[int(DocID) - 1] = float(Score)

    return doc_scores


def get_weighted_sum_rank(file_path, document_id, search_relevance):
    ranks = np.load(file_path).reshape(-1)

    score = (
        float(search_relevance[document_id - 1]) * 0.5
        + ranks[int(document_id) - 1] * 0.5
    )
    return score


def get_rank_with_custom_method(file_path, document_id, search_relevance):
    ranks = np.load(file_path).reshape(-1)

    score = float(search_relevance[document_id - 1]) * ranks[int(document_id) - 1]
    return score


def global_page_rank_prediction(file_path, rank_file_path, user_id, query_id, type):
    ranks = []
    if type == NS:
        with open(file_path + f"\indri-lists\{user_id}-{query_id}.results.txt") as file:
            lines = file.readlines()
            for line in lines:
                QueryID, Q0, DocID, _, Score, RunID = line.split(" ")
                score = get_rank_with_no_search_relevance(
                    rank_file_path + r"\rank.npy", int(DocID)
                )
                ranks.append((score, user_id, query_id, Q0, DocID, score, RunID))
    if type == WS:
        search_relevance = get_rank_with_weighted_sum(file_path, user_id, query_id)
        with open(file_path + f"\indri-lists\{user_id}-{query_id}.results.txt") as file:
            lines = file.readlines()
            for line in lines:
                QueryID, Q0, DocID, Rank, Score, RunID = line.split(" ")
                score = get_weighted_sum_rank(
                    rank_file_path + r"\rank.npy", int(DocID), search_relevance
                )
                ranks.append((score, user_id, query_id, Q0, DocID, score, RunID))
    if type == CM:
        search_relevance = get_rank_with_weighted_sum(file_path, user_id, query_id)
        with open(file_path + f"\indri-lists\{user_id}-{query_id}.results.txt") as file:
            lines = file.readlines()
            for line in lines:
                QueryID, Q0, DocID, Rank, Score, RunID = line.split(" ")
                score = get_rank_with_custom_method(
                    rank_file_path + r"\rank.npy", int(DocID), search_relevance
                )
                ranks.append((score, user_id, query_id, Q0, DocID, score, RunID))
        pass
    ranks.sort(key=lambda x: -x[0])
    with open(rank_file_path + rf"\rank_{type}", "a") as output_file:
        idx = 1
        for rank_info in ranks:
            rank, user_id, query_id, Q0, DocID, score, RunID = rank_info
            output_file.write(
                f"{user_id}-{query_id} {Q0} {DocID} {idx} {score} {RunID}"
            )
            idx += 1
