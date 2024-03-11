YELP_TRAIN_DIR = "../src/data/yelp_reviews_train.json"
YELP_DEV_DIR = "../src/data/yelp_reviews_dev.json"
YELP_TEST_DIR = "../src/data/yelp_reviews_test.json"
STOP_WORD_DIR = "../src/data/stopword.list"


import json
from sklearn.metrics import mean_squared_error


def load_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            json_data = [json.loads(line) for line in file]
            return json_data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return None


def get_stop_words(file_path):
    stop_words = {}
    with open(file_path, "r") as file:
        for line_number, word in enumerate(file, start=1):
            word = word.strip()  # Remove leading and trailing whitespaces
            stop_words[word] = True
    return stop_words


train_data = load_json_file(YELP_TRAIN_DIR)
stop_words = get_stop_words(STOP_WORD_DIR)


def star_distribution(reviews):
    stars = {}

    for review in reviews:
        stars[review["stars"]] = 1 + stars.get(review["stars"], 0)

    total = sum(stars.values())
    distribution = [
        {"stars": key, "dist": stars[key] / total} for key in sorted(stars.keys())
    ]

    return stars, distribution


star_distribution(train_data)
import string


def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def preprocess_text(text):
    sentence = remove_punctuation(text.lower()).split(" ")
    return [
        word
        for word in sentence
        if word != ""
        and word not in stop_words
        and all(char.isalpha() for char in word)
    ]


import numpy as np


def preprocess_dataset_df(reviews, vocab):
    vocab_index = list(vocab.keys())
    features = np.zeros((len(reviews), 2000), dtype=np.int32)
    targets = []

    for review_idx in range(len(reviews)):

        sentence = preprocess_text(reviews[review_idx]["text"])
        targets.append(reviews[review_idx]["stars"])

        reviews[review_idx] = None

        for word in sentence:
            if word in vocab:
                features[review_idx][vocab_index.index(word)] = vocab[word]

    return features, targets


def preprocess_dataset_df_test(reviews, vocab):
    vocab_index = list(vocab.keys())
    features = np.zeros((len(reviews), 2000), dtype=np.int32)

    for review_idx in range(len(reviews)):
        sentence = preprocess_text(reviews[review_idx]["text"])

        reviews[review_idx] = None

        for word in sentence:
            if word in vocab:
                features[review_idx][vocab_index.index(word)] = vocab[word]

    return features


import numpy as np


def preprocess_dataset_ctf(reviews, vocab):
    vocab_index = list(vocab.keys())
    features = np.zeros((len(reviews), 2000), dtype=np.int32)
    targets = []

    for review_idx in range(len(reviews)):

        sentence = preprocess_text(reviews[review_idx]["text"])
        targets.append(reviews[review_idx]["stars"])

        reviews[review_idx] = None

        for word in sentence:
            if word in vocab and features[review_idx][vocab_index.index(word)] == 0:
                num_of_word_count = sentence.count(word)
                features[review_idx][vocab_index.index(word)] = num_of_word_count

    return features, targets


def preprocess_dataset_ctf_test(reviews, vocab):
    vocab_index = list(vocab.keys())
    features = np.zeros((len(reviews), 2000), dtype=np.int32)

    for review_idx in range(len(reviews)):
        sentence = preprocess_text(reviews[review_idx]["text"])

        reviews[review_idx] = None

        for word in sentence:
            if word in vocab and features[review_idx][vocab_index.index(word)] == 0:
                num_of_word_count = sentence.count(word)
                features[review_idx][vocab_index.index(word)] = num_of_word_count

    return features


def build_vocab_ctf(reviews):
    vocab = {}
    for review in reviews:
        words = preprocess_text(review["text"])

        for word in words:
            vocab[word] = 1 + vocab.get(word, 0)

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    top_2000_vocab = dict(sorted_vocab[:2000])

    return top_2000_vocab


def build_vocab_df(reviews):
    vocab = {}
    for review in reviews:
        document_frequency = {}
        words = preprocess_text(review["text"])

        for word in words:
            document_frequency[word] = 1

        for word in document_frequency.keys():
            vocab[word] = document_frequency[word] + vocab.get(word, 0)

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    top_2000_vocab = dict(sorted_vocab[:2000])

    return top_2000_vocab


def min_max_normalize_inplace(data):
    data -= data.min(axis=0)
    data /= data.max(axis=0) - data.min(axis=0) + 1e-10


def merge_and_save_vectors(vector1, vector2, output_file_path):
    merged_vector = np.column_stack((vector1, vector2))
    np.savetxt(output_file_path, merged_vector, fmt=["%d", "%.3f"], delimiter=" ")
