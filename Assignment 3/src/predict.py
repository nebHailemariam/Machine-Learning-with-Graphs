from utils import *
import time

# Initialize dictionaries to hold total time for each function type
total_time = {}


# Function to update total time for each function type
def update_total_time(func_name, execution_time):
    if func_name in total_time:
        total_time[func_name] += execution_time
    else:
        total_time[func_name] = execution_time


# Load user topic distribution
user_topic_distro = load_user_topic_distro(DATA_PATH)

# Measure execution time for global_page_rank_prediction
for user, query_id in sorted(user_topic_distro.keys()):
    print(user, query_id)
    for method in [NS, WS, CM]:
        start_time = time.time()
        global_page_rank_prediction(DATA_PATH, RANK_PATH, user, query_id, method)
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"Execution time for global_page_rank_prediction with method {method}: {execution_time}"
        )
        update_total_time(f"global_page_rank_prediction_{method}", execution_time)

# Load query topic distribution
topic_distro = load_query_topic_distro(DATA_PATH)

# Measure execution time for query_based_topic_sensitive_page_rank
for user, query_id in sorted(topic_distro.keys()):
    print(user, query_id)
    for method in [NS, WS, CM]:
        start_time = time.time()
        query_based_topic_sensitive_page_rank(
            DATA_PATH,
            RANK_PATH,
            user,
            query_id,
            topic_distro[(user, query_id)],
            method,
            "QTSPR",
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"Execution time for query_based_topic_sensitive_page_rank with method {method}: {execution_time}"
        )
        update_total_time(
            f"query_based_topic_sensitive_page_rank_QTSPR_{method}", execution_time
        )

# Measure execution time for query_based_topic_sensitive_page_rank (second loop)
for user, query_id in sorted(user_topic_distro.keys()):
    print(user, query_id)
    for method in [NS, WS, CM]:
        start_time = time.time()
        query_based_topic_sensitive_page_rank(
            DATA_PATH,
            RANK_PATH,
            user,
            query_id,
            topic_distro[(user, query_id)],
            method,
            "PTSPR",
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"Execution time for query_based_topic_sensitive_page_rank with method {method}: {execution_time}"
        )
        update_total_time(
            f"query_based_topic_sensitive_page_rank_PTSPR_{method}", execution_time
        )

# Print sum of execution time for each function type
for func_name, sum_time in total_time.items():
    print(f"Sum of execution time for {func_name}: {sum_time}")
