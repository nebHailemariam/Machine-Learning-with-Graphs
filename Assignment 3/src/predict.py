from utils import *

user_topic_distro = load_user_topic_distro(DATA_PATH)

for user, query_id in sorted(user_topic_distro.keys()):
    print(user, query_id)
    global_page_rank_prediction(DATA_PATH, RANK_PATH, user, query_id, NS)


topic_distro = load_query_topic_distro(DATA_PATH)

for user, query_id in sorted(topic_distro.keys()):
    print(user, query_id)
    query_based_topic_sensitive_page_rank(
        DATA_PATH,
        RANK_PATH,
        user,
        query_id,
        topic_distro[(user, query_id)],
        NS,
        "query_topic",
    )

for user, query_id in sorted(user_topic_distro.keys()):
    print(user, query_id)
    query_based_topic_sensitive_page_rank(
        DATA_PATH,
        RANK_PATH,
        user,
        query_id,
        topic_distro[(user, query_id)],
        NS,
        "user_topic",
    )
