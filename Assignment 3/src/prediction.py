from utils import *

user_topic_distro = load_user_topic_distro(DATA_PATH)
load_query_topic_distro(DATA_PATH)

for user, query_id in sorted(user_topic_distro.keys()):
    print(user, query_id)
    global_page_rank_prediction(DATA_PATH, RANK_PATH, user, query_id, CM)
