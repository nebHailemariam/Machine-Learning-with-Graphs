import json
import random


# Function to filter instances for each star rating
def filter_instances(file_path, train_size=1000, test_size=200):
    reviews_by_stars_train = {}
    reviews_by_stars_test = {}

    # Initialize an empty list for each star rating
    for i in range(1, 6):
        reviews_by_stars_train[i] = []
        reviews_by_stars_test[i] = []

    # Read the JSON file
    with open(file_path, "r") as f:
        for line in f:
            review = json.loads(line)
            stars = review["stars"]
            # Append the review to the respective star rating list
            if len(reviews_by_stars_train[stars]) < train_size:
                reviews_by_stars_train[stars].append(review)
            elif len(reviews_by_stars_test[stars]) < test_size:
                reviews_by_stars_test[stars].append(review)

    # Select 1000 instances for each star rating for train set
    filtered_reviews_train = []
    for i in range(1, 6):
        filtered_reviews_train.extend(reviews_by_stars_train[i])

    # Select 200 instances for each star rating for test set
    filtered_reviews_test = []
    for i in range(1, 6):
        filtered_reviews_test.extend(reviews_by_stars_test[i])

    return filtered_reviews_train, filtered_reviews_test


# Example usage
file_path = r"C:\Users\Nebiyou Hailemariam\Desktop\development\Machine-Learning-with-Graphs\Assignment 2\Applied Machine Learning Project\data\yelp_reviews_train.json"
filtered_instances_train, filtered_instances_test = filter_instances(file_path)

# Save the filtered instances to files for train and test sets
train_output_file_path = "filtered_yelp_reviews_train.json"
with open(train_output_file_path, "w") as f_train:
    for review in filtered_instances_train:
        f_train.write(json.dumps(review) + "\n")

test_output_file_path = "filtered_yelp_reviews_test.json"
with open(test_output_file_path, "w") as f_test:
    for review in filtered_instances_test:
        f_test.write(json.dumps(review) + "\n")

# Print the number of instances for each star rating for train set
print("Train Set:")
for i in range(1, 6):
    print(
        f"Number of instances for {i} stars: {len([review for review in filtered_instances_train if review['stars'] == i])}"
    )

# Print the number of instances for each star rating for test set
print("\nTest Set:")
for i in range(1, 6):
    print(
        f"Number of instances for {i} stars: {len([review for review in filtered_instances_test if review['stars'] == i])}"
    )
