# COMS 4771
# Steven Liu xl2948

import decimal
import glob
import random
import shutil
import numpy as np
import os
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer


# File Processing - Extract from a file, perform word stemming, and return a list of root words
def extract_file(file_address):
    trailing_form = ['ing', 'ed', 's', 'es']
    reader = open(file_address, encoding="ISO-8859-1")
    raw_word_list = reader.read().split()
    word_list = []
    for word in raw_word_list:
        new_word = None
        if word.isalpha():
            # Check if ends in trailing form. If so, remove it.
            for trailing in trailing_form:
                if word.endswith(trailing) and word != trailing:
                    new_word = word[:-len(trailing)]
            # If not, use the original word
            new_word = word.lower()
            word_list.append(new_word)
    reader.close()
    return word_list


# File Processing - Extract from folder, perform word stemming, and return bag of root words
def extract_folder(folder_address):
    print("Extracting from", folder_address)
    bag_of_words = {}
    trailing_form = ['ing', 'ed', 's', 'es']
    for address in glob.glob(folder_address + "*.txt"):
        r = open(address, encoding = "ISO-8859-1")
        file_word_list = r.read().split()
        for word in file_word_list:
            new_word = None
            if word.isalpha():
                # Check if ends in trailing form. If so, remove it.
                for trailing in trailing_form:
                    if word.endswith(trailing) and word != trailing:
                        new_word = word[:-len(trailing)]
                # If not, use the original word
                new_word = word.lower()
                if new_word in bag_of_words:
                    bag_of_words.update({new_word: bag_of_words.get(new_word) + decimal.Decimal(1)})
                else:
                    bag_of_words[new_word] = decimal.Decimal(1)
        r.close()

    feature_names = list(bag_of_words.keys())
    # Turn count into frequency
    s = sum(bag_of_words.values()) * decimal.Decimal(1)
    for key, value in bag_of_words.items():
        bag_of_words.update({key: value/s})

    return bag_of_words


# File Processing - Concatenate ham and spam data into vectors
def vectorize_training(ham_address, spam_address, num_features):
    print("Vectorizing training data...")
    corpus = []
    vectorizer = CountVectorizer(analyzer='word',max_features=num_features)
    num_ham = 0
    num_spam = 0
    for address in glob.glob(ham_address + "*.txt"):
        r = open(address, encoding="ISO-8859-1")
        file = r.read()
        corpus.append(file)
        num_ham += 1
    for address in glob.glob(spam_address + "*.txt"):
        r = open(address, encoding="ISO-8859-1")
        file = r.read()
        corpus.append(file)
        num_spam += 1
    X = vectorizer.fit_transform(corpus)
    vectors = X.toarray()
    features = vectorizer.get_feature_names()
    return vectors, features, vectorizer, num_ham, num_spam


# File Processing - Turn a file into vector
def vectorize_file(address, vectorizer):
    with open(address, encoding="ISO-8859-1") as file:
        text = file.read().replace('\n', '')
        vector = vectorizer.transform([text])
        return vector.toarray()[0]


# File Processing - Divide the data into training set and testing set
def train_test_split(ham_address, spam_address, test_prop):
    print("="*80)
    print("[Processing files]")
    print("-"*80)

    # Make temporary directories to hold testing & training data
    print("Creating temporary directories...")
    temp_all = './temp_all/'
    temp_train_ham = './temp_train_ham/'
    temp_train_spam = './temp_train_spam/'
    temp_test = './temp_test/'
    if os.path.exists(temp_all):
        shutil.rmtree(temp_all)
    os.makedirs(temp_all)
    if os.path.exists(temp_train_ham):
        shutil.rmtree(temp_train_ham)
    os.makedirs(temp_train_ham)
    if os.path.exists(temp_train_spam):
        shutil.rmtree(temp_train_spam)
    os.makedirs(temp_train_spam)
    if os.path.exists(temp_test):
        shutil.rmtree(temp_test)
    os.makedirs(temp_test)

    # Copy all ham and spam into ./temp_all
    print("Copying the dataset into temporary directory...")
    for path in glob.glob(os.path.join(ham_address, '*.txt')):
        file_name = path.split("/")[-1]
        shutil.copyfile(path, './temp_all/' + file_name)
    for path in glob.glob(os.path.join(spam_address, '*.txt')):
        file_name = path.split("/")[-1]
        shutil.copyfile(path, './temp_all/' + file_name)

    # Randomly select a portion of files and put into the testing set
    print("Selecting data for testing...")
    num_files = len(glob.glob(os.path.join(temp_all, '*.txt')))
    num_test = int(num_files * test_prop)
    file_indexes = random.sample(range(0, num_files-1), num_test)
    file_list = glob.glob(os.path.join(temp_all, '*.txt'))
    for file_index in file_indexes:
        file_name = os.path.basename(file_list[file_index])
        shutil.move(file_list[file_index], "./temp_test/" + file_name)

    # Put remaining files into the respective ham & spam training directories
    print("Copying ham and spam emails into respective folders...")
    num_ham = 0
    num_spam = 0
    for path in glob.glob(os.path.join(temp_all, '*.txt')):
        file_name = path.split('/')[-1]
        label = file_name.split('.')[-2]
        if label == 'ham':
            shutil.move(path, "./temp_train_ham/" + file_name)
            num_ham += 1
        else:
            shutil.move(path, "./temp_train_spam/" + file_name)
            num_spam += 1

    # Print out results
    print("Summary:")
    print("- Number of emails:", len(file_list))
    print("- Number of training emails:", num_ham + num_spam)
    print("--- Number of ham:", num_ham)
    print("--- Number of spam:", num_spam)
    print("- Number of testing emails:", num_test)

    return temp_all, temp_train_ham, temp_train_spam, temp_test


# TODO: Classifier - Naive Bayes
def naive_bayes(ham_address, spam_address, test_address):
    pred = []
    bag_of_words_ham = extract_folder(ham_address)
    bag_of_words_spam = extract_folder(spam_address)

    p_ham = decimal.Decimal(len(ham_address) / (len(spam_address) + len(ham_address)))
    p_spam = decimal.Decimal(1 - p_ham)
    print("P(ham) in training set:", p_ham)
    print("P(spam) in training set:", p_spam)

    for address in glob.glob(test_address + "*.txt"):
        word_list = extract_file(address)
        ham = decimal.Decimal(1)
        spam = decimal.Decimal(1)
        for word in word_list:
            if word in bag_of_words_ham and word in bag_of_words_spam:
                ham = ham * p_ham * bag_of_words_ham.get(word)
                spam = spam * p_spam * bag_of_words_spam.get(word)
        if ham > spam:
            pred.append("ham")
        else:
            pred.append("spam")
    num_ham = 0
    num_spam = 0
    for res in pred:
        if res == 'ham':
            num_ham += 1
        else:
            num_spam += 1

    return pred


# Nearest Neighbor Helper - L1, L2, and Vector Max Distance between two emails
def distance(train_vector, test_vector, mode):
    score = 0
    # L1 distance
    if mode == 1:
        for i in range(len(train_vector)):
            score += min(train_vector[i], test_vector[i])
    # L2 distance
    elif mode == 2:
        for i in range(len(train_vector)):
            score += (train_vector[i] - test_vector[i]) * (train_vector[i] - test_vector[i])
        score = sqrt(score)
    # Vector Max distance
    else:
        for i in range(len(train_vector)):
            diff = abs(train_vector[i] - test_vector[i])
            if diff > score:
                score = diff
    return score


# TODO: Classifier - Nearest Neighbor
def nearest_neighbor(ham_address, spam_address, test_address, k, num_features, mode):
    print("Select", k, "nearest neighbors and", num_features, "features.")
    vectors, feature_names, vectorizer, num_ham, num_spam = vectorize_training(ham_address, spam_address, num_features)
    pred = []

    # Calculate distance between each test email and training set
    print("Calculating distance...(This would take some time)")
    for address in glob.glob(test_address + "*.txt"):
        # Get vector of current email
        test_vector = vectorize_file(address, vectorizer)
        dist_dict = {}
        # Calculate distance
        for i in range(len(vectors)):
            dist_dict[i] = distance(vectors[i], test_vector, mode)
        # Pick nearest k neighbors (index in the dictionary)
        dist_list = sorted(dist_dict, key=dist_dict.get, reverse=True)
        # Calculate number of hams in neighbors
        ham_tags = 0
        for index in dist_list[1:k]:
            if index < num_ham:
                ham_tags += 1
        # If ham is the majority
        if ham_tags > k/2:
            pred.append("ham")
        # If spam is the majority
        else:
            pred.append("spam")
    return pred


# Decision Tree Helper - Return a dict of feature, left dataset and right dataset, calls split_data()
def get_split(dataset, features):
    max_reduction = float('-inf')
    max_feature = -1
    max_left = None
    max_right = None
    for i in range(len(features)):
        # print("i:", i)
        left, right = split_data(i, dataset)
        reduction = uncertainty_reduction(left, right)
        # print("reduction:", reduction)
        if reduction > max_reduction:
            max_feature = i
            max_reduction = reduction
            max_left = left
            max_right = right
    # print('max_reduction:', max_reduction)
    # print('feature:', features[max_feature])
    # print("---------")
    return {'feature': features[max_feature], 'left': max_left, 'right': max_right, 'leaf': False}


# Decision Tree Helper - Split dataset into left and right using the feature
def split_data(i, dataset):
    left, right = [], []
    for data in dataset:
        # Contains the feature
        if data[0][i] > 0:
            left.append(data)
        # Doesn't contain the feature
        else:
            right.append(data)
    return left, right


# Decision Tree Helper - Get majority and proportion of majority of a dataset, used in uncertainty_reduction
def get_majority(dataset):
    majority = None
    proportion = 0
    # data: data[0] is list of words, data[1] is tag
    num_ham = 0
    num_spam = 0
    for data in dataset:
        if data[1] == 'ham':
            num_ham += 1
        else:
            num_spam += 1
    # print('num_ham:', num_ham)
    # print('num_spam:', num_spam)
    if num_ham > num_spam:
        majority = 'ham'
        proportion = num_ham/(num_ham+num_spam)
    else:
        majority = 'spam'
        proportion = num_spam/(num_ham+num_spam)
    return majority, proportion


# Decision Tree Helper - Calculate uncertainty reduction
def uncertainty_reduction(left, right):
    if len(left) == 0 or len(right) == 0:
        return -100

    left_majority, left_majority_prop = get_majority(left)
    right_majority, right_majority_prop = get_majority(right)

    left_entropy = 0
    right_entropy = 0

    if left_majority_prop == 1:
        left_entropy = 0
    else:
        left_entropy = left_majority_prop * np.log(1 / left_majority_prop) + (1 - left_majority_prop) * np.log(1 / (1 - left_majority_prop))

    if right_majority_prop == 1:
        right_entropy = 0
    else:
        right_entropy = right_majority_prop * np.log(1 / right_majority_prop) + (1 - right_majority_prop) * np.log(1 / (1 - right_majority_prop))

    return (len(left) / len(left) + len(right)) * left_entropy + (len(right) / len(left) + len(right)) * right_entropy


# Decision Tree Helper - The main splitting function, using recursion
def split_at_root(root, features, max_depth, depth):
    # print('depth:', depth)
    left = root['left']
    right = root['right']
    # print(len(left))
    # print(len(right))
    if len(left) == 0 or len(right) == 0:
        lf = leaf(left+right)
        root['left'] = lf
        root['right'] = lf
        # print("LEAF")
        return
    if depth >= max_depth:
        root['left'], root['right'] = leaf(left), leaf(right)
        # print("MAX DEPTH")
        return

    root['left'] = get_split(left, features)
    split_at_root(root['left'], features, max_depth, depth+1)

    root['right'] = get_split(right, features)
    split_at_root(root['right'], features, max_depth, depth+1)


# Decision Tree Helper - Return a leaf node
def leaf(dataset):
    # return get_majority(dataset)
    return {'feature': get_majority(dataset), 'left': None, 'right': None, 'leaf': True}


# TODO: Classifier - Decision Tree
def decision_tree(ham_address, spam_address, test_address, Decision_Tree_Depth):
    # Vectors: list of lists containing #times each feature appear in an email
    vectors, feature_names, vectorizer, num_ham, num_spam = vectorize_training(ham_address, spam_address, 100)
    # print("VECTORS:", len(vectors))
    # print("NUMHAM", num_ham)
    # print("NUMSPAM", num_spam)

    # Dataset: a list of tuples [vector, tag]
    dataset = []
    counter = 0
    for vector in vectors:
        if counter < num_ham:
            dataset.append([vector, 'ham'])
        else:
            dataset.append([vector, 'spam'])
        counter += 1

    # Build the tree
    print("Building tree...")
    root = get_split(dataset, feature_names)
    split_at_root(root, feature_names, Decision_Tree_Depth, 1)

    # Test
    pred = []
    for address in glob.glob(test_address + "*.txt"):
        # print(address)
        word_list = extract_file(address)
        node = root
        feature = node['feature']
        while node is not None:
            # print(node['feature'])
            # Check if leaf node
            if node['leaf']:
                pred.append(node['feature'][0])
                break
            # Check if email contains the feature
            if feature in word_list:
                node = node['left']
            else:
                node = node['right']
            # Update feature
            feature = node['feature']

    return pred


# TODO: Benchmark all algorithms
def benchmark(ham_address, spam_address, KNN_k, KNN_feature_num, Decision_Tree_Depth, train_prop):
    training, ham, spam, X_test = train_test_split(ham_address, spam_address, train_prop)

    # Get true labels of the testing set from file names
    y_test = []
    for path in glob.glob(os.path.join(X_test, '*.txt')):
        label = str(path.split("/")[-1].split(".")[-2])
        y_test.append(label)

    # Run naive bayes
    print("="*80)
    print("[Naive Bayes]")
    print("-"*80)
    pred = naive_bayes(ham, spam, X_test)
    num_correct = 0
    for i in range(len(pred)):
        if y_test[i] == pred[i]:
            num_correct += 1
    accuracy = num_correct / len(y_test)
    print("Accuracy:", accuracy)

    # Run nearest neighbor L1
    print("="*80)
    print("[Nearest Neighbor: L1]")
    print("-"*80)
    pred = nearest_neighbor(ham, spam, X_test, KNN_k, KNN_feature_num, 1)
    num_correct = 0
    for i in range(len(y_test)):
        if y_test[i] == pred[i]:
            num_correct += 1
    accuracy = num_correct / len(y_test)
    print("- Accuracy:", accuracy)

    # Run nearest neighbor L2
    print("="*80)
    print("[Nearest Neighbor: L2]")
    print("-"*80)
    pred = nearest_neighbor(ham, spam, X_test, KNN_k, KNN_feature_num, 2)
    num_correct = 0
    for i in range(len(y_test)):
        if y_test[i] == pred[i]:
            num_correct += 1
    accuracy = num_correct / len(y_test)
    print("- Accuracy:", accuracy)

    # Run nearest neighbor L_inf
    print("="*80)
    print("[Nearest Neighbor: L_inf]")
    print("-"*80)
    pred = nearest_neighbor(ham, spam, X_test, KNN_k, KNN_feature_num, 3)
    num_correct = 0
    for i in range(len(y_test)):
        if y_test[i] == pred[i]:
            num_correct += 1
    accuracy = num_correct / len(y_test)
    print("- Accuracy:", accuracy)

    # Run decision tree
    print("="*80)
    print("[Decision Tree]")
    print("-"*80)
    pred = decision_tree(ham, spam, X_test, Decision_Tree_Depth)
    num_correct = 0
    for i in range(len(y_test)):
        if y_test[i] == pred[i]:
            num_correct += 1
    accuracy = num_correct / len(y_test)
    print("Accuracy:", accuracy)


# Parameters: [ham_address], [spam_address], [KNN_k], [KNN_feature_num], [Decision_Tree_Depth], [train_prop]
if __name__ == '__main__':
    benchmark("./enron1/ham/", "./enron1/spam/", 13, 100, 10, 0.2)
