'''
    Date: 7/8/2020

    Natural Noise filter class
    Paper: 5 - Detecting Noise in Recommender System Databases
    use it to apply noise filtering to a certain dataset of ratings

    Description: 
        1- We define the consistency c of a rating ru,v as the Mean Absolute Error between the actual and predicted rating
        2- if c is greater that the threshold th, then the rating is considered as noise

        In this class, we need to select a recommender and traing our data before we use it since it relies on the predicted ratings

'''

import swifter
import os
import re
print(os.getcwd())
from scipy import stats
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

new_path2 = os.path.join(os.getcwd(), "Wissam's Work", "research-master")

#sys.path.append(new_path)
sys.path.append(new_path2)

from helpers.dataset import get_config_data, load_ratings, load_training_ratings
from surprise import KNNWithMeans, Dataset, Reader
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# A reader is still needed but only the rating_scale param is requiered.
# first of all rmi is 0.5 not 1
#r_min = 1
#r_min = 0.5
#r_max = 5
CHANGE_DATA = False # The split to train the model, will be done only one time, and if it's needed, as we will need the files to use with the rest of the codes
batch_size = 50000  


def load_data():
    # The columns must correspond to user id, item id and ratings (in that order).
    if CHANGE_DATA:
        dataset_path = get_config_data()['dataset']
        dataset_dir = get_config_data()['dataset_dir']
        dataset_name = get_config_data()['dataset_name']
        ratings_df = load_ratings(dataset_path)[['userId','movieId','rating']].rename({'movieId': 'itemId'}, axis=1)
        # Data is for training
        
        train, test = train_test_split(ratings_df, test_size=0.2, random_state=42)
        train.to_csv(dataset_dir+'/'+ dataset_name +'/train_data.csv', index=False)
        test.to_csv(dataset_dir+'/'+ dataset_name +'/ratings_test.csv', index=False)
        # We will split them and save the split in a directory specific for the data
        return train, test
    else:
        dataset_path = get_config_data()['dataset']
        dataset_dir = get_config_data()['dataset_dir']
        dataset_name = get_config_data()['dataset_name']
        test_df = load_ratings(dataset_path)[['userId','itemId','rating']]
        train_df = load_training_ratings(dataset_path)[['userId','itemId','rating']]
        return train_df, test_df
    

def compute_prediction(userId, itemId, rating,algo):
    pred = algo.predict(userId, itemId, r_ui=rating, verbose=True)

    return pred


def compute_prediction_paper_algorithm(userId, itemId, rating,ratings,similarity,algo):
    # rating should have the current rating we want to target
    # ratings is the array
    # similarity is an array as well
    pred = algo.predict(userId, itemId, r_ui=rating, verbose=True)

    return pred

def compute_noise(userId, itemId, rating, prediction, r_max, r_min, th):

    match = re.search(r'est = ([0-9]+\.[0-9]+)', prediction)

    if match:
        estimate = match.group(1)
        #print("estimate " + estimate)
        pred = estimate[:4]
        coherence = abs(rating - float(pred))/(r_max - r_min)

        if coherence > th:
            noise = 1
        else:
            noise = 0

        return noise
    else:
        
        raise Exception("Estimate not found")
    
def compute_pearson_similarity_matrix(users, items, ratings):

    return stats.pearsonr(users, items, ratings,method=None, axis=1).statistic



##  Here is our Main
def main():
    train_data_df, test_data_df = load_data() # Rating_df is the testing dataset, I've named it this way to ensure that the rest of the code will stay valid
    r_max = np.max(test_data_df['rating'])
    r_min = np.min(test_data_df['rating'])
    reader = Reader(rating_scale=(r_min,r_max))
    train_data = Dataset.load_from_df(train_data_df, reader)
    test_data = Dataset.load_from_df(test_data_df, reader)


    # Threshold is valid, it gives 100% increase in the number of good predictions
    th = 0.05 # MAE rnd threshold used in coherence formula

    print(os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf3_afterSplit_3.csv'))
    if(not os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf3_afterSplit_3.csv')):
        # Retrieve the trainset.
        # In the paper, it is explicitly said not to train with KNearest
        trainset = train_data.build_full_trainset()
        print((trainset))
    
        #### ratings_df and data are different representations of the same underlying data, but they serve different purposes and are structured differently
        #### Anon: In the paper, it is explicitly said not to train with KNearest

        # Build an algorithm, and train it.

        algo = knn(k=35, sim_options={'name': 'pearson_baseline',
                                                'user_based': True  # compute  similarities between users
                                                 })
        algo.fit(trainset)
        num_batches = len(test_data_df) // batch_size + 1
        appended_dataframes = []
        for i in range(num_batches):
            batch = test_data_df[i * batch_size: (i + 1) * batch_size]
            batch['prediction'] = batch.swifter.apply(lambda x: compute_prediction(x.userId, x.itemId, x.rating, algo), axis=1)
            appended_dataframes.append(batch)
        #### Anon:
        #### TRAINING SET = PREDICTION SET -> OVERFITTING, what we should do, is train on dataset, then, test on another, That shares items and users
        #### But Training data will affect either way what we have as a result, so I am not sure, if we can take this algrithm in the benchmark
        #### I think the best way, is to, maybe, maybe train on ml-100k, after making sure it has the same users and ratings,
        #### As input, we will give the algorithm, itemid and userid
        #### Anon3: Noisy rating isn't fixed, so it will never be able to accurately get a pattern to be able to predict well, so by default
        #### We won't have noisy ratings
        test_df = pd.concat(appended_dataframes)
        print("test_df['userId']" + str(test_df['userId']))
        print("test_df['itemId']" + str(test_df['itemId']))
        print("test_df['rating']" + str(test_df['rating']))


        # test_df['prediction'] = test_df.swifter.apply \
        #     (lambda x: compute_prediction(x.userId, x.itemId, x.rating,algo), axis=1)
        print("creating")
        test_df.to_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf3_afterSplit_3.csv')

    else:
        print("reading")
        test_df = pd.read_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf3_afterSplit_3.csv')

    appended_dataframes = []
    splits = np.array_split(test_df, len(test_df)/18000)

    for split in splits:
        split['coherence'] = split.swifter.apply \
                (lambda x: compute_noise(x.userId, x.itemId, x.rating, x.prediction,r_max,r_min, th), axis=1) # axis 1 is horizontal
        
        appended_dataframes.append(split)

    noise_df = pd.concat(appended_dataframes)
    noise_df.to_csv(r'NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf3_afterSplit_3.csv', index=False)



if __name__ == '__main__':
    main()
