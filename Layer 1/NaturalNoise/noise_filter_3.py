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
from annoy import AnnoyIndex
from sklearn.model_selection import train_test_split
from helpers.dataset import get_config_data, load_ratings, load_training_ratings
from surprise import KNNWithMeans, Dataset, Reader

import pandas as pd
import numpy as np

os.environ["MODIN_ENGINE"] = "dask"  # Can also use "ray"


# A reader is still needed but only the rating_scale param is requiered.
r_min = 1
r_max = 5
th = 0.4 # MAE rnd threshold used in coherence formula
CHANGE_DATA = False # The split to train the model, will be done only one time, and if it's needed, as we will need the files to use with the rest of the codes
reader = Reader(rating_scale=(r_min,r_max))

batch_size = 50000  # Adjust batch size based on your system's capabilities




def incremental_fit(algo, data, batch_size=10000):
    num_batches = len(data) // batch_size + 1
    for i in range(num_batches):
        batch_data = data.iloc[i*batch_size:(i+1)*batch_size]
        reader = Reader(rating_scale=(data['rating'].min(), data['rating'].max()))
        batch_dataset = Dataset.load_from_df(batch_data[['userId', 'itemId', 'rating']], reader)
        batch_trainset = batch_dataset.build_full_trainset()
        algo.fit(batch_trainset)
        print(f'Batch {i+1}/{num_batches} completed')


def compute_prediction(userId, itemId, rating, algo):
    pred = algo.predict(userId, itemId, r_ui=rating, verbose=True)
    print("pred "+ str(pred))
    return pred

def compute_noise(userId, itemId, rating, prediction):
    # print("prediction " + str(prediction))
    # estimate = prediction[54:]
    # print("estimate " + str(estimate))
    # pred = estimate[:4]
    # print(str(pred))
    # Unfortunately, While testing the code once more, this method of slicing didnt work; I checked, And it seems it is prone to errors
    # Hence why I will use regex matching

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
        test.to_csv(dataset_dir+'/'+ dataset_name +'/rating.csv', index=False)
        # We will split them and save the split in a directory specific for the data
        return train, test
    else:
        dataset_path = get_config_data()['dataset']
        dataset_dir = get_config_data()['dataset_dir']
        dataset_name = get_config_data()['dataset_name']
        test_df = load_ratings(dataset_path)[['userId','movieId','rating']].rename({'movieId': 'itemId'}, axis=1)
        train_df = load_training_ratings(dataset_path)[['userId','itemId','rating']].rename({'movieId': 'itemId'}, axis=1)
        return train_df, test_df
train_data_df, test_data_df = load_data()
train_data = Dataset.load_from_df(train_data_df, reader)
test_data = Dataset.load_from_df(test_data_df, reader)


print(os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_predictions_nf3_2.csv'))
if(not os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_predictions_nf3_2.csv')):
    # Retrieve the trainset.
    trainset = train_data.build_full_trainset()

    # Build an algorithm, and train it.
    algo = KNNWithMeans()
    #         #algo = KNNWithMeans(k=35, sim_options={'name': 'pearson_baseline',
    #                                        'user_based': True  # compute  similarities between users
    #                                        })
    #algo.fit(trainset)
    incremental_fit(algo, train_data_df, batch_size=10000)

    #### Anon:
    #### TRAINING SET = PREDICTION SET -> OVERFITTING, what we should do, is train on dataset, then, test on another, That shares items and users
    #### But Training data will affect either way what we have as a result, so I am not sure, if we can take this algrithm in the benchmark
    #### As input, we will give the algorithm, itemid and userid
    ####
    num_batches = len(test_data_df) // batch_size + 1
    appended_dataframes = []
    for i in range(num_batches):
        batch = test_data_df[i * batch_size: (i + 1) * batch_size]
        batch['prediction'] = batch.swifter.apply(lambda x: compute_prediction(x.userId, x.itemId, x.rating, algo), axis=1)
        appended_dataframes.append(batch)

    test_df = pd.concat(appended_dataframes)
    # test_data_df['prediction'] = test_data_df.swifter.apply \
    #     (lambda x: compute_prediction(x.userId, x.itemId, x.rating), axis=1)
    print("creating")
    test_data_df.to_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_predictions_nf3_2.csv')

else:
    print("reading")
    ratings_df = pd.read_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_predictions_nf3_2.csv')

appended_dataframes = []
splits = np.array_split(ratings_df, len(ratings_df)/18000)

for split in splits:
    split['coherence'] = split.swifter.apply \
            (lambda x: compute_noise(x.userId, x.itemId, x.rating, x.prediction), axis=1) # axis 1 is horizontal
    
    appended_dataframes.append(split)

noise_df = pd.concat(appended_dataframes)
noise_df.to_csv(r'NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf3_2.csv', index=False)
