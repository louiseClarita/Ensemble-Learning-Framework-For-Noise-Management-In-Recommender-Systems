from surprise import SVD, KNNWithMeans, NMF
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from datetime import datetime as dt
import pandas as pd
import numpy as np

## import from custom scripts
from helpers.dataset import get_config_data, load_ratings
from LocalImpact.custom_train_test_split import mod_train_test_split
from LocalImpact.compute_preceision_recall import compute_prec_rec
from LocalImpact.compute_accuracy_at_user import compute_mae_at_user
from LocalImpact.compute_utility_at_user import compute_ndcg_at_user
from LocalImpact.get_top_k_neighbors import get_top_k_neighbors
##


dataset_path = get_config_data()['dataset']
raw_ratings_df = load_ratings(dataset_path)[['userId','movieId','rating','timestamp']].rename({'movieId': 'itemId'}, axis=1)
raw_ratings_df['date'] = raw_ratings_df['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())

### testing with recommender algorithms
data = Dataset.load_from_df(
                raw_ratings_df[['userId','itemId','rating']],
                Reader(rating_scale=(1,5))
            )
# sample random trainset and testset (default call using Surprise library train_test_split_method)
    # (Default - surpriselib) trainset, testset = train_test_split(data, test_size=.15, shuffle=True)
trainset, testset = mod_train_test_split(raw_ratings_df, remove_ratings=True)

# load a recommender algorithm: SVD, NMF or KNNWithMeans algorithm
sim_options = {
    'name': 'pearson',
    'user_based': True
    }
algo = KNNWithMeans(sim_options=sim_options)

# train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
neighbors = get_top_k_neighbors(raw_ratings_df, algo, k=10)
predictions = algo.test(testset)

# compute MAE and RMSE
accuracy.mae(predictions)
accuracy.rmse(predictions)
accuracy.fcp(predictions)
# get the accuracy at the neighborhood level (user-level)
compute_mae_at_user(predictions, neighbors)
compute_ndcg_at_user(testset, predictions, neighbors)

# 5-fold cross validation (to avoid bias)
# cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=5, verbose=True)

# comput Precision and Recall
# compute_prec_rec(trainset, testset, data, algo, predictions)
### end testing