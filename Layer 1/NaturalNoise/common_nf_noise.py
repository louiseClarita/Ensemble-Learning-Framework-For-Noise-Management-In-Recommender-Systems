'''
This script was to load the data from the three noise algorithms and remove the common noisy ratings between the 3 filters.
'''
import pandas as pd
from helpers.dataset import get_config_data, load_ratings


# load the first batch of ratings that were common noise between the three nfs
common_noise_location = 'E:/ProgramData/Dropbox/Research/Publications/Journal 3/to_be_removed_ml_latest_small_round_3.csv'
common_noise = pd.read_csv(common_noise_location).rename({'movieId': 'itemId'}, axis=1)

# load the main dataset
dataset_path = get_config_data()['dataset']
ratings_df = load_ratings(dataset_path)[['userId','movieId','rating']].rename({'movieId': 'itemId'}, axis=1)


# remove the common noise items from the main dataset
ratings_without_common_noise = pd.merge(ratings_df, common_noise, how='outer', indicator=True)\
    .query('_merge == "left_only"')\
    .drop('_merge', 1)

ratings_without_common_noise.to_csv('NaturalNoise/output/ratings_1.csv')