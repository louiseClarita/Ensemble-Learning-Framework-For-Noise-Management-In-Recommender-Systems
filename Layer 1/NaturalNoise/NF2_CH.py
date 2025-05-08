
r'''
    Date: 7/5/2020

    Natural Noise filter class
    Papers: 
            13 - A Novel Framework to Process the Quantity and Quality of User Behavior Data in Recommender Systems
    use it to apply noise filtering to a certain dataset of ratings
               Locally Saved:  "C:\Users\Pc\Desktop\CH\Lebanese University\Research\Dr Jacques Bou Abdo\Recommender  System\0 - Papers\NF2 - Novel Framework to Process the Quantity....pdf"


    Description: 
        1- we measure the coherence of a user c(u)
        2- use the coherence to categorize users into heavy and easy groups
        3- use the RND formula to determine the rating noise degree using a threshold v
                v is dependant on the user group (heavy, medium, or light)'''

import pandas as pd
import numpy as np
import swifter #  UserWarning: You are using pyarrow version 11.0.0 which is known to be insecure. See https://www.cve.org/CVERecord?id=CVE-2023-47248 for further details. Please upgrade to pyarrow>=14.0.1 or install pyarrow-hotfix to patch your current version.
               #  warnings.warn(
import math
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor


new_path2 = os.path.join(os.getcwd(), "Wissam's Work", "research-master","NaturalNoise")
print(os.getcwd())
#sys.path.append(new_path)
sys.path.append(new_path2)
from pathlib import Path
from helpers.dataset import get_config_data, load_ratings, load_training_ratings
# Join the current working directory with the new path
# new_path = os.path.join(os.getcwd(), "Wissam's Work", "research-master", "NaturalNoise")
# new_path2 = os.path.join(os.getcwd(), "Wissam's Work", "research-master")

# #sys.path.append(new_path)
# sys.path.append(new_path)
print(os.getcwd())
from NoiseFilter2.Helpers import Helpers
from helpers.dataset import get_config_data, load_ratings, load_items
from NoiseFilter2.Coherence_CH import Coherence

'''
    Find the rating noise degree of every rating in the dataset.
    The user groups is used to determine the threshold value that's set in the rnd formula:
        th = 0.075 for heavy and medium users
        th = 0.05 for light user groups
'''
class Noise:

    def __init__(self, ratings, movies):
        genres_list = Helpers().get_genres(movies) 
        # 1- call Coherence class to group users in the dataset
        if(not os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_groups_protocol_2.csv')):
            print("computing user groups...")
            t0 = time.time()
            ratings_detailed = ratings[['userId', 'itemId', 'rating']].merge(movies[['itemId', 'genres']], on='itemId', how='left')
            item_features = item_features = ratings_detailed.set_index('itemId')['genres'].to_dict()
            users_categories_details = Coherence().group_users(ratings_detailed, movies, item_features)
            users_categories = users_categories_details[['userId', 'user_group', 'coherence']]
            #= pd.DataFrame(users_categories_details, columns=['userId', 'user_group','coherence']).reset_index()
            #users_categories_details.columns = ['userId', 'Group','coherence']

            #users_categories = users_categories_details[['userId', 'user_group','coherence']]
            t1 = time.time()
            print('time take', t1-t0)
        else:
            print("loading pre-computed groups for: " + get_config_data()['dataset_name'])
            users_categories_details = pd.read_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_groups_protocol_3.csv')
            #users_categories = users_categories_details[['userId', 'user_group','coherence']]

        ## 2- Calculate the rating noise degree for every rating in the dataset
        if(not os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_ratings_rnd_protocol_3.csv')):
            print("computing noise in the dataset...")
            #self.evaluate_noise(ratings, movies, genres_list, users_categories)
            ratings_detailed = ratings.merge(users_categories.reset_index(drop=True),on='userId',how="left")
            self.process_noise(ratings_detailed, movies, genres_list, users_categories)
        else:
            print("noise already calculated for the dataset: " + get_config_data()['dataset_name'])

    # Define your class or function
    def process_noise(self, ratings, movies, genres_list, users_categories):
        dataset_name = get_config_data()['dataset_name']
        output_file = Path('NaturalNoise/output') / f'{dataset_name}_ratings_rnd_protocol_3.csv'
        
        if not output_file.exists():
            print("Computing noise in the dataset...")
            # Use ThreadPoolExecutor to run evaluate_noise in a separate thread
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self.evaluate_noise, ratings, movies, genres_list, users_categories)
                result = future.result()  # This will block until the result is available
        else:
            print(f"Noise already calculated for the dataset: {dataset_name}")

        
    def evaluate_noise(self, ratings_df, movies_df, genres_list, users_categories):
        user_features_dict = {}
        appended_dataframes = []
        user_ids = ratings_df['userId'].unique().tolist()


        for user_id in user_ids:
            target_user = ratings_df.loc[ratings_df['userId'] == user_id]
            target_user_full = target_user.set_index('itemId').join(movies_df.set_index('itemId'), how='left').reset_index()
            #target_user_full = target_user_full.set_index('userId').join(users_categories.set_index('userId'), how='left').reset_index()
            target_user_full = target_user_full.set_index('userId').reset_index()

            #Ensure genres column contains lists:
            target_user_full['genres'] = target_user_full['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else (x if isinstance(x, list) else []))


            user_features_dict = {} # Reset for each user to prevent cross-user feature averaging

            for genre in genres_list:
                target_feature = target_user_full[target_user_full['genres'].apply(lambda x: genre in x)]
                if not target_feature.empty:
                    user_feature_ratings = target_feature['rating'].tolist()
                    feature_avg_rating = sum(user_feature_ratings) / len(user_feature_ratings)
                    user_features_dict[(user_id, genre)] = feature_avg_rating
            target_user_full['RND'] = target_user_full.swifter.apply(lambda x: self.compute_rnd(user_id, x['rating'], x['genres'], x['user_group'], user_features_dict), axis=1)
            target_user_full['thresh'] = target_user_full['user_group'].apply(
                                            lambda group: 0.075 if group in ['HEUG', 'HDUG', 'MEUG', 'MDUG'] else 0.05
                                        )
            target_user_full['isNoisy'] = target_user_full.apply(
                                                                    lambda x: 1 if x['RND'] > x['thresh'] else 0, axis=1
                                                                )

            appended_dataframes.append(target_user_full)


        noise_df = pd.concat(appended_dataframes)
        noise_df.to_csv(r'NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf2_aftersplit3.csv', index=False)
        return noise_df


    #   def compute_rnd(self, rating, genres, user_group, user_features_dict):
    def compute_rnd(self, user_id, rating, item_features, user_group, avg_ratings_by_feature):
        """Computes the Rating Noisy Degree (RND)."""
        if user_group in ['HEUG', 'HDUG']:
            rnd_threshold = 0.075
        else:
            rnd_threshold = 0.05

        try:
            genres_list = item_features.split('|') if isinstance(item_features, str) else item_features
        except:  # Handle cases where item_features is not a string or list
            genres_list = []
            print(f"Warning: Invalid item_features format for user {user_id}, rating {rating}. Skipping RND calculation.")

        total_relative_deviation = 0
        num_features = len(genres_list)

        if num_features > 0 and not '(no genres listed)' in genres_list:
            for feature in genres_list:
                try:
                    avg_rating = avg_ratings_by_feature[(user_id, feature)]
                    relative_deviation = abs(rating - avg_rating) / avg_rating
                    total_relative_deviation += relative_deviation
                except KeyError:
                    print(f"Warning: Average rating missing for user {user_id}, feature {feature}. Skipping this feature for RND calculation.")
            rnd = total_relative_deviation / num_features
            return rnd if rnd > rnd_threshold else 0 #Only return RND if it exceeds threshold for that rating.

        else:
            return 0



def main():
    dataset_path = get_config_data()['dataset']

    print(dataset_path)
    if get_config_data()['split_data_is_generated']:
        ratings = load_ratings(dataset_path)[['userId','itemId','rating']]
        items = load_items(dataset_path)[['movieId','title','genres']].rename({'movieId': 'itemId'}, axis=1)
    else:
        ratings = load_ratings(dataset_path)[['userId','movieId','rating']].rename({'movieId': 'itemId'}, axis=1)
        items = load_items(dataset_path)[['movieId','title','genres']].rename({'movieId': 'itemId'}, axis=1)
    
# Calculate unique users and items
    num_unique_users = ratings['userId'].nunique()
    num_unique_items = ratings['itemId'].nunique()

    # Get the number of rows
    num_ratings_rows = ratings.shape[0]
    num_items_rows = items.shape[0]

    # Print the resultsF
    print(f'Number of unique users: {num_unique_users}')
    print(f'Number of unique items: {num_unique_items}')
    print(f'Number of rows in ratings: {num_ratings_rows}')
    print(f'Number of rows in items: {num_items_rows}')
    print(ratings, items)
    Noise(ratings, items)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time
    print(f"main() took {elapsed_time:.4f} seconds to load")