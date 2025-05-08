import time
import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
import time

import re
print(os.getcwd())
import sys
new_path2 = os.path.join(os.getcwd(), "Wissam's Work", "research-master")
sys.path.append(new_path2)

from helpers.dataset import get_config_data, load_items, load_ratings
#  from helpers.dataset import get_config_data, load_ratings -- hidden to run on Anon's PC
# (23)	Correcting noisy ratings in collaborative recommender systems - ScienceDirect

class NoiseFilter1:
   
    def get_dataset_with_noise(self, ratings_df):

        print("Loading Noise Filter 1")
        start = time.time()

        # main algo variable: 
        # These are the old values, set by Wissam, on the scale of rating-range = {1-5}
        r_max = np.max(ratings_df['rating'])
        r_min = np.min(ratings_df['rating'])
        #k = 2 
        #v = 4
        k = r_min + round(1/3 * (r_max - r_min))
        v = r_max - round(1/3 * (r_max - r_min))
        print("k " + str(k))
        print("v " + str(v))

        # New Values, calculated for the rating-range={0.5,5}   
        #k=2.5
        #v=3
        user_groups_dict = {}
        item_groups_dict = {}
        noise_dict = {}

        # load dataset
        dataset = ratings_df
        
        # group ratings
        conditions = [
            (dataset['rating'] < k),
            ((dataset['rating'] >= k) & (dataset['rating'] < v)),
            (dataset['rating'] >= v)
            ]
        values = ['Wu', 'Au', 'Su']
        dataset['rating_group'] = np.select(conditions, values)

        # group users
        dataset_dict = dataset.groupby(['userId','rating_group']).size().unstack().fillna(0).to_dict('index')

        for i in dataset_dict:
            value = dataset_dict[i]
            if(value['Wu'] >= (value['Au'] + value['Su'])):
                user_groups_dict[i] = "Critical"
            elif(value['Au'] >= (value['Wu'] + value['Su'])):
                user_groups_dict[i] = "Average"
            elif(value['Su'] >= (value['Wu'] + value['Au'])):
                user_groups_dict[i] = "Benevolent"
            else:
                user_groups_dict[i] = "Variable"

        # item users
        item_groups_dict1 = dataset.groupby(['itemId','rating_group']).size().unstack().fillna(0).to_dict('index')

        # # # Anon ANALYSIS ON THE CODE:
        # # # Here Anon2 is trying to classify the Wu _> weakly user
        # # # And it looks different from what ze have in the algorithm in the paper, in the paper we compare against KU VU
  
        # RESULT -> V=K, hence Ai/Wi/Si = Ai/Wi/Si

        
        # This is correct for classification
        for i in item_groups_dict1:
            value = item_groups_dict1[i]
            if(value['Wu'] >= (value['Au'] + value['Su'])):
                item_groups_dict[i] = "Weakly-preferred"
            elif(value['Au'] >= (value['Wu'] + value['Su'])):
                item_groups_dict[i] = "Averagely-preferred"
            elif(value['Su'] >= (value['Wu'] + value['Au'])):
                item_groups_dict[i] = "Strongly-preferred"
            else:
                item_groups_dict[i] = "Variably-preferred"

        # after grouping, join all the dfs together with the initial dataset file
        user_groups_df = pd.DataFrame.from_dict(user_groups_dict, orient='index', columns=['user_cat']).reset_index().rename({'index': 'userId'}, axis=1)
        item_groups_df = pd.DataFrame.from_dict(item_groups_dict, orient='index', columns=['item_cat']).reset_index().rename({'index': 'itemId'}, axis=1)


        df_1 = user_groups_df.set_index('userId').\
                    join(dataset.set_index('userId'), how='left').reset_index()
        ratings_groups_df = df_1.set_index('itemId').\
                    join(item_groups_df.set_index('itemId'), how='left').reset_index()

        # apply the noise protocol to filter out noisy ratings
        noise_df = ratings_groups_df.set_index(['userId','itemId'])
        noise_df_dict = noise_df.to_dict('index')

       # Correct Loop
        for i in noise_df_dict:
            value = noise_df_dict[i]
            if( 
                (value['user_cat'] == "Critical" and value['item_cat'] == "Weakly-preferred" and value['rating'] >= k) \
                or (value['user_cat'] == "Average" and value['item_cat'] == "Averagely-preferred" and (v <= value['rating'] < k)) \
                or (value['user_cat'] == "Benevolent" and value['item_cat'] == "Strongly-preferred" and value['rating'] < v)
            ):
                noise_dict[i] = 1
            else:
                noise_dict[i] = 0

        noise_df = pd.DataFrame(noise_dict.values(), index=pd.MultiIndex.from_tuples(noise_dict.keys()), columns=['isNoisy']) \
            .reset_index() \
            .rename({'level_0': 'userId', 'level_1': 'itemId'}, axis=1)

        noise_df_final = ratings_groups_df.merge(noise_df, how='inner', on=['itemId', 'userId'])

        print("\nTime taken: " + str(time.time() - start) + "\nDataset size: " + str(len(dataset)))

        return noise_df_final
        # end

def main():
    nf1 = NoiseFilter1()
    dataset_path = get_config_data()['dataset']
    if get_config_data()['split_data_is_generated']:
        ratings_df = load_ratings(dataset_path)[['userId','itemId','rating']]
    else:
        ratings_df = load_ratings(dataset_path)[['userId','movieId','rating']].rename({'movieId': 'itemId'}, axis=1)


    ratings_df_noise = nf1.get_dataset_with_noise(ratings_df)
    ratings_df_noise.to_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf1_aftersplit.csv', index=False)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time
    print(f"main() took {elapsed_time:.4f} seconds to load")

######## CUSTOM ########
# custom script for a couple of users for the 20m dataset:
# example_users = noise_df_final[(noise_df_final['userId'] == 5155) \
#                                     | (noise_df_final['userId'] == 138435) \
#                                     | (noise_df_final['userId'] == 138235) \
#                                     | (noise_df_final['userId'] == 578) \
#                                     | (noise_df_final['userId'] == 137976) \
#                                 ]
# example_users['timestamp'] = example_users['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())
# example_users.to_csv('custom-20m')

