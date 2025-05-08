import pandas as pd
import numpy as np
import swifter
import math

from helpers.dataset import get_config_data

'''
    Find the users groups (categories) based on their coherence values values in the dataset.
    There are 3 main groups based on the user total ratings: Light, Medium, Light
    Then in each group and based on the user coherence, the users are split into two: Low coherence and High coherence
        By that, the six user groups are formed: HEUG, HDUG, MEUG, MDUG, LEUG, LDUG 
'''
class Coherence:

    def compute_user_groups(self, ratings_df, movies_df, genres_list):
        
        # perform the coherence euqation to classify all users in the dataset
        user_coherence_dict = {}
        # get all users (ids) from the ratings dataframe
        user_ids = ratings_df.userId.drop_duplicates().tolist()

        for user_id in user_ids: 
            user = ratings_df.loc[ratings_df['userId'] == user_id]
            # used set_index for faster merge of dfs
            user_full_set = user.set_index('itemId').\
                join(movies_df.set_index('itemId'), how='left').reset_index()

            # here we calculate the average Genre value to achieve the ultimate coherence value
            # np.vectorize (fast dataframe search method)
            # This is a wrapper around a loop, but with lesser overhead than most pandas str methods.
            f = np.vectorize(lambda haystack, needle: needle in haystack)

            # loop the user's items and calculate the feature deviation of every genre. Then use them to calculate the coherence
            user_coherence = 0
            for genre in genres_list:
                # find all rows that conaint the Genre "Action", "Drama", etc.
                target_feature = user_full_set[f(user_full_set['genres'], genre)]

                # check whether the genre exists in at least one item row to avoid division by zero:
                if not target_feature.empty:
                    user_feature_ratings = target_feature.rating.tolist()
                    feature_avg_rating = sum(user_feature_ratings) / len(user_feature_ratings)

                    feature_deviation = math.sqrt(sum([(number - feature_avg_rating)**2 for number in user_feature_ratings])) 
                    user_coherence += feature_deviation
            user_coherence = -user_coherence
            user_coherence_dict[user_id] = user_coherence
            # break #remove this!

        users_coherence_df = pd.DataFrame.from_dict(user_coherence_dict, orient='index').reset_index()
        users_coherence_df.columns = ['userId', 'coherence']
        # let's add the total ratings per user to the users_coherece_df:
        total_ratings_per_user = ratings_df.groupby(['userId']).size().reset_index(name='total_ratings')
        users_coherence_df2 = users_coherence_df.set_index('userId').\
            join(total_ratings_per_user.set_index('userId'), how='left')

        # Add the sum of total ratings and the total coherece as a constant column for easy user grouping operations
        # User the swifter library to apply the grouping formula in a fast way
        total_ratings_list = users_coherence_df2.total_ratings.tolist()
        users_coherence_df2['max_total_ratings'] = max(total_ratings_list)
        users_coherence_df2['min_total_ratings'] = min(total_ratings_list)

        total_coherence_list = users_coherence_df2.coherence.tolist()
        users_coherence_df2['max_coherence'] = max(total_coherence_list)
        users_coherence_df2['min_coherence'] = min(total_coherence_list)

        users_coherence_df2['user_group'] = users_coherence_df2.swifter.apply \
            (lambda x: self.compute_user_group(x.coherence, x.total_ratings, x.max_total_ratings,\
                x.min_total_ratings, x.max_coherence, x.min_coherence), axis=1)

        users_coherence_df2.to_csv(r'NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_groups_protocol_2.csv')

        return users_coherence_df2

    def compute_user_group(self, coherence, total_ratings, max_total_ratings, 
                                min_total_ratings, max_coherence, min_coherence):
        mr = (max_total_ratings/3)*2
        lr = max_total_ratings/3
        mc = min_coherence/2

        # check if the user is in the high ratings group
        if(mr <= total_ratings <= max_total_ratings):
            if(mc <= coherence <= max_coherence):
                group = "HEUG"
            else:
                group = "HDUG"
        # check if the user is in the medium ratings group
        elif(lr <= total_ratings <= mr):
            if(mc <= coherence <= max_coherence):
                group = "MEUG"
            else:
                group = "MDUG"
        # if the user is in the low ratings group 
        else:
            if(mc <= coherence <= max_coherence):
                group = "LEUG"
            else:
                group = "LDUG"
        
        return group