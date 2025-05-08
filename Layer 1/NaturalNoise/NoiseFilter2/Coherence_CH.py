import pandas as pd
import numpy as np
from helpers.dataset import get_config_data

class Coherence:
            
    def compute_coherence(self, user_ratings, item_features, avg_ratings_by_feature):
        user_id = list(user_ratings.keys())[0] if user_ratings else None  # Get user_id from the dictionary

        if not user_ratings or not item_features or user_id is None:
            return 0

        total_coherence = 0
        total_items = len(user_ratings)

        if total_items > 0:
            unique_features = set()

            for item_id, rating in user_ratings.items():
                # Get the features (genres) for the current item_id
                features = item_features.get(item_id, "")
                
                # Split and clean genres
                if features:
                    genres = [genre.strip() for genre in features.split('|')]
                    unique_features.update(genres)

            for feature in unique_features:
                if '(no genres listed)' not in feature:
                    items_with_feature = [
                        item_id for item_id, features in item_features.items()
                        if feature in features and item_id in user_ratings
                    ]

                    if items_with_feature:
                        # Calculate the sum of squared deviations for this feature
                        feature_sum_squared_dev = sum(
                            (user_ratings[item_id] - avg_ratings_by_feature.get((user_id, feature), 0)) ** 2
                            for item_id in items_with_feature
                        )
                        total_coherence += np.sqrt(feature_sum_squared_dev)

            coherence = total_coherence / np.sqrt(total_items)
            return coherence
        else:
            return 0

        


    # def compute_coherence(user_ratings, item_features, avg_ratings_by_feature):
    #     """Computes user coherence as defined in the research paper."""
    #     user_id = list(user_ratings.keys())[0] if user_ratings else None
    #     if not user_ratings or not item_features or user_id is None:
    #         return 0
    #     total_coherence = 0
    #     total_items = len(user_ratings)
    #     if total_items > 0:
    #         unique_features = set()
    #         for item_id, rating in user_ratings.items():
    #             unique_features.update(item_features.get(item_id, []))
    #         for feature in unique_features:
    #             items_with_feature = [item_id for item_id, features in item_features.items() if feature in features and item_id in user_ratings]
    #             if items_with_feature:
    #                 feature_sum_squared_dev = sum((user_ratings[item_id] - avg_ratings_by_feature[(user_id, feature)])**2 for item_id in items_with_feature)
    #                 total_coherence += np.sqrt(feature_sum_squared_dev)
    #         coherence = total_coherence / np.sqrt(total_items)
    #         return coherence
    #     else:
    #         return 0

    def assign_user_group(self,coherence, num_ratings, rating_quantiles):
        """Assigns users to groups based on coherence and number of ratings."""
        'this is errored as the coherence threshold is false'
        
        heavy_threshold = rating_quantiles[2] # 75th percentile
        medium_threshold = rating_quantiles[1] # 50th percentile
        coherence_quantiles = pd.Series(list(coherence.values())).quantile([0.5]).tolist() #example quantile
        coherence_threshold = coherence_quantiles[0]


        if num_ratings >= heavy_threshold:
            group = "HEUG" if coherence >= coherence_threshold else "HDUG"
        elif num_ratings >= medium_threshold:
            group = "MEUG" if coherence >= coherence_threshold else "MDUG"
        else:
            group = "LEUG" if coherence >= coherence_threshold else "LDUG"
        return group

    def group_users(self, ratings_df, movies_df, item_features):
        """Groups users based on coherence and rating count."""
        user_groups = {}
        
        # Step 1: Determine rating thresholds (heavy, medium, light)
        rating_counts = ratings_df['userId'].value_counts()
        rating_quantiles = rating_counts.quantile([0.25, 0.5, 0.75]).tolist()
        heavy_threshold = rating_quantiles[2]  # 75th percentile
        medium_threshold = rating_quantiles[1]  # 50th percentile

        # Step 2: Prepare the data for average ratings calculation
        ratings_df['genres'] = ratings_df['genres'].str.split('|')  # Split genres
        ratings_exploded = ratings_df.explode('genres')  # Explode into separate rows
        avg_ratings_by_feature = ratings_exploded.groupby(['userId', 'genres'])['rating'].mean().to_dict()

        # Optimize types to reduce memory usage
        ratings_df_small = ratings_df[['userId', 'itemId', 'rating']]
        ratings_df_small['userId'] = ratings_df_small['userId'].astype('int32')
        ratings_df_small['itemId'] = ratings_df_small['itemId'].astype('int32')
        ratings_df_small['rating'] = ratings_df_small['rating'].astype('float32')

        # Step 3: Calculate coherence scores for all users
        heavy_coherence_scores = []
        medium_coherence_scores = []
        light_coherence_scores = []

        user_coherence_data = {}  # To store coherence for each user

        for user_id in ratings_df_small['userId'].unique():
            # Get user ratings
            user_ratings = ratings_df_small[ratings_df_small['userId'] == user_id]
            
            # Create a dictionary for user ratings
            user_ratings_dict = dict(zip(user_ratings['itemId'], user_ratings['rating']))
            
            # Extract features for user's items
            user_item_features = {item_id: item_features[item_id] for item_id in user_ratings_dict}

            # Calculate coherence
            coherence = self.compute_coherence(user_ratings_dict, user_item_features, avg_ratings_by_feature)
            user_coherence_data[user_id] = (coherence, len(user_ratings_dict))

            # Classify into temporary groups to calculate thresholds
            num_ratings = len(user_ratings_dict)
            if num_ratings >= heavy_threshold:
                heavy_coherence_scores.append(coherence)
            elif num_ratings >= medium_threshold:
                medium_coherence_scores.append(coherence)
            else:
                light_coherence_scores.append(coherence)

        # Step 4: Determine median coherence thresholds for each group
        heavy_coherence_threshold = np.median(heavy_coherence_scores) if heavy_coherence_scores else 0
        medium_coherence_threshold = np.median(medium_coherence_scores) if medium_coherence_scores else 0
        light_coherence_threshold = np.median(light_coherence_scores) if light_coherence_scores else 0

        # Step 5: Assign users to final groups based on coherence and thresholds
        for user_id, (coherence, num_ratings) in user_coherence_data.items():
            if num_ratings >= heavy_threshold:
                group = "HEUG" if coherence >= heavy_coherence_threshold else "HDUG"
            elif num_ratings >= medium_threshold:
                group = "MEUG" if coherence >= medium_coherence_threshold else "MDUG"
            else:
                group = "LEUG" if coherence >= light_coherence_threshold else "LDUG"

            # Store final group and coherence
            user_groups[user_id] = (group, coherence)

        # Convert the results to a DataFrame
        user_groups_df = pd.DataFrame.from_dict(user_groups, orient='index', columns=['user_group', 'coherence'])
        user_groups_df.reset_index(inplace=True)  # This will add the index (which is userId) as a column
        user_groups_df.rename(columns={'index': 'userId'}, inplace=True)   # Set the index name to 'userId'
        # Save to CSV
        user_groups_df.to_csv(r'NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_groups_protocol_3.csv')

        return user_groups_df

    # def group_users(self, ratings_df, movies_df, item_features):
    #     """Groups users based on coherence and rating count."""
    #     user_groups = {}
    #     rating_counts = ratings_df['userId'].value_counts()
    #     rating_quantiles = rating_counts.quantile([0.25, 0.5, 0.75]).tolist()

    #     # Merge only the necessary columns and avoid creating a large intermediate DataFrame.
    #     ratings_df = ratings_df[['userId', 'itemId', 'rating']].merge(
    #         movies_df[['itemId', 'genres']], on='itemId', how='left')

    #     # Precompute average ratings per user and feature:
    #     avg_ratings_by_feature = ratings_df.groupby(['userId', 'genres'])['rating'].mean().to_dict()

    #     if (ratings_df['itemId'] == 8108).any():
    #         print("ItemId 8108 exists in ratings_df")
    #     else:
    #         print("ItemId 8108 does not exist in ratings_df")
                
    #     # Use a generator to handle large data efficiently:
    #     for user_id, user_ratings in ratings_df.groupby('userId')['rating']:
    #         user_ratings_dict = user_ratings.to_dict()
    #         # Extract features for user's items:
    #         print(user_ratings_dict)
    #         print(8108 in item_features)  # Check if 8108 is in item_features
    #         print(item_features.get(8108))  # See what value it might have
    #         user_item_features = {item_id: item_features[item_id] for item_id in user_ratings_dict}
    #         for item_id in user_ratings_dict:
    #             print("Checking item_id:", item_id)  # Debug print
    #             if item_id not in item_features:
    #                 print(f"Warning: item_id {item_id} not found in item_features")
            
    #         coherence = self.compute_coherence(user_ratings_dict, user_item_features, avg_ratings_by_feature)
    #         num_ratings = len(user_ratings_dict)

    #         # Store coherence only if it's needed:
    #         user_groups[user_id] = [self.assign_user_group(coherence, num_ratings, rating_quantiles), coherence]

    #     # Convert to DataFrame efficiently:
    #     user_groups_df = pd.DataFrame.from_dict(user_groups, orient='index', columns=['User Group', 'Coherence'])
    #     user_groups_df.to_csv(f"NaturalNoise/output/{get_config_data()['dataset_name']}_user_groups_protocol_2.csv")

    #     return user_groups
