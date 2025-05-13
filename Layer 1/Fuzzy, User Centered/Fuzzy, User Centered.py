# This code follows this paper for https://sci-hub.se/10.1016/j.asoc.2015.10.060
# NF4
# Master 2 - Web Dev
# 
#



from joblib import Parallel, delayed


from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import skfuzzy.control as ctrl
import pandas as pd
import os
import sys
import time
print("os " + os.getcwd())
from collections import defaultdict
new_path2 = os.path.join(os.getcwd(),"..","Wissam's Work", "research-master", "NaturalNoise")
sys.path.append(new_path2)
print("os 2" + os.getcwd())
import dask.dataframe as dd
from helpers.dataset import get_config_data, load_ratings

# low = 0
# medium = 0
# high = 0
#print(os.getcwd())
#INPUT_PATH = "dataset/ml_latest_small/ratings.csv"
#ratings_df = pd.read_csv(INPUT_PATH) 








def fuzzify_rating(rating,x,low,medium,high):
    
    # membership_low = fuzz.interp_membership(x, low, rating)
    # membership_medium = fuzz.interp_membership(x, medium, rating)
    # membership_high = fuzz.interp_membership(x, high, rating)
    # return {'low': membership_low, 'medium': membership_medium, 'high': membership_high}
    #low = fuzz.trapmf(x, [r_min, r_min, r_min_end, r_min_outerbound])
    # 
    # low = fuzz.trapmf(x, [r_min, r_min, r_min_end, r_min_outerbound])

    membership_low = fuzz.trapmf(x, lowlist)[np.where(np.isclose(x,rating))[0]]
    membership_medium = fuzz.trapmf(x, mediumlist)[np.where(np.isclose(x,rating))[0]]
    membership_high = fuzz.trapmf(x, highlist)[np.where(np.isclose(x,rating))[0]]

    return {'low': membership_low[0], 'medium': membership_medium[0], 'high': membership_high[0]}


def fuzzify_rating_manual(rating):
    # Testng MANUAL RULES from the paper

    # LOW
    if 0 <= rating <= 1.5:
        membership_low = 1
    elif 1.5 <= rating <= 2:
        membership_low = -2*rating +4
    elif 2  <= rating <=5:
        membership_low = 0


    # MEDIUM
    if 0 <= rating <= 1.5:
        membership_medium = 0
    elif 1.5 <= rating <= 2:
        membership_medium = 2*rating -3
    elif 2 <= rating <=3:
        membership_medium = 1
    elif 3 <= rating <= 4:
        membership_medium = -rating + 4
    elif 4 <= rating <= 5:
        membership_medium = 0

    # HIGH
    if 0 <= rating <= 3:
        membership_high = 0
    elif 3 <= rating <= 4:
        membership_high = rating - 3
    elif 4 <= rating <=5:
        membership_high = 1  

    return {'low': membership_low, 'medium': membership_medium, 'high': membership_high}

def prefiltering(user_profiles, item_profiles, rating_profiles):
        elegible_ratings = []
        #Lambda1
        # Explanation for lamda Selection: Remark 3. Additional experiments were executed to empirically find the best value
        #  for the parameter δ1, concluding that for δ1 = 1 the proposal obtains a similar performance to the optimal value
        lambda1 = 1
        threshold = 0.01
        for rating_profile in rating_profiles:
                user_id = rating_profile['user_id']
                item_id = rating_profile['item_id']
                rating_membership = rating_profile['rating_membership']
                rating = rating_profile['rating']
                user_profile = user_profiles[user_id]
                item_profile = item_profiles[item_id]
                count = 0
                #P*Ru is user_profile
                #P*Ri is item_profile
                #S = {Low, Medium, High}
                manhattan_distance = abs(item_profile['low'] - user_profile['low'])
                + abs(item_profile['medium'] - user_profile['medium'])
                + abs(item_profile['high'] - user_profile['high'])
                is_variable_user = all(abs(val) < threshold for val in [user_profile['low'], user_profile['medium'], user_profile['high']])
                is_variable_item = all(abs(val) < threshold for val in [item_profile['low'], item_profile['medium'], item_profile['high']])

                if manhattan_distance < lambda1 and not is_variable_user and not is_variable_item:
                    elegible_ratings.append(rating_profile)
                    ## Added the variable condition equation 8 from the paper, if PRx = (0,0,0) then this profile is variable and isnt eligible for checking
                     # if the item isn't elegible, then, 
                    # it is dificult to know if for real it is noisy or no, 
                    # In this case we will set the noisy detection result to -1
                    # Theoretically it means, It doesnt fall under any set of S
                #print("manhattan_distance " + str(manhattan_distance))
                # if manhattan_distance < lambda1:
                #     elegible_ratings.append(rating_profile)
                #     # if the item isn't elegible, then, 
                #     # it is dificult to know if for real it is noisy or no, 
                #     # In this case we will set the noisy detection result to -1
                #     # Theoretically it means, It doesnt fall under any set of S
                else:
                    count = count + 1
        #print('count unelegibile '+ str(count))        
        return elegible_ratings



# Used for user and rating
def create_fuzzy_profileNOTUSED(ratings):
    # This was implemented for Sample data not from an excel file
    profiles = {}
    for user, user_ratings in ratings.items():
        profile = {'low': [], 'medium': [], 'high': []}
        for item, rating in user_ratings.items():
            fuzzified = fuzzify_rating(rating)
            fuzzifiedm = fuzzify_rating_manual(rating)
            if fuzzified != fuzzifiedm:
                print('they are not equal in create_fuzzy_profileNOTUSED')
            for label, membership in fuzzified.items():
                profile[label].append(membership)
        profiles[user] = {label: np.mean(memberships) for label, memberships in profile.items()}
    return profiles

def f1(x, k=0.35):
    if x > k:
        return (x - k) / (1 - k)
    else:
        return x
    
# For Rating profiles
def create_fuzzy_rating_profiles(df, x, low, medium, high):
    rating_profiles = []
    for i, row in df.iterrows():
        user_id = row['userId']
        item_id = row['movieId']
        rating = row['rating']
        # if  rating == 2.00 : 
        #     print('Printing out the fuzziness')
        # print(' current rating is :' + str(rating))
        rating_membership = fuzzify_rating(rating, x, low, medium, high)
        for label in ['low', 'medium', 'high']:
           rating_membership[label] = f1(rating_membership[label])
        # #  These lines were added to make sure the library and the manual fuzzy membership work the same the same way
        # fuzzifiedm = fuzzify_rating_manual(rating)
        # # # if  rating == 2.00 : 
        # # #     print('Printing out the fuzziness')
        # # #     print(' manual version ' + str(fuzzifiedm))
        # # #     print(' Fuzzy from the library' + str(rating_membership))
        # # if rating_membership != fuzzifiedm:
        # #         print('they are not equal in create_fuzzy_rating_profiles')
        # #         print(' manual version ' + str(fuzzifiedm))
        # #         print(' Fuzzy from the library' + str(rating_membership))
        rating_profiles.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating':rating,
            'rating_membership': rating_membership
        })
    
    return rating_profiles

def create_fuzzy_user_profile(rating_profiles,type):
    # type = user_id, item_id
    user_profiles = defaultdict(lambda: {'low': 0, 'medium': 0, 'high': 0, 'count': 0})
    for profile in rating_profiles:
        user_profiles = defaultdict(lambda: {'low': [], 'medium': [], 'high': [], 'count': 0})
    for profile in rating_profiles:
        user_id = profile[type]
        user_profiles[user_id]['low'].append(profile['rating_membership']['low'])
        user_profiles[user_id]['medium'].append(profile['rating_membership']['medium'])
        user_profiles[user_id]['high'].append(profile['rating_membership']['high'])
        user_profiles[user_id]['count'] += 1

        # user_id = profile[type]
        # user_profiles[user_id]['low'] += profile['rating_membership']['low']
        # user_profiles[user_id]['medium'] += profile['rating_membership']['medium']
        # user_profiles[user_id]['high'] += profile['rating_membership']['high']
        # user_profiles[user_id]['count'] += 1
    
    for user_id, profile in user_profiles.items():
        for label in ['low', 'medium', 'high']:
            profile[label] = np.mean(profile[label])
            profile[label] = f1(profile[label])
            

        print(str(profile))
    return user_profiles

# Rating pre-filtering: Ratings are analysed to determine whether it is eligible for
#the noise classification step by comparing the user’s and item’s profiles with a
#distance function, and determining if they are close enough


def process_rating(rating_profile, user_profiles, item_profiles):
    noisy_ratings = []
    lambda2 = 1
    user_id = rating_profile['user_id']
    item_id = rating_profile['item_id']
    rating_membership = rating_profile['rating_membership']
    rating = rating_profile['rating']

    # Initialize partial DataFrame to update noise status
    partial_df = pd.DataFrame()

    user_profile = user_profiles[user_id]
    item_profile = item_profiles[item_id]

    # Calculate Manhattan distances
    distance_user = sum(abs(user_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])
    distance_item = sum(abs(item_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])

    # Apply the minimum t-norm
    noise_degree = min(distance_user, distance_item)

    # Determine if it's noisy and return updated data
    if noise_degree >= lambda2:
        # Create a record to update in main DataFrame
        partial_df = pd.DataFrame([{
            'userId': user_id,
            'movieId': item_id,
            'rating': rating,
            'isNoisy': 1
        }])
        noisy_ratings.append((user_id, item_id, rating_membership))

    return partial_df, noisy_ratings



def process_rating1(rating_profile, user_profiles, item_profiles, lambda2):
    user_id = rating_profile['user_id']
    item_id = rating_profile['item_id']
    rating_membership = rating_profile['rating_membership']
    rating = rating_profile['rating']

    user_profile = user_profiles[user_id]
    item_profile = item_profiles[item_id]

    # Calculate Manhattan distances
    distance_user = sum(abs(user_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])
    distance_item = sum(abs(item_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])

    # Apply the minimum t-norm
    noise_degree = min(distance_user, distance_item)

    is_noisy = noise_degree >= lambda2
    return (user_id, item_id, rating_membership, is_noisy)

def detect_noise_v1(ratings_df, user_profiles, item_profiles, rating_profiles):
    noisy_ratings = []
    lambda2 = 1
    
    # Start timing
    start_time = time.time()

    # Process ratings in parallel
    results = Parallel(n_jobs=-1)(delayed(process_rating1)(rating_profile, user_profiles, item_profiles, lambda2) for rating_profile in rating_profiles)

    # Iterate through results to update noisy ratings and DataFrame
    for user_id, item_id, rating_membership, is_noisy in results:
        if is_noisy:
            noisy_ratings.append((user_id, item_id, rating_membership))
            ratings_df.loc[(ratings_df['userId'] == user_id) & 
                           (ratings_df['movieId'] == item_id) & 
                           (ratings_df['rating'] == rating_membership['rating']), 'isNoisy'] = 1

    # Calculate and print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    return ratings_df, noisy_ratings


def process_rating0(rating_profile, user_profiles, item_profiles):
    # Initialize an empty DataFrame to store results for updating
    partial_df = pd.DataFrame()

    user_id = rating_profile['user_id']
    item_id = rating_profile['item_id']
    rating_membership = rating_profile['rating_membership']
    rating = rating_profile['rating']
    lambda2 = 1

    user_profile = user_profiles.get(user_id)
    item_profile = item_profiles.get(item_id)

    if not user_profile or not item_profile:
        return partial_df, []  # Return empty if profiles are missing

    # Calculate Manhattan distances
    distance_user = sum(abs(user_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])
    distance_item = sum(abs(item_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])
    print('a')
    # Apply the minimum t-norm
    noise_degree = min(distance_user, distance_item)

    # Determine if it's noisy
    if noise_degree >= lambda2:
        partial_df = pd.DataFrame({
            'userId': [user_id],
            'movieId': [item_id],
            'rating': [rating],
            'isNoisy': [1]
        })

        noisy_ratings = [(user_id, item_id, rating_membership)]
        return partial_df, noisy_ratings

    return partial_df, []


def detect_noise(ratings_df, user_profiles, item_profiles, rating_profiles):
    noisy_ratings = []
    start_time = time.time()

    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_rating, profile, user_profiles, item_profiles)
            for profile in rating_profiles
        ]

        # Collect results from each future
        partial_dfs = []
        for future in as_completed(futures):
            try:
                partial_df, result_noisy_ratings = future.result()  # Retrieve results
                if not partial_df.empty:
                    partial_dfs.append(partial_df)
                noisy_ratings.extend(result_noisy_ratings)
            except Exception as e:
                print(f"Error processing a profile: {e}")

    # Merge partial DataFrames to update ratings_df
    if partial_dfs:
        update_df = pd.concat(partial_dfs)
        ratings_df = pd.merge(ratings_df, update_df, on=['userId', 'movieId', 'rating'], how='left')
        ratings_df['isNoisy'] = ratings_df['isNoisy'].fillna(0).astype(int)

    # End timer
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return ratings_df, noisy_ratings







def detect_noise_v0(ratings_df, user_profiles, item_profiles, rating_profiles):
    noisy_ratings = []
    #Temporaru
    start_time = time.time()

# Your code block here
    # For example:
    # run_your_function()

    # End the timer
    


    lambda2 = 1
    count = 0
    print('QBC')
    i=0
    for rating_profile in rating_profiles:
            # Calculate and print the time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        i = i + 1
        print('treating ' +str(i))
        user_id = rating_profile['user_id']
        item_id = rating_profile['item_id']
        rating_membership = rating_profile['rating_membership']
        rating = rating_profile['rating']
        print(user_id, item_id, rating_membership)
        user_profile = user_profiles[user_id]
        item_profile = item_profiles[item_id]
        
        #P*Ru is user_profile
        #P*Ri is item_profile
        #S = {Low, Medium, High}
        # Calculate Manhattan distances
        distance_user = sum(abs(user_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])
        distance_item = sum(abs(item_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])

        # Apply the minimum t-norm
        ratings_df['noisedegree'] = noise_degree = min(distance_user, distance_item)


        #noise_degree = min(abs(user_profile['low'] - rating_membership['low']), abs(item_profile['medium'] - rating_membership['medium']), abs(item_profile['high'] - rating_membership['high']))
        #print("noise degreee is the following " + str(noise_degree))
        if noise_degree >= lambda2:
            noisy_ratings.append((user_id, item_id, rating_membership))
            count = count + 1
            ratings_df.loc[(ratings_df['userId'] == user_id) & 
                                 (ratings_df['movieId'] == item_id) &
                                 (ratings_df['rating'] == rating)
                                 , 'isNoisy'] = 1
    #print("noisy count " + str(count))
    return ratings_df, noisy_ratings

# def detect_noise(ratings_df, user_profiles, item_profiles, rating_profiles):
#     client = Client()  # Create a Dask client

#     # Convert the input DataFrames to Dask DataFrames
#     ratings_df = dd.from_pandas(ratings_df, npartitions=8)  # Adjust the number of partitions as needed
#     user_profiles = dd.from_dict(user_profiles, npartitions=8)
#     item_profiles = dd.from_dict(item_profiles, npartitions=8)
#     rating_profiles_df = pd.DataFrame(rating_profiles)
#     rating_profiles_dask = dd.from_pandas(rating_profiles_df, npartitions=8)

#     # Parallelize the computations
#     noisy_ratings = []
#     lambda2 = 1
#     count = 0

#     for rating_profile in rating_profiles:
#         user_id = rating_profile['user_id']
#         item_id = rating_profile['item_id']
#         rating_membership = rating_profile['rating_membership']
#         rating = rating_profile['rating']

#         user_profile = user_profiles.loc[user_id].compute()
#         item_profile = item_profiles.loc[item_id].compute()

#         distance_user = sum(abs(user_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])
#         distance_item = sum(abs(item_profile[label] - rating_membership[label]) for label in ['low', 'medium', 'high'])

#         noise_degree = min(distance_user, distance_item)
#         print("noise degree is the following " + str(noise_degree))

#         if noise_degree >= lambda2:
#             noisy_ratings.append((user_id, item_id, rating_membership))
#             count = count + 1
#             ratings_df.loc[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == item_id) & (ratings_df['rating'] == rating), 'isNoisy'] = 1

#     print("noisy count " + str(count))
#     return ratings_df, noisy_ratings



def detect_noiseNOTUSED(user_profiles, item_profiles, ratings):
    noisy_ratings = []
    #Temporaru
    threshold = 1
    for user, user_ratings in ratings.items():
        for item, rating in user_ratings.items():
            user_profile = user_profiles[user]
            item_profile = item_profiles[item]
            #P*Ru is user_profile
            #P*Ri is item_profile
            #S = {Low, Medium, High}
            fuzzified = fuzzify_rating(rating)
            ## These lines were added when we were comparing the manual vs library fuzzy membership functions, to make sure the library is valid
            # # fuzzifiedm = fuzzify_rating_manual(rating)
            # # if fuzzified != fuzzifiedm:
            # #     print('they are not equal in detect_noiseNOTUSED')
            noise_degree = min(abs(fuzzified['low'] - user_profile['low']), abs(fuzzified['medium'] - user_profile['medium']), abs(fuzzified['high'] - user_profile['high']))
            print("noise degreee is the following " + str(noise_degree))
            if noise_degree > threshold:
                noisy_ratings.append((user, item))
    return noisy_ratings



#############################################################################################
# Main
#############################################################################################



########################################### Sample data ####################################################
# ALT K U to undo comment on sample data
# # Sample ratings data: {user: {item: rating}}
# ratings = {
#     '1': {'1': 1, '2': 4},
#     '2': {'1': 3, '2': 4},
# }

# # Step 1: Fuzzy Profiling
# user_profiles = create_fuzzy_profile(ratings)
# item_profiles = create_fuzzy_profile({item: {user: ratings[user][item] for user in ratings} for item in ratings['user1']})

# print("user_profiles" + str(user_profiles))
# print("item_profiles" + str(item_profiles))

# # Step 2: Noise Detection
# #noisy_ratings = detect_noise(user_profiles, item_profiles, ratings)

# # Step 3: Noise Correction
# #corrected_ratings = correct_noisy_ratings(noisy_ratings, ratings)

#



#print(ratings_df)
def main():
    global lowlist, mediumlist, highlist
    dataset_config = get_config_data()
    dataset_name = dataset_config['dataset_name']
    
    dataset_path = dataset_config['dataset']
    if get_config_data()['split_data_is_generated']:
        ratings_df = load_ratings(dataset_path)[['userId','itemId','rating']].rename({'itemId': 'movieId'}, axis=1)
    else:
        ratings_df = load_ratings(dataset_path)[['userId','movieId','rating']]
    OUTPUT_PATH = "Outputs/"+  dataset_name.replace("-", "_") +"_user_ratings_nf4_afterSplit_Test_5-fuuldt.csv"

    if not os.path.exists(OUTPUT_PATH):
        # If it doesn't exist, create the file
        with open(OUTPUT_PATH, 'w'):  # This will create an empty file
            pass

    print(f"we currently have {len(ratings_df['userId'].unique())} users and {len(ratings_df['movieId'].unique())}")

    ## Check photo in index_papers fpr a screenshot of the namings on the graph
    steps = 0.5 ## Can we auto calculate?
    r_max = np.max(ratings_df['rating'])
    r_min = np.min(ratings_df['rating'])
    print("here1")
    # These settings will be used on Datasets, concerning ML from the ranges [0.5-5] and [1-5]
    if dataset_name.startswith('ml-') or dataset_name.startswith('ml_') :
        # Movie Lens Datasets
        # In the paper we have another dataset as well =>> Movie Tweetings
        r_min_end = r_mid_innerbound = 1.5
        r_max_start = r_mid_outerbound = 4

        r_mid_start = r_min_outerbound = 2
        r_mid_end = r_max_innerbound = 3
        print("here2")


    else:
        #This will be used on additional datasets when needed - Other Than Movie Lens

        # This is something I set myself, because I didnt find anything whatsoever
        #     Low Set	r_min		->	r_min + ROUNDUP(25%(COUNT ))              
        #     r_min + ROUNDUP(25%(COUNT ))		->	r_max/2 - ROUNDUP(10%(COUNT ))              
        #     Neutral Set	r_max/2 - ROUNDUP(10%(COUNT ))		->	r_max/2 + ROUNDUP(10%(COUNT ))              
        #     r_max/2 + ROUNDUP(10%(COUNT ))		->	r_max - 25%(COUNT )                 
        #     High Set	r_max - 25%(COUNT )			r_max
        r_min_end =  r_mid_innerbound = r_min + round(0.25*(r_max - r_min))
        r_max_start = r_mid_outerbound = r_max - round(0.25*(r_max - r_min))

        r_mid_start = r_mid_innerbound = (r_max - r_min)/2 - round(0.10*(r_max - r_min))
        r_mid_start = r_mid_outerbound =(r_max - r_min)/2 + round(0.10*(r_max - r_min))
        print("here2.2")


        print("here3 main")


    x = np.arange(r_min, r_max + steps, steps)
    # Step 1: Fuzzy Profiling
    # Step 1.1: Fuzzification - Creating Sets: trimf for triangle, trapmf for trapezodial
    #When plotting this, we have eccess 0.5 from 1 and 5, this wil not affect us as we dont have 0.5 -> 1 NOR 5 -> 5.5 ratings!


    low = fuzz.trapmf(x, [r_min, r_min, r_min_end, r_min_outerbound])
    medium = fuzz.trapmf(x, [r_mid_innerbound, r_mid_start, r_mid_end, r_mid_outerbound])
    high = fuzz.trapmf(x, [r_max_innerbound, r_max_start, r_max, r_max])


    lowlist = [r_min, r_min, r_min_end, r_min_outerbound]
    mediumlist = [r_mid_innerbound, r_mid_start, r_mid_end, r_mid_outerbound]
    highlist = [r_max_innerbound, r_max_start, r_max, r_max]

    # Visualize the membership functions
    # plt.figure()
    # # Plotting low membership function
    # plt.plot(x, low, 'b', linewidth=1.5, label='Low')
    # # Plotting medium membership function
    # plt.plot(x, medium, 'g', linewidth=1.5, label='Medium')
    # #plt.plot(x, medium2, 'g', linewidth=1.5)
    # #plt.plot(x, medium3, 'g', linewidth=1.5)
    # # Plotting high membership function
    # plt.plot(x, high, 'r', linewidth=1.5, label='High')
    # # Add labels and legend
    # plt.title('Fuzzy Membership Functions')
    # plt.xlabel('X')
    # plt.ylabel('Membership')
    # plt.legend()

    # Show plot
    #plt.show()



    ########################################### Excel Sample data ####################################################


    # Step 1.2: Fuzzy Profiling
    rating_profile = create_fuzzy_rating_profiles(ratings_df, x , low, medium, high)
    # user id 1 item id 256 rating 4 rating membership low 0 medium 0 high 1
    # user id 1 item id 256 rating 3.5 rating membership low 0 medium 0 high 1
    user_profile = create_fuzzy_user_profile(rating_profile, "user_id")
    item_profile = create_fuzzy_user_profile(rating_profile, "item_id")
    print("rating_profile " + str(rating_profile[1]))
    print("user_profile " + str(user_profile))
    print("item_profile " + str(item_profile))

    ## Step 2: Noise Detection
    ratings_df['isNoisy'] = 0 # Adding the column to check if noisy or no
    # Step 2.1 : prefiltering
    eligible_ratings = prefiltering(user_profile, item_profile, rating_profile)
    print("eligible_ratings " + str(eligible_ratings[1]))
    #print("Rating Profile " + str(rating_profile[1]))
    # Step 2.2 : Noise classification
    #noisy_ratings = detect_noise(user_profile, item_profile, rating_profile)
    ratings_df, noisy_ratings = detect_noise_v1(ratings_df, user_profile, item_profile, eligible_ratings)
    #ratings_df = ratings_df.compute()
    #noisy_ratings = [tuple(r) for r in noisy_ratings.compute()]
    #print("noisy_ratings " + str(noisy_ratings[1:2]))
    # Step #? : save to excel
    #ratings_df_pandas = ratings_df.compute()
    ratings_df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset saved to", OUTPUT_PATH)

if __name__ == '__main__':
    #from dask.distributed import Client
    #client = Client(n_workers=4, threads_per_worker=2)
    start_time = time.time()

    main()

    end_time = time.time()  # Record the end time

    elapsed_time = end_time - start_time
    print(f"main() took NF4 {elapsed_time:.4f} seconds to load")