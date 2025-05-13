# Code 1 Will be Of category = Classic / Supervised / Bagging
# Algorithm = Random Forest
# Paper from 2017
# Paper Name = N. Altman and M. Krzywinski, "Ensemble methods: Bagging and random forests", Nature Methods, vol. 14, no. 10, pp. 933-934, Oct. 2017.
# Using ML-100k


# Data Processing
import pandas as pd
import numpy as np
from swifter import swifter

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.preprocessing import MultiLabelBinarizer

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
SHEET_NAME = "Analysis"
#INPUT_FILE = r"C:\Users\Pc\Desktop\Clarita - M2 - Thesis\Ensemble Learning\DataPreperation\output\ml-25m\Fully_Merged_Data_cleaned.xlsx"

INPUT_FILE =r"..\..\Input\ml-25m-subset (4)\ratings_ml-25m-subset(4)_Combined.csv"
batch_size = 50000
dataset_name = 'ml-25m-subset(4)-#1'
NF_count = 4
def load_data(INPUT_FILE, SHEET_NAME):
    #ratings_df = pd.read_csv(INPUT_FILE, sheet_name=SHEET_NAME)
    ratings_df = pd.read_csv(INPUT_FILE).rename(columns={'itemId':'movieId'})
    if NF_count == 5:
        test = ratings_df["1&2&3&3.1&4 = 1"]
        test2 = ratings_df[ratings_df['1&2&3&3.1&4 = 1'] == 1] # Should be one
        # Transforming the genres to Binary
        ratings_df = Label_Encoding(ratings_df)

        # Manually getting what i need from the analysis I did, meaning, the test 6 train

        
        # Here I am removing title because I do not need it
        #cols_to_include = ratings_df.columns.difference(['title','genres'], sort=False)
        #cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","cluster-ab-nDCG","equiv-ab-nDCG",	"ab-ndcg",	"perc-change",	"condition-1", "user_serendipity"]
        

        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 1'] == 1), 'result'] = 1
        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 0']  == 1), 'result'] = 0
        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 1'] == 0) & (ratings_df['1&2&3&3.1&4 = 0']  == 0), 'result'] = -1 # Unlabeled
        
        training_dataset = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 1) | (ratings_df['1&2&3&3.1&4 = 0']  == 1)].copy() 
        testing_dataset = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 0) & (ratings_df['1&2&3&3.1&4 = 0']  == 0)].copy() 

        cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","result"]
        training_dataset = training_dataset.loc[:, cols_to_include].copy()
        testing_dataset = testing_dataset.loc[:, cols_to_include].copy()

        training_dataset_X,training_dataset_Y =  training_dataset.iloc[:, :9], training_dataset.loc[:, 'result']
        testing_dataset_X, testing_dataset_Y = testing_dataset.iloc[:, :9], testing_dataset.loc[:, 'result']
        

        print(str(training_dataset_X))
        print(str(training_dataset_Y))

        print(str(testing_dataset_X.iloc[:,6:]))
        print(str(testing_dataset_Y))
    elif NF_count == 4:
        test = ratings_df["1&2&3&4 = 1"]
        test2 = ratings_df[ratings_df['1&2&3&4 = 1'] == 1] # Should be one
        # Transforming the genres to Binary
        ratings_df = Label_Encoding(ratings_df)

        # Manually getting what i need from the analysis I did, meaning, the test 6 train

        
        # Here I am removing title because I do not need it
        #cols_to_include = ratings_df.columns.difference(['title','genres'], sort=False)
        #cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","cluster-ab-nDCG","equiv-ab-nDCG",	"ab-ndcg",	"perc-change",	"condition-1", "user_serendipity"]
        

        ratings_df.loc[(ratings_df['1&2&3&4 = 1'] == 1), 'result'] = 1
        ratings_df.loc[(ratings_df['1&2&3&4 = 0']  == 1), 'result'] = 0
        ratings_df.loc[(ratings_df['1&2&3&4 = 1'] == 0) & (ratings_df['1&2&3&4 = 0']  == 0), 'result'] = -1 # Unlabeled
        
        training_dataset = ratings_df[(ratings_df['1&2&3&4 = 1'] == 1) | (ratings_df['1&2&3&4 = 0']  == 1)].copy() 
        testing_dataset = ratings_df[(ratings_df['1&2&3&4 = 1'] == 0) & (ratings_df['1&2&3&4 = 0']  == 0)].copy() 

        #cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","result"]
        cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","result"]
        training_dataset = training_dataset.loc[:, cols_to_include].copy()
        testing_dataset = testing_dataset.loc[:, cols_to_include].copy()

        training_dataset_X,training_dataset_Y =  training_dataset.iloc[:, :9], training_dataset.loc[:, 'result']
        testing_dataset_X, testing_dataset_Y = testing_dataset.iloc[:, :9], testing_dataset.loc[:, 'result']
        

        print(str(training_dataset_X))
        print(str(training_dataset_Y))

        print(str(testing_dataset_X.iloc[:,6:]))
        print(str(testing_dataset_Y))

    return ratings_df, training_dataset_X,training_dataset_Y, testing_dataset_X, testing_dataset_Y


def Label_Encoding(ratings_df):
    Multi_Label_Binarizer = MultiLabelBinarizer()
    print(Multi_Label_Binarizer.fit_transform(ratings_df["genres"]))
    genres_bin = Multi_Label_Binarizer.fit_transform(ratings_df["genres"])
    genres_bin = np.apply_along_axis(lambda x: ''.join(x.astype(str)), axis=1, arr=genres_bin)
    genres_bin_df = pd.DataFrame(genres_bin, columns=["genres_bin"])

    #genres_bin_df = genres_bin_df.apply(lambda x: ''.join(x.astype(str)), axis=1)
    #genres_bin_df.rename({"0":"genres_bin"}, axis=1) # kermel to rename the column li jebto inplace=True)
    ratings_df = pd.concat([ratings_df, genres_bin_df], axis=1)
    return ratings_df

def classify_rating(userId,movieId,rating,timestamp,genres_bin, nf1,nf2,nf3,nf4 ,algorithm):
    #x.userId,x.movieId,x.rating,x.genres_bin, randforest
    input_list = [userId, movieId, rating, timestamp, genres_bin,nf1,nf2,nf3,nf4]
    for index, element in enumerate(input_list):
        if isinstance(element, (list, np.ndarray)):  # Add other sequence types if necessary
           print(f"Element at index {index} is a sequence: {element}")

    print(userId,movieId,rating,timestamp,genres_bin)
    input_array = np.column_stack(input_list)
    #prediction = algorithm.predict([userId,movieId,rating,timestamp,genres_bin,nf1,nf2,nf3,nf4,cluster_ab_nDCG,equiv_ab_nDCG,	ab_ndcg,perc_change,condition_1, user_serendipity])
    prediction = algorithm.predict(input_array)
    return prediction
# def classify_rating(userId,movieId,rating,timestamp,genres_bin ,nf1,nf2,nf3,nf4,cluster_ab_nDCG,equiv_ab_nDCG,	ab_ndcg,perc_change,condition_1, user_serendipity,algorithm):
#     #x.userId,x.movieId,x.rating,x.genres_bin, randforest
#     input_list = [userId, movieId, rating, timestamp, genres_bin, nf1, nf2, nf3, nf4, 
#               cluster_ab_nDCG, equiv_ab_nDCG, ab_ndcg, perc_change, condition_1, user_serendipity]
#     for index, element in enumerate(input_list):
#         if isinstance(element, (list, np.ndarray)):  # Add other sequence types if necessary
#            print(f"Element at index {index} is a sequence: {element}")

#     print(userId,movieId,rating,timestamp,genres_bin,nf1,nf2,nf3,nf4,cluster_ab_nDCG,equiv_ab_nDCG,	ab_ndcg,perc_change,condition_1, user_serendipity)
#     input_array = np.column_stack(input_list)
#     #prediction = algorithm.predict([userId,movieId,rating,timestamp,genres_bin,nf1,nf2,nf3,nf4,cluster_ab_nDCG,equiv_ab_nDCG,	ab_ndcg,perc_change,condition_1, user_serendipity])
#     prediction = algorithm.predict(input_array)
#     return prediction


ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y = load_data(INPUT_FILE, SHEET_NAME)
randforest = RandomForestClassifier()

randforest.fit(training_dataset_X,training_dataset_Y)

num_batches = len(testing_dataset_X) // batch_size + 1
appended_dataframes = []
for i in range(num_batches):
    # iloc bas kermel ysir aande series la e2dar estaamel swifter kermel l parallization // using dask
    batch = testing_dataset_X[i * batch_size: (i + 1) * batch_size].copy()
                                                                     #["userId","movieId","rating",	"timestamp","genres_bin",
    #    batch['isNoisy'] = batch.swifter.apply(lambda x: classify_rating(x.userId,x.movieId,x.rating,x.timestamp,x.genres_bin,x.nf1,x.nf2,x.nf3,x.nf4,x.cluster_ab_nDCG,x.equiv_ab_nDCG, x.ab_ndcg,x.perc_change,x.condition_1, x.user_serendipity, randforest), axis=1)
                                                                 
    batch['isNoisy'] = batch.swifter.apply(lambda x: classify_rating(x.userId,x.movieId,x.rating,x.timestamp,x.genres_bin,x.nf1,x.nf2,x.nf3,x.nf4, randforest), axis=1)
    appended_dataframes.append(batch)


final_output = pd.concat(appended_dataframes, axis=0) # Vertically 
final_output.to_csv(r'..\\..\\output\\' +dataset_name+ '\\'+ dataset_name + '_el1_withnfs_fine_tuning.csv', index=False)


#X_train, X_test, y_train, y_test = train_test_split(ratings_df, test_size=0.2)

