''''

Extended Isolation Forest, 2019



'''

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
h2o.init()
import pandas as pd
import numpy as np


dataset_name = 'ml-25m-subset(4)-#5'
SHEET_NAME = "Analysis"
INPUT_FILE =r"C:\Users\clari\Desktop\M2 - Thesis\Research\Dr Jacques Bou Abdo\Recommender System\5 - Ensemble Learning Model\Input\ml-25m-subset (4)\ratings_ml-25m-subset(4)_Combined.csv"
#INPUT_FILE = r"../../DataPreperation/output/ml-5m/Fully_Merged_Data_cleaned.xlsx"


def load_data(INPUT_FILE, SHEET_NAME):
    #ratings_df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME,nrows=15)
    ratings_df = pd.read_csv(INPUT_FILE).rename(columns={'itemId':'movieId'})
                             #, sheet_name=SHEET_NAME)
    # Transforming the genres to Binary
    ratings_df = Label_Encoding(ratings_df)
    ratings_df = ratings_df
    #.rename(columns={'cluster-ab-nDCG':'cluster_ab_nDCG','equiv-ab-nDCG':'equiv_ab_nDCG',  'ab-ndcg':'ab_ndcg','perc-change':'perc_change', 'condition-1':'condition_1'})
    ratings_df.loc[(ratings_df['1&2&3&4 = 1'] == 1), 'result'] = 1
    ratings_df.loc[(ratings_df['1&2&3&4 = 0']  == 1), 'result'] = 0
    ratings_df.loc[(ratings_df['1&2&3&4 = 1'] == 0) & (ratings_df['1&2&3&4 = 0']  == 0), 'result'] = '' # Unlabeled
    # Manually getting what I need from the analysis I did
    Labeled = ratings_df[(ratings_df['1&2&3&4 = 1'] == 1) | (ratings_df['1&2&3&4 = 0'] == 1)].copy()
    UnLabeled = ratings_df[(ratings_df['1&2&3&4 = 1'] == 0) & (ratings_df['1&2&3&4 = 0'] == 0)].copy()
    
    #cols_to_include = ["userId","movieId","rating", "timestamp","genres_bin","nf1","nf2","nf3","nf4","cluster_ab_nDCG","equiv_ab_nDCG", "ab_ndcg",  "perc_change",  "condition_1", "user_serendipity","result"]
    cols_to_include = ["userId","movieId","rating","timestamp","nf1","nf2","nf3","nf4","genres_bin"]

    Labeled = Labeled.loc[:, cols_to_include].copy()
    UnLabeled = UnLabeled.loc[:, cols_to_include].copy()

    #training_dataset_X, training_dataset_Y = training_dataset.iloc[:, :], training_dataset.loc[:, 'result']
    #testing_dataset_X, testing_dataset_Y = testing_dataset.iloc[:, :], testing_dataset.loc[:, 'result']
    ratings_df2 = ratings_df.loc[:, cols_to_include].copy()
    
    return ratings_df, ratings_df2, Labeled, UnLabeled

def Label_Encoding(ratings_df):
    mlb = MultiLabelBinarizer()
    genres_bin = mlb.fit_transform(ratings_df["genres"])
    genres_bin = np.apply_along_axis(lambda x: ''.join(x.astype(str)), axis=1, arr=genres_bin)
    genres_bin_int = np.array([int(bin_str, 2) for bin_str in genres_bin])
    
    genres_bin_df = pd.DataFrame(genres_bin_int, columns=["genres_bin"])
    ratings_df = pd.concat([ratings_df, genres_bin_df], axis=1)
    return ratings_df


ratings_df, ratings_df2, Labeled, UnLabeled = load_data(INPUT_FILE, SHEET_NAME)


UnLabeled.to_csv('#9.5 - Isolation Forest/Input/unlabeled.csv')

# Import the prostate dataset
h2o_df = h2o.import_file("#9.5 - Isolation Forest/Input/unlabeled.csv")


# Set the predictors#
# predictors = ["userId","movieId","rating","genres_bin","nf1","nf2","nf3","nf4"]
predictors = ["userId","movieId","rating", "timestamp","genres_bin"]
#predictors = ["AGE","RACE","DPROS","DCAPS","PSA","VOL","GLEASON"]
sample_size = int(UnLabeled.shape[0])
# Define an Extended Isolation forest model
eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                          ntrees = 100,
                               #           sample_size = sample_size,
                                          extension_level = len(predictors) - 1)

# Train Extended Isolation Forest
eif.train(x = predictors,
          training_frame = h2o_df)

# Calculate score
eif_result = eif.predict(h2o_df)

# Number in [0, 1] explicitly defined in Equation (1) from Extended Isolation Forest paper
# or in paragraph '2 Isolation and Isolation Trees' of Isolation Forest paper
anomaly_score = eif_result["anomaly_score"]

# Average path length  of the point in Isolation Trees from root to the leaf
mean_length = eif_result["mean_length"]

# Define a threshold for anomalies
threshold = 0.8

anomaly_scores_df = anomaly_score.as_data_frame()

# Count the number of anomalies
num_anomalies = np.sum(anomaly_scores_df > threshold)

UnLabeled['isNoisy'] = [1 if i > threshold else 0 for i in anomaly_scores_df.to_numpy()]

print(f"Number of anomalies: {num_anomalies}")

UnLabeled.to_csv('../Output/' +dataset_name + '/'+dataset_name+'el9_TestWithNF.csv')