# Code 1 Will be Of category = Classic / Supervised / Bagging
# Algorithm = Random Forest
# Paper from 2017
# Paper Name = N. Altman and M. Krzywinski, "Ensemble methods: Bagging and random forests", Nature Methods, vol. 14, no. 10, pp. 933-934, Oct. 2017.
# Using ML-100k
from sklearn.preprocessing import OneHotEncoder


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
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import os
import graphviz
SHEET_NAME = "Analysis"
dataset_name = 'ml-5m-#1.2'
SHEET_NAME = "Analysis"
INPUT_DATASET = "ml-5m-nf-1-2-3-4"
INPUT_FILE = r"../../DataPreperation/output" +"/"+ INPUT_DATASET + f"/Fully_Merged_Data_cleaned_{INPUT_DATASET}.xlsx"
batch_size = 100
print(os.getcwd())

def calculate_accuracy(Y_true, Y_pred):
    #values_without_brackets = np.array([value[1:-1] for value in Y_pred])
    accuracy = np.mean(Y_pred == Y_true)
    print("Accuracy:", accuracy)
    return accuracy


def load_data(INPUT_FILE, SHEET_NAME):
    ratings_df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME).rename(columns={'genres_y.1':'genres'})
    #test = ratings_df["1&2&3&4 = 1"]
    #test2 = ratings_df[ratings_df['1&2&3&4 = 1'] == 1] # Should be one
    # Transforming the genres to Binary
    ratings_df = Label_Encoding(ratings_df)
    print('before' + str(ratings_df.columns))
    new_ratings_df = rename_columns_and_drop_y_suffix(ratings_df)

    # new_ratings_df = ratings_df.copy().drop(['user_cat_y','user_group_y','Column1_x',
    #                                                                                                             'Column1','Unnamed: 0.1', 'Unnamed: 0_x',
    #                                                                                                            'Unnamed: 0_x',
    #                                                                                                               'Unnamed: 0_y','prediction_y','title_y.1',
    #                                                                                                               'item_cat_y', 'rating_group_y' , 'prediction_x',
    #                                                                                                               'user_group_x', 'user_cat_x', 'item_cat_x'
    #                                                                                                               ],axis=1)
    #new_ratings_df['prediction_x'] = new_ratings_df['prediction_x'].astype(float)
    #new_ratings_df['genres_bin'] = new_ratings_df['genres_bin'].astype(np.int64)
    print('after' + str(new_ratings_df.columns))
    
    X = new_ratings_df.drop(['isNoisy_Graph','genres'], axis=1)
    cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","Delta_GV_Y", "Delta_SERENDIPITY_X"]
    X =  X.loc[:, cols_to_include].copy()
    Y = new_ratings_df['isNoisy_Graph']
    training_dataset_X, testing_dataset_X, training_dataset_Y,testing_dataset_Y, = train_test_split(X, Y, test_size=0.25, stratify=Y)
    # Manually getting what i need from the analysis I did, meaning, the test 6 train
    #training_dataset = ratings_df[(ratings_df['1&2&3&4 = 1'] == 1) | (ratings_df['1&2&3&4 =0']  == 1)].copy() 
    #testing_dataset = ratings_df[(ratings_df['1&2&3&4 = 1'] == 0) & (ratings_df['1&2&3&4 =0']  == 0)].copy() 
    
    # Here I am removing title because I do not need it
    #cols_to_include = ratings_df.columns.difference(['title','genres'], sort=False)
    # cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1"]
    # training_dataset_X = training_dataset_X.loc[:, cols_to_include].copy()
    # testing_dataset_X = testing_dataset_X.loc[:, cols_to_include].copy()

    #training_dataset_X,training_dataset_Y =  training_dataset.iloc[:, 0:5], training_dataset.iloc[:, 5:6]
    #testing_dataset_X, testing_dataset_Y = testing_dataset.iloc[:, 0:5], testing_dataset.iloc[:, 5:6]

    # print(str(training_dataset_X))
    # print(str(training_dataset_Y))

    # print(str(testing_dataset_X))
    # print(str(testing_dataset_Y))

    return new_ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y


def Label_Encoding(ratings_df):
    Multi_Label_Binarizer = MultiLabelBinarizer()
    ratings_df = one_hot_encode_columns(ratings_df, [ 'rating_group_x'])

    #print(Multi_Label_Binarizer.fit_transform(ratings_df["genres"]))
    genres_bin = Multi_Label_Binarizer.fit_transform(ratings_df["genres"])
    genres_bin = np.apply_along_axis(lambda x: ''.join(x.astype(str)), axis=1, arr=genres_bin)
    genres_bin_df = pd.DataFrame(genres_bin, columns=["genres_bin"])

    #genres_bin_df = genres_bin_df.apply(lambda x: ''.join(x.astype(str)), axis=1)
    #genres_bin_df.rename({"0":"genres_bin"}, axis=1) # kermel to rename the column li jebto inplace=True)
    ratings_df = pd.concat([ratings_df, genres_bin_df], axis=1)
    return ratings_df

def classify_rating(userId,movieId,rating,timestamp,genres_bin ,nf1,nf2,nf3,nf4,Delta_GV_X, Delta_SERENDIPITY_Y,algorithm):
    #x.userId,x.movieId,x.rating,x.genres_bin, randforest
    input_list = [userId, movieId, rating, timestamp, genres_bin, nf1, nf2, nf3, nf4, Delta_SERENDIPITY_Y,Delta_GV_X              ]
    for index, element in enumerate(input_list):
        if isinstance(element, (list, np.ndarray)):  # Add other sequence types if necessary
           print(f"Element at index {index} is a sequence: {element}")

    #print(userId,movieId,rating,timestamp,genres_bin,nf1,nf2,nf3,nf4,cluster_ab_nDCG,equiv_ab_nDCG,	ab_ndcg,perc_change,condition_1, user_serendipity)
    input_array = np.column_stack(input_list)
    prediction = algorithm.predict([[userId, movieId, rating, timestamp, genres_bin, nf1, nf2, nf3, nf4, Delta_SERENDIPITY_Y,Delta_GV_X]])
    return prediction

def one_hot_encode_columns(df, columns_to_encode):
    """Encodes specified columns in a DataFrame using one-hot encoding.

    Args:
        df: The DataFrame to modify.
        columns_to_encode: A list of column names to encode.

    Returns:
        The modified DataFrame with the encoded columns.
    """

    encoder = OneHotEncoder()

    for col in columns_to_encode:
        encoded_data = encoder.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(encoded_data, index=df.index)
        df = pd.concat([df, encoded_df], axis=1).drop(columns=[col])
    return df



def rename_columns_and_drop_y_suffix(df):
    """Renames columns without '_y' suffix and drops columns ending with '_y'.

    Args:
        df: The DataFrame to modify.

    Returns:
        The modified DataFrame with renamed columns and dropped columns.
    """

    new_columns = {str(col): str(col).replace('_x', '') for col in df.columns if not str(col).endswith('_y')}
    df.rename(columns=new_columns, inplace=True)
    df.drop(columns=[str(col) for col in df.columns if str(col).endswith('_y')], inplace=True)

    return df


ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y = load_data(INPUT_FILE, SHEET_NAME)
print('we will have  ' + str(len(testing_dataset_Y)) + 'rows to test')
print('we will have  ' + str(len(training_dataset_Y)) + 'rows to train')

randforest = RandomForestClassifier()
column_names_types = training_dataset_X.dtypes.to_frame(name='Type').index.tolist()
# print(column_names_types)
column_names_types = []
for col_name, col_type in training_dataset_X.dtypes.items():
    column_names_types.append((col_name, col_type))
# print(column_names_types)
randforest.fit(training_dataset_X,training_dataset_Y)

num_batches = len(testing_dataset_X) // batch_size + 1
appended_dataframes = []
for i in range(num_batches):
    # iloc bas kermel ysir aande series la e2dar estaamel swifter kermel l parallization // using dask
    batch = testing_dataset_X[i * batch_size: (i + 1) * batch_size].copy()
                                                                     #["userId","movieId","rating",	"timestamp","genres_bin",
    a = batch.swifter.apply(lambda x: classify_rating(x.userId,x.movieId,x.rating,x.timestamp,x.genres_bin,x.nf1,x.nf2,x.nf3,x.nf4,x.Delta_GV_Y,x.Delta_SERENDIPITY_X , randforest), axis=1)
    batch['isNoisy_#1'] = a.ravel()
    #np.array([value[1:-1] for value in a ])
    appended_dataframes.append(batch)

for i, df in enumerate(appended_dataframes):
    if df.shape[0] > len(appended_dataframes[0]):
        appended_dataframes[i] = df.iloc[:len(appended_dataframes[0])]

reference_shape = appended_dataframes[0].shape
for df in appended_dataframes[1:]:
    if df.shape != reference_shape:
        print(f"DataFrame at index {i} has a different shape: {df.shape}, expected {reference_shape}")
        # Handle the different shape (e.g., truncate or pad)
df_first_7 = appended_dataframes[0:8]
df_eight =   appended_dataframes[8:9] 
# Concatenate
final_df = pd.concat(df_first_7, axis=0)

#final_output = pd.concat([df.reset_index(drop=True) for df in appended_dataframes], axis=1)
#final_output = pd.concat([df.reset_index(drop=True) for df in appended_dataframes], axis=0)
#final_output = pd.concat(final_output,testing_dataset_Y, axis=0)
#print(appended_dataframes)
#final_df['isNoisy_#1'] = final_df['isNoisy_#1'].apply(lambda x: int(x.strip('[]')))
# Assuming each element in predictions is wrapped in an extra []
predictions_clean = [pred[0].astype(int) if isinstance(pred, list) else pred for pred in final_df['isNoisy_#1']]

testing_dataset_Y_clean = testing_dataset_Y[:800].reset_index(drop=True).astype(int)  # Ensure it's a string if needed
#predictions_clean = final_df['isNoisy_#1'].astype(str)         # Ensure it's a string if needed
#accuracy = accuracy_score(testing_dataset_Y[:800], final_df['isNoisy_#1'])
accuracy = accuracy_score(testing_dataset_Y_clean, predictions_clean)
final_df['isNoisy_#1'] = predictions_clean 
print(f"Accuracy of the model: {accuracy:.2f}")
final_df['isNoisy_Graph'] = testing_dataset_Y_clean
final_df.to_csv(f'../../output/{dataset_name}/el1_on_{dataset_name}_2.csv', index=False)


#X_train, X_test, y_train, y_test = train_test_split(ratings_df, test_size=0.2)

