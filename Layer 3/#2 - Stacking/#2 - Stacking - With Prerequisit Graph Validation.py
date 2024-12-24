#2 - Stacking
# Paper Source https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9893798
# 2nd Paper, not used yet
# P. K. Jain, R. Pamula, and G. Srivastava, ‘‘A systematic literature review 1799 on machine learning applications for consumer sentiment analysis using 1800 online reviews,’’ Comput. Sci. Rev., vol. 41, Aug. 2021, Art. no. 100413, 1801 doi: 10.1016/j.cosrev.2021.100413 
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from swifter import swifter

from sklearn.tree import DecisionTreeClassifier
from swifter import swifter
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import os

dataset_name = 'ml-5m-#2.2'
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


def get_models():
    # Each of these models behqve in a unique way to the other
    models = dict()
    #models['lr'] = LogisticRegression()
    # CNN = Sequential()
    # CNN.add(Dense(25, input_dim=2, activation='relu'))
    # CNN.add(Dense(3, activation='softmax'))
    # CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #models['CNN'] = CNN
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    models['cart'] = DecisionTreeClassifier()
    models['knn'] = KNeighborsClassifier()
    return models
	
	
	

def get_stacking():
    # define the base models
    level0 = list()
    #level0.append(('lr', LogisticRegression()))

    # CNN = Sequential()
    # CNN.add(Dense(25, input_dim=4, activation='relu'))
    # CNN.add(Dense(3, activation='softmax'))
    # CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #level0.append(('CNN',CNN))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    # define meta learner model
    level1 = LogisticRegression() # Sigmoid
    # define the stacking ensemble
    # cv = int, cross-validation generator or an iterable, default=None
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# NOT USED AS WE DONT HAVE THE Y evaluate a given model using cross-validation 
def evaluate_model(model, X, y):
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
 return scores

def classify_rating_0(userId,movieId,rating,timestamp,genres_bin ,nf1,nf2,nf3,nf4,Delta_GV_X, Delta_SERENDIPITY_Y,algorithm):
    #x.userId,x.movieId,x.rating,x.genres_bin, randforest
    X = np.array([userId, movieId, rating, timestamp, genres_bin, nf1, nf2, nf3, nf4, Delta_SERENDIPITY_Y,Delta_GV_X  ], dtype=np.float64)
    X = np.array([X])
    prediction = algorithm.predict(X)
    return prediction


def classify_rating(userId,movieId,rating,timestamp,genres_bin ,nf1,nf2,nf3,nf4,Delta_GV_X, Delta_SERENDIPITY_Y,algorithm):
    #x.userId,x.movieId,x.rating,x.genres_bin, randforest
    numeric_values = [userId, movieId, rating, timestamp, genres_bin, nf1, nf2, nf3, nf4, Delta_SERENDIPITY_Y, Delta_GV_X]
    numeric_values = [pd.to_numeric(val, errors='coerce') for val in numeric_values]
    
    # Check for NaNs that might indicate failed conversions
    if any(pd.isna(numeric_values)):
        print(f"Warning: Non-numeric data detected and converted to NaN: {numeric_values}")

    # Convert to NumPy array
    X = np.array(numeric_values, dtype=np.float64).reshape(1, -1)

    # Make prediction using the provided algorithm
    prediction = algorithm.predict(X)
    
    return prediction



ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y = load_data(INPUT_FILE, SHEET_NAME)
model = get_stacking()

model.fit(training_dataset_X,training_dataset_Y)


num_batches = len(testing_dataset_X) // batch_size + 1
appended_dataframes = []
for i in range(num_batches):
    # iloc bas kermel ysir aande series la e2dar estaamel swifter kermel l parallization // using dask
    batch = testing_dataset_X[i * batch_size: (i + 1) * batch_size].copy()
                                                                     #["userId","movieId","rating",	"timestamp","genres_bin",
    print(batch.columns)                                                                 
    batch['isNoisy_#2'] = batch.swifter.apply(lambda x: classify_rating(x.userId,x.movieId,x.rating,x.timestamp,x.genres_bin,x.nf1,x.nf2,x.nf3,x.nf4,x.Delta_GV_Y,x.Delta_SERENDIPITY_X, model), axis=1)
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

predictions_clean = [pred[0].astype(int) if isinstance(pred, list) else pred for pred in final_df['isNoisy_#2']]

testing_dataset_Y_clean = testing_dataset_Y[:800].reset_index(drop=True).astype(int)  # Ensure it's a string if needed
#predictions_clean = final_df['isNoisy_#1'].astype(str)         # Ensure it's a string if needed
#accuracy = accuracy_score(testing_dataset_Y[:800], final_df['isNoisy_#1'])
accuracy = accuracy_score(testing_dataset_Y_clean, predictions_clean)
final_df['isNoisy_#1'] = predictions_clean 
print(f"Accuracy of the model: {accuracy:.2f}")
final_df['isNoisy_Graph'] = testing_dataset_Y_clean

final_output = pd.concat(appended_dataframes, axis=1)
final_df.to_csv(f'../../output/{dataset_name}/el2_on_{dataset_name}_1.csv', index=False)













