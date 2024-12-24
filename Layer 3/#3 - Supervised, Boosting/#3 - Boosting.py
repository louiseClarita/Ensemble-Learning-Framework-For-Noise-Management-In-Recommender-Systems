# Category : Supervised, Boosting
# Paper : https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf
# Doc : https://www.geeksforgeeks.org/catboost-ml/



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from swifter import swifter
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer


INPUT_FILE =r"C:\Users\clari\Desktop\M2 - Thesis\Research\Dr Jacques Bou Abdo\Recommender System\5 - Ensemble Learning Model\Input\ml-25m-subset (4)\ratings_ml-25m-subset(4)_Combined.csv"
dataset_name = 'ml-25m-subset(4)-#3'
NF_count = 4


batch_size = 50000


def get_genres(INPUT_FILE):
    itens_path = INPUT_FILE + '/movies.csv'

    items = pd.read_csv(itens_path)
    genres_array = items['genres']
    return genres_array


def load_data(INPUT_FILE):
    #ratings_df = pd.read_csv(INPUT_FILE, sheet_name=SHEET_NAME)
    ratings_df = pd.read_csv(INPUT_FILE).rename(columns={'itemId':'movieId'})
    if NF_count == 5:
        test = ratings_df["1&2&3&3.1&4 = 1"]
        test2 = ratings_df[ratings_df['1&2&3&3.1&4 = 1'] == 1] # Should be one
        # Transforming the genres to Binary
        #ratings_df = Label_Encoding(ratings_df)

        # Manually getting what i need from the analysis I did, meaning, the test 6 train

        
        # Here I am removing title because I do not need it
        #cols_to_include = ratings_df.columns.difference(['title','genres'], sort=False)
        #cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","cluster-ab-nDCG","equiv-ab-nDCG",	"ab-ndcg",	"perc-change",	"condition-1", "user_serendipity"]
        

        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 1'] == 1), 'result'] = 1
        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 0']  == 1), 'result'] = 0
        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 1'] == 0) & (ratings_df['1&2&3&3.1&4 = 0']  == 0), 'result'] = -1 # Unlabeled
        
        training_dataset = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 1) | (ratings_df['1&2&3&3.1&4 = 0']  == 1)].copy() 
        testing_dataset = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 0) & (ratings_df['1&2&3&3.1&4 = 0']  == 0)].copy() 

        cols_to_include = ["userId","movieId","rating",	"timestamp","genres","nf1","nf2","nf3","nf4","result"]
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
        #ratings_df = Label_Encoding(ratings_df)

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
        cols_to_include = ["userId","movieId","rating",	"timestamp","genres","nf1","nf2","nf3","nf4","result"]
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
    return ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y
def classify_rating(userId,movieId,rating,timestamp,genres,nf1,nf2,nf3,nf4 ,algorithm):
    #x.userId,x.movieId,x.rating,x.genres_bin, randforest
    #X = np.array([userId,movieId,rating,timestamp,genres], dtype=np.float64)
    print(genres)
    for genre in genres:
        if isinstance(genre, float):
            print(f'Float found: {genre}')
    X = np.array([userId,movieId,rating,timestamp,str(genres),nf1,nf2,nf3,nf4])
    #X = np.array([X])
    print(X)
    prediction = algorithm.predict(X)
    return prediction

ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y = load_data(INPUT_FILE)

# Categorical Features should be Movies Genres - We can get this Automatically via Code
# OneVsRestClassifier to handle multi-label classification
# create and train the CatBoostClassifier
# model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
#                            loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
#model = OneVsRestClassifier(CatBoostClassifier(iterations=100, depth=6, use_best_model=True, learning_rate=0.1, verbose=False))

model = CatBoostClassifier(iterations=100, depth=6, eval_metric='Accuracy', learning_rate=0.1,  random_seed=1234)
model.fit(training_dataset_X, training_dataset_Y, cat_features=['genres'])

num_batches = len(testing_dataset_X) // batch_size + 1
appended_dataframes = []
for i in range(num_batches):
    # iloc bas kermel ysir aande series la e2dar estaamel swifter kermel l parallization // using dask
    batch = testing_dataset_X[i * batch_size: (i + 1) * batch_size].copy()
                                                                     #["userId","movieId","rating",	"timestamp","genres_bin",
    batch['isNoisy'] = batch.swifter.apply(lambda x: classify_rating(x.userId,x.movieId,x.rating,x.timestamp,x.genres,x.nf1,x.nf2,x.nf3,x.nf4, model), axis=1)
    appended_dataframes.append(batch)


final_output = pd.concat(appended_dataframes, axis=0)
final_output.to_csv(r'C:\\Users\\clari\\Desktop\\M2 - Thesis\\Research\\Dr Jacques Bou Abdo\\Recommender System\\5 - Ensemble Learning Model\\output\\' +dataset_name+ '\\'+ dataset_name + '_el3TestWithNF.csv', index=False)
