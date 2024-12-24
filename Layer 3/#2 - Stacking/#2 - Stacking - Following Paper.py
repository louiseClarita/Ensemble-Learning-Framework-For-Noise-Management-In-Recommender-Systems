#2 - Variation
# https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2021.600040/full




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
from sklearn.tree import DecisionTreeClassifier
from swifter import swifter
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.svm import SVR

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

SHEET_NAME = "Analysis"
#INPUT_FILE = r"C:\\Users\\Pc\\Desktop\\Clarita - M2 - Thesis\\Ensemble Learning\\dataset\\nf_result_small_latest_dataset_modified_Should be replaced by 5M.xlsx"
INPUT_FILE =r"C:\Users\clari\Desktop\M2 - Thesis\Research\Dr Jacques Bou Abdo\Recommender System\5 - Ensemble Learning Model\Input\25m-subset\ratings_ml-25m_Combined__30092024_0508_1.csv"
dataset_name = 'ml-25m-subset-#2.3'


batch_size = 50000

def load_data(INPUT_FILE, SHEET_NAME):
    #ratings_df = pd.read_csv(INPUT_FILE, sheet_name=SHEET_NAME)
    ratings_df = pd.read_csv(INPUT_FILE).rename(columns={'itemId':'movieId'})
    test = ratings_df["1&2&3&3.1&4 = 1"]
    test2 = ratings_df[ratings_df['1&2&3&3.1&4 = 1'] == 1] # Should be one
    # Transforming the genres to Binary
    ratings_df = Label_Encoding(ratings_df)

    # Manually getting what i need from the analysis I did, meaning, the test 6 train
    training_dataset = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 1) | (ratings_df['1&2&3&3.1&4 = 0']  == 1)].copy() 
    testing_dataset = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 0) & (ratings_df['1&2&3&3.1&4 = 0']  == 0)].copy() 
    
    # Here I am removing title because I do not need it
    #cols_to_include = ratings_df.columns.difference(['title','genres'], sort=False)
    #cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","cluster-ab-nDCG","equiv-ab-nDCG",	"ab-ndcg",	"perc-change",	"condition-1", "user_serendipity"]

    cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4"]
    training_dataset = training_dataset.loc[:, cols_to_include].copy()
    testing_dataset = testing_dataset.loc[:, cols_to_include].copy()

    training_dataset_X,training_dataset_Y =  training_dataset.iloc[:, :], training_dataset.iloc[:, 5:6]
    testing_dataset_X, testing_dataset_Y = testing_dataset.iloc[:, :], testing_dataset.iloc[:, 5:6]

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

def classify_rating(userId,movieId,rating,timestamp,genres_bin,nf1,nf2,nf3,nf4 ,algorithm):
    #x.userId,x.movieId,x.rating,x.genres_bin, randforest
    X = np.array([userId,movieId,rating,timestamp,genres_bin,nf1,nf2,nf3,nf4], dtype=np.float64)
    X = np.array([X])
    print(X)
    prediction = algorithm.predict(X)
    return prediction

ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y = load_data(INPUT_FILE, SHEET_NAME)
model = get_stacking()

model.fit(training_dataset_X,training_dataset_Y)


num_batches = len(testing_dataset_X) // batch_size + 1
appended_dataframes = []

# Convert testing_dataset_X to Dask DataFrame
dask_df = dd.from_pandas(testing_dataset_X, npartitions=10)

# Define the function in a way that works with Dask (you cannot use swifter with Dask)
def classify_row(row):
    return classify_rating(row.userId, row.movieId, row.rating, row.timestamp, row.genres_bin, row.nf1, row.nf2, row.nf3, row.nf4, model)

# Apply the function in parallel using Dask
with ProgressBar():
    dask_df['isNoisy'] = dask_df.apply(classify_row, axis=1, meta=('x', 'f8'))

# Compute and convert back to a pandas DataFrame if needed
result_df = dask_df.compute()

# Append to the list
appended_dataframes.append(result_df)
# for i in range(num_batches):
#     # iloc bas kermel ysir aande series la e2dar estaamel swifter kermel l parallization // using dask
#     batch = testing_dataset_X[i * batch_size: (i + 1) * batch_size].copy()
#                                                                      #["userId","movieId","rating",	"timestamp","genres_bin",
#     batch['isNoisy'] = batch.swifter.apply(lambda x: classify_rating(x.userId,x.movieId,x.rating,x.timestamp,x.genres_bin,x.nf1,x.nf2,x.nf3,x.nf4, model), axis=1)
#     appended_dataframes.append(batch)


final_output = pd.concat(appended_dataframes, axis=1)
final_output.to_csv(r'C:\\Users\\clari\\Desktop\\M2 - Thesis\\Research\\Dr Jacques Bou Abdo\\Recommender System\\5 - Ensemble Learning Model\\output\\' +dataset_name+ '\\'+ dataset_name + '_el2.csv', index=False)