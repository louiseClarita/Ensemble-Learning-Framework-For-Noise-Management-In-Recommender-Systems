#2 - Stacking
# Paper Source https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9893798
# 2nd Paper, not used yet
# P. K. Jain, R. Pamula, and G. Srivastava, ‘‘A systematic literature review 1799 on machine learning applications for consumer sentiment analysis using 1800 online reviews,’’ Comput. Sci. Rev., vol. 41, Aug. 2021, Art. no. 100413, 1801 doi: 10.1016/j.cosrev.2021.100413 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

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
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

INPUT_FILE =r"C:\Users\clari\Desktop\M2 - Thesis\Research\Dr Jacques Bou Abdo\Recommender System\5 - Ensemble Learning Model\Input\ml-25m-subset (4)\ratings_ml-25m-subset(4)_Combined.csv"
dataset_name = 'ml-25m-subset(4)-#2'
NF_count = 4


batch_size = 50000
def load_data(INPUT_FILE):
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
    level0.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42)))
    level0.append(('ef', ExtraTreesClassifier(n_estimators=100, random_state=42)))
    level0.append(('lr', LogisticRegression(random_state=42)))
    level0.append(('knn', KNeighborsClassifier()))
    #level0.append(("dt", DecisionTreeClassifier(random_state=42)))

                  
    #level0.append(('bayes', GaussianNB()))
    # define meta learner model
    level1 = LogisticRegression(random_state=42)

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
    X = np.array([userId,movieId,rating,timestamp,nf1,nf2,nf3,nf4,genres_bin], dtype=np.float64)
    X = np.array([X])
    print(X)
    prediction = algorithm.predict(X)
    return prediction

ratings_df, training_dataset_X, training_dataset_Y, testing_dataset_X, testing_dataset_Y = load_data(INPUT_FILE)
model = get_stacking()
print(training_dataset_X.shape, training_dataset_Y.shape)
training_dataset_Y = np.array(training_dataset_Y)

model.fit(training_dataset_X,training_dataset_Y)


num_batches = len(testing_dataset_X) // batch_size + 1
appended_dataframes = []
for i in range(num_batches):
    # iloc bas kermel ysir aande series la e2dar estaamel swifter kermel l parallization // using dask
    batch = testing_dataset_X[i * batch_size: (i + 1) * batch_size].copy()
                                                                     #["userId","movieId","rating",	"timestamp","genres_bin",
    batch['isNoisy'] = batch.swifter.apply(lambda x: classify_rating(x.userId,x.movieId,x.rating,x.timestamp,x.nf1,x.nf2,x.nf3,x.nf4,x.genres_bin, model), axis=1)
    appended_dataframes.append(batch)


final_output = pd.concat(appended_dataframes, axis=0)
final_output.to_csv(r'C:\\Users\\clari\\Desktop\\M2 - Thesis\\Research\\Dr Jacques Bou Abdo\\Recommender System\\5 - Ensemble Learning Model\\output\\'+dataset_name+'\\' + dataset_name + '_el2.2_TestWithNF_fine_tuning.csv', index=False)