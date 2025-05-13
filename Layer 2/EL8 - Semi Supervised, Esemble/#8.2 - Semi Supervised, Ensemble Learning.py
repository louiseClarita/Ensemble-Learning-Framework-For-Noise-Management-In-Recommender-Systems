
# Paper: A reliable ensemble based approach to semi-supervised learning - ScienceDirect
# Catgeory : Semi Supervised



import numpy as np
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import mode

import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
#import tensorflow as tf

print(os.getcwd())
INPUT_FILE =r"..\Input\ml-25m-subset (4)\ratings_ml-25m-subset(4)_Combined.csv"
dataset_name = 'ml-25m-subset(4)-#4.2'
NF_count = 4

SHEET_NAME = "Analysis"
#INPUT_FILE = r"../DataPreperation/output/ml-5m/Fully_Merged_Data_cleaned.xlsx"
batch_size = 1000
n_clusters = 2
n_samples = 10  # To test Of course
uf = 0.3  # Fraction of unlabeled data to sample
k = 5
Same_Classifier = False

from scipy.optimize import minimize


def load_data(INPUT_FILE, SHEET_NAME):
    #ratings_df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME,nrows=15)
    ratings_df = pd.read_csv(INPUT_FILE).rename(columns={'itemId':'movieId'})

    if NF_count == 5:
        ratings_df = Label_Encoding(ratings_df)
        ratings_df = ratings_df
        #.rename(columns={'cluster-ab-nDCG':'cluster_ab_nDCG','equiv-ab-nDCG':'equiv_ab_nDCG',  'ab-ndcg':'ab_ndcg','perc-change':'perc_change', 'condition-1':'condition_1'})
        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 1'] == 1), 'result'] = 1
        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 0']  == 1), 'result'] = 0
        ratings_df.loc[(ratings_df['1&2&3&3.1&4 = 1'] == 0) & (ratings_df['1&2&3&3.1&4 = 0']  == 0), 'result'] = '' # Unlabeled
        # Manually getting what I need from the analysis I did
        Labeled = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 1) | (ratings_df['1&2&3&3.1&4 = 0'] == 1)].copy()
        UnLabeled = ratings_df[(ratings_df['1&2&3&3.1&4 = 1'] == 0) & (ratings_df['1&2&3&3.1&4 = 0'] == 0)].copy()
        
        #cols_to_include = ["userId","movieId","rating", "timestamp","genres_bin","nf1","nf2","nf3","nf4","cluster_ab_nDCG","equiv_ab_nDCG", "ab_ndcg",  "perc_change",  "condition_1", "user_serendipity","result"]
        cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","result"]

        Labeled = Labeled.loc[:, cols_to_include].copy()
        UnLabeled = UnLabeled.loc[:, cols_to_include].copy()


        ratings_df2 = ratings_df.loc[:, cols_to_include].copy()
    elif NF_count == 4:
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
        cols_to_include = ["userId","movieId","rating",	"timestamp","genres_bin","nf1","nf2","nf3","nf4","result"]

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

def calculate_oob_0(L, Li):
    # This is not tested yet
    # Extract features from both L and Li, excluding the label column
    features_L = L.iloc[:, :-1]  # All columns except the last one (label)
    features_Li = Li.iloc[:, :-1]  # All columns except the last one (label)
    
    # Add an index column to uniquely identify rows in L
    features_L['index'] = features_L.index
    features_Li['index'] = features_Li.index

    # Perform a left join on 'features_L' to keep all rows not in 'features_Li'
    OOBi = pd.merge(features_L, features_Li, on=features_L.columns[:-1].tolist(), how='left', indicator=True)
    
    # Filter rows where the merge indicator shows that the row was only in L
    OOBi = OOBi[OOBi['_merge'] == 'left_only']

    # Drop the merge indicator and index column
    OOBi = OOBi.drop(['_merge', 'index'], axis=1)
    
    return OOBi



def calculate_oob(L, Li):
    # Assuming the last column is the label and all other columns are features
    try:
        # Extract features from both L and Li, excluding the label column
        features_L = L.iloc[:, :-1]  # All columns except the last one
        features_Li = Li.iloc[:, :-1]  # Same for Li

        # Convert rows to tuples for set operations (tuples can handle mixed types)
        features_L_tuples = set(map(tuple, features_L.values))
        features_Li_tuples = set(map(tuple, features_Li.values))

        # Calculate the OOB as the difference between the sets of tuples
        OOBi_tuples = features_L_tuples - features_Li_tuples
        
        # Convert back to DataFrame (if needed) or handle as needed
        # Correct assignment of labels
        OOBi = pd.DataFrame(list(OOBi_tuples), columns=features_L.columns)

        # Map labels based on matching features in L
        OOBi['result'] = OOBi.apply(lambda row: L.loc[(L.iloc[:, :-1] == row).all(axis=1), 'result'].values[0], axis=1)
    except Exception as e:
        print("Error calculating OOB:", e)
        # Print shapes and types for debugging
        print(f"L shape: {L.shape}, Li shape: {Li.shape}")
        print(f"L types: {L.dtypes}, Li types: {Li.dtypes}")
        raise

    return OOBi

def calculate_distribution(L):
    """
    Calculate the class distribution of labeled data set L.

    Args:
        L: A pandas DataFrame or Series.

    Returns:
        A dictionary containing the class distribution.
    """

    if isinstance(L, pd.DataFrame):
        last_column = L.columns[-1]
    else:
        last_column = L.name

    unique, counts = np.unique(L[last_column], return_counts=True)
    return dict(zip(unique, counts))

def calculate_distribution_(L):
    """
    This didnt work!
    Calculate the class distribution of labeled data set L.
    """
    unique, counts = np.unique(L[:, -1], return_counts=True)  # Assuming labels are in the last column
    return dict(zip(unique, counts))

def robust_self_training_2(df,XL,YL,OOBi,D_class, classifier, iterations=10):
    """
    Implements Robust Self-Training for a given dataframe.

    Args:
        df: The input dataframe // Unlabeled Sampled data
        classifier: A trained classifier.
        iterations: Number of iterations.
        batch_size: Batch size for unlabeled data selection.

    Returns:
        The enriched classifier.
    """

    X = df.drop('result', axis=1)
    y = df['result']

    XOOBi = OOBi.drop('result', axis=1)
    YOOBi = OOBi['result']
    # Encode categorical features if necessary
    # if y.dtype == 'object':
    #     le = LabelEncoder()
    #     y = le.fit_transform(y)

    # Split into labeled, unlabeled, and OOB sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_unlabeled = X_test.copy()
    y_unlabeled = None

    best_classifier = classifier
    best_accuracy = 0

    for i in range(iterations):
        # Predict probabilities for unlabeled data
        #probs = classifier.predict_proba(X_unlabeled)
        #weights = np.array([1 - probs[i, y_train.mode()[0]] for i in range(len(probs))])

        # Adjust weights based on class distribution (optional)
        #weights *= D_class[y_train.mode()[0]] / D_class
        #OnesCount = pd.value_counts(YL['result'])
        df_reset = df.reset_index(drop=True)
        YL_reset = YL.reset_index(drop=True)

        # Count occurrences of each class
        class_counts = YL_reset.value_counts()
        #OnesCount = YL.value_counts()

        class_weights = 1 /  class_counts # It counts 1s
        weights = YL_reset.map(class_weights)
        batch_size = len(YOOBi)
        # Create a weighted sample
        X_batch = df_reset.sample(n=batch_size, weights=weights).drop('result',axis=1)
        #batch_indices = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        #batch_indices = np.random.choice(len(X_unlabeled), batch_size, replace=False, p=weights / weights.sum())
        # (You might implement a more sophisticated selection strategy)
        #batch_indices = np.random.choice(len(X_unlabeled), batch_size, replace=False) This is without Taking the distribution into account
        #X_batch = X_unlabeled.iloc[batch_indices]
        y_batch = classifier.predict(X_batch)

        # Update labeled dataset
        X_train = pd.concat([XL, X_batch])
        y_train = pd.concat([YL, pd.Series(y_batch)])

        # Retrain classifier
        classifier.fit(X_train, y_train)
        # Count unknown values in YOOBi


        # Evaluate on OOB set
        Yp_OOBI = classifier.predict(XOOBi)

        unknown_count_YOOBi = YOOBi[~YOOBi.isin([0, 1])].count()
        print(f"Unknown values in YOOBi: {unknown_count_YOOBi}")

        # Count unknown values in Yp_OOBI
        unknown_count_Yp_OOBI = np.sum((Yp_OOBI =='Nan'))

        print(f"Unknown values in Yp_OOBI: {unknown_count_Yp_OOBI}")
        # Generate classification report
        report = classification_report(y_batch, Yp_OOBI)
        print("Classification Report:\n", report)

        accuracy = accuracy_score(YOOBi, Yp_OOBI)


        if accuracy > best_accuracy:
            best_classifier = classifier
            best_accuracy = accuracy

    return best_classifier

def robust_self_training_1(df,XL,YL, classifiers, iterations=10, batch_size=100):
  """
  Implements Robust Self-Training for a given dataframe with an ensemble of classifiers.

  Args:
      df: The input dataframe.
      classifiers: An array of trained classifiers.
      iterations: Number of iterations.
      batch_size: Batch size for unlabeled data selection.

  Returns:
      An array of enriched classifiers.
  """

  X = df.drop('result', axis=1)
  y = df['result'] # Thwy are unlabeled so this will be empty
  #.astype(int)

  # Split into labeled, unlabeled, and OOB sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  X_unlabeled = X_test.copy()
  y_unlabeled = None

  best_classifiers = []
  best_accuracies = [0] * len(classifiers)  # Initialize accuracy list for each classifier

  for i in range(iterations):
    # Iterate through each classifier in the ensemble
    for classifier_index, classifier in enumerate(classifiers):

      # Predict probabilities for unlabeled data using current classifier
      probs = classifier.predict_proba(X_unlabeled)

      # Select batch based on probabilities and class distribution
      # (You might implement a more sophisticated selection strategy)
      batch_indices = np.random.choice(len(X_unlabeled), batch_size, replace=False)
      X_batch = X_unlabeled.iloc[batch_indices]

      # Predict labels for the batch using the current classifier
      y_batch = classifier.predict(X_batch)  

      # Update labeled dataset for the current classifier
      X_train_current = pd.concat([XL, X_batch])
      y_train_current = np.concatenate((YL.astype(int),y_batch.astype(int)))

      # Retrain the current classifier in the ensemble
      classifier.fit(X_train_current,y_train_current)

      # Evaluate on OOB set for the current classifier
      y_pred = classifier.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)

      # Update best classifier and accuracy for the current classifier
      if accuracy > best_accuracies[classifier_index]:
        best_classifiers[classifier_index] = classifier.copy()  # Deep copy to avoid modifying original
        best_accuracies[classifier_index] = accuracy

  return best_classifiers

def robust_self_training_0(df, classifier, iterations=10, batch_size=100):
    """
    Implements Robust Self-Training for a given dataframe.

    Args:
        df: The input dataframe.
        classifier: A trained classifier.
        iterations: Number of iterations.
        batch_size: Batch size for unlabeled data selection.

    Returns:
        The enriched classifier.
    """

    X = df.drop('result', axis=1)
    y = df['result']

    # Split into labeled, unlabeled, and OOB sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_unlabeled = X_test.copy()
    y_unlabeled = None

    best_classifier = classifier
    best_accuracy = 0

    for i in range(iterations):
        # Predict probabilities for unlabeled data
        probs = classifier.predict_proba(X_unlabeled)

        # Select batch based on probabilities and class distribution
        # (You might implement a more sophisticated selection strategy)
        batch_indices = np.random.choice(len(X_unlabeled), batch_size, replace=False)
        X_batch = X_unlabeled.iloc[batch_indices]
        y_batch = classifier.predict(X_batch)  # Assuming predicted labels

        # Update labeled dataset
        X_train = pd.concat([X_train, X_batch])
        y_train = pd.concat([y_train, pd.Series(y_batch)])

        # Retrain classifier
        classifier.fit(X_train, y_train)

        # Evaluate on OOB set
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_classifier = classifier
            best_accuracy = accuracy

    return best_classifier


def robust_self_training(C, L, U, OOB, D_class):
    """
    Perform robust self-training on classifier C.
    
    Parameters:
    - C: Classifier to train
    - L: Labeled dataset
    - U: Unlabeled dataset
    - OOB: Out-Of-Bag samples
    - D_class: Class distribution of L
    
    Returns:
    - Trained classifier C
    """
    # For demonstration purposes, let's assume a simple self-training approach:
    # 1. Predict on the unlabeled set U
    # 2. Select high-confidence predictions
    # 3. Add high-confidence predictions to the training set
    # 4. Retrain the classifier

    # Predict probabilities on U
    probas = C.predict_proba(U[:, :-1])  # Assuming U has the same features as L but without labels

    # Select high-confidence samples
    confidence_threshold = 0.9
    high_confidence_indices = np.max(probas, axis=1) > confidence_threshold
    high_confidence_samples = U[high_confidence_indices]

    if high_confidence_samples.size > 0:
        # Add these samples to L with their predicted labels
        high_confidence_labels = np.argmax(probas[high_confidence_indices], axis=1)
        high_confidence_samples = np.column_stack((high_confidence_samples, high_confidence_labels))

        # Combine with L
        L = np.vstack((L, high_confidence_samples))

    # Retrain the classifier with the updated L
    C.fit(L[:, :-1], L[:, -1])

    return C

def RESSEL(L, U, C, uf, k):
    """
    RESSEL algorithm implementation.
    
    Parameters:
    - L: Labeled data set
    - U: Unlabeled data set
    - C: Base classifier
    - uf: Fraction of the unlabeled data set to sample
    - k: Number of classifiers in the ensemble
    
    Returns:
    - ensemble: List of trained classifiers
    """
    # Duplicate base classifier C k times
    if not Same_Classifier:
        classifiers = C
    else:
        classifiers = [clone(C) for _ in range(k)]
    ensemble = []
    classifiers_study = [None] * k
    for classifier_index, classifier in enumerate(classifiers):
    #for i in range(k):
        # Sample with replacement from L to create Li
        sample_size = int(len(U) * uf)
        Li = resample(L, replace=True)
        OOBi = calculate_oob(L, Li)
        X = Li.drop('result', axis=1)
        Y = Li['result'].astype(int)
        if pd.api.types.is_string_dtype(Y):
                 encoder = LabelEncoder()
                 Y = encoder.fit_transform(Y)

        # Check for unexpected values
        if not set(Y).issubset({0, 1}):
          raise ValueError("Target variable must only contain 0 and 1.")
        # Sample without replacement from U to create Ui with fraction uf
        
        Ui = resample(U, replace=False, n_samples=sample_size)

        # Calculate the complement: OOBi = L \ Li np.setdiff1d(L[:, :-1], Li[:, :-1], axis=0)
        
        Xu = Ui.drop('result', axis=1)
        Yu = Ui['result']
        #Xu_train, Xu_test, yu_train, yu_test = train_test_split(Xu, Yu, test_size=0.3, random_state=42)

        # Calculate class distribution of Li
        D_class = calculate_distribution(Li)
        # svm = SVC()
        # Train classifier Ci on Li
        #classifiers[i].fit(X, Y)
        print(X.index)
        #X_train = X.loc[X.index[:5], :] #Select rows from 'start' (inclusive) to 'end' (exclusive)
        
        #y_train = Y[Y.index[:5]]
        classifier.fit(X, Y)
        #svm.fit(X_train,y_train)
        # Robust self-training for classifier Ci
        # df, classifier, iterations=10, batch_size=100)
        #classifiers[i] = robust_self_training_0(U,classifiers[i], Li, Ui, OOBi, D_class)
        print(Y [Y == 1])
        classifiers_study[classifier_index] = robust_self_training_2(U,X,Y,OOBi,D_class,classifier)
         
        # Add trained classifier to the ensemble
        #ensemble.append(classifiers_study)

    return classifiers_study

def predict_with_ensemble(ensemble, X):
    """
    Make predictions using an ensemble of classifiers by majority voting.
    
    Parameters:
    - ensemble: List of trained classifiers
    - X: Test data features
    
    Returns:
    - Final predictions based on majority vote
    """
    #predictions = [classifier.predict(X) for classifier in ensemble]

    predictions = np.array([clf.predict(X) for clf in ensemble])
    df = pd.DataFrame(predictions.T, columns=[f'Classifier_{i+1}' for i in range(predictions.shape[0])])

    # Save to CSV
   
    df.to_csv(r'../output/'+dataset_name+'/all_predictions.csv', index=False)
    #final_prediction, _ = mode(predictions, axis=1)
    indexes = np.argmax(np.array(predictions), axis=0)
    indexes_df = pd.DataFrame(indexes, columns=['index of the most frequent answer'])
    indexes_df.to_csv(r'../output/'+dataset_name+'/indexes.csv', index=False)
    final_predictions = np.array([predictions[idx, i] for i, idx in enumerate(indexes)])

    #final_predictions, _ = mode(predictions, axis=0)
    #final_predictions = np.array([np.bincount(predictions[:, i]).argmax() for i in range(predictions.shape[1])])
    # Now here I ma using the votijg technique (Using Highest frequency to get the best classifier)
    #final_prediction = np.argmax(np.array(predictions), axis=0)
    return final_predictions


# Example usage:
if __name__ == "__main__":

    
    # Main execution
    ratings_df, ratings_df2, Labeled, UnLabeled = load_data(INPUT_FILE, SHEET_NAME)
    #final_result = DECP(ratings_df2)

    #L = np.column_stack((L, L_labels))

    # Base classifier
    C = DecisionTreeClassifier() ## This is variation 2
    # List of Classifiers used in the Paper, We will set same settings as the ones in the paper!
    # Create instances of the classifiers

    ## I wil set the parameters/settings with the same values as the paper used
    # Gaussian Naive Bayes (GNB)
    gnb = GaussianNB()

    # Support Vector Machine (SVM)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')

    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=10)

    # Random Decision Tree (RDT)
    rdt = DecisionTreeClassifier(max_depth=4, max_features='sqrt')

    # Stochastic Gradient Descent (SGD) Classifier (LR)  ## log alone gives an error!
    sgd = SGDClassifier(loss='log_loss')
    # Parameters
   # Number of classifiers in the ensemble

    # Run RESSEL
    Classifiers = [gnb, svm, knn,rdt, sgd]

    Classifiers_Variation2 = [rdt, sgd]

    #unLabeled_train, unLabeled_test = train_test_split(UnLabeled, test_size=0.2, random_state=42)



    if Same_Classifier:
        ensemble = RESSEL(Labeled, UnLabeled, gnb, uf, k)

        print(ensemble)

        # These are the unlabeled data, we will use them here actually
        Xu = UnLabeled.drop('result', axis=1)
        Yu = UnLabeled['result']
        #Xu_train, Xu_test, yu_train, yu_test = train_test_split(Xu, Yu, test_size=0.3, random_state=42)
        ensemble_predictions = predict_with_ensemble(ensemble, Xu)
        Xu['isNoisy'] = ensemble_predictions
        Xu.to_csv(r'../output/'+dataset_name+'/El8_Result_same_GB_full.csv', index=False)
    else:
        ensemble = RESSEL(Labeled, UnLabeled, Classifiers, uf, k)

        print(ensemble)

        # These are the unlabeled data, we will use them here actually
        Xu = UnLabeled.drop('result', axis=1)
        Yu = UnLabeled['result']
        #Xu_train, Xu_test, yu_train, yu_test = train_test_split(Xu, Yu, test_size=0.3, random_state=42)
        ensemble_predictions = predict_with_ensemble(ensemble, Xu)
        Xu['isNoisy'] = ensemble_predictions
        Xu.to_csv(r'../output/'+dataset_name+'/El8.2_Result_Different_full_TestWithNF.csv', index=False)

    print("Ensemble of classifiers trained.")
