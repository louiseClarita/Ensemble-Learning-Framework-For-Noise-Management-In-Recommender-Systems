'''
    Helper functions
    ----------------
'''
import numpy as np, csv
import pandas as pd
import os


'''
    function that loads a recommender dataset
    @param folder location
    @return dictionary
'''
def load_ratings(path):
    ratings_path = path + '/ratings.csv'

    ratings = pd.read_csv(ratings_path)

    return ratings

def load_training_ratings(path):
    ratings_path = path + '/train_data.csv'

    ratings = pd.read_csv(ratings_path)

    return ratings

def load_items(path):
    itens_path = path + '/movies.csv'

    items = pd.read_csv(itens_path)

    return items

'''
    function that retrieves the main dataset files location from the config file
'''
def get_config_data():
    myObject = {}
    ## Directory to config file --- This is added by Clarita
    config_file = os.path.dirname("C:\\Users\\clari\\Desktop\\M2 - Thesis\\Research\\Dr Jacques Bou Abdo\\Recommender System\\4 - Review\\Wissam's Work\\research-master\\")
    
    # Construct the path to the data file relative to this directory
    data_file_path = os.path.join(config_file, "config.txt")
    with open(data_file_path) as f:
        for line in f.readlines():
            print('line readng <<'+ str(line))
            key, value = line.rstrip("\n").split("=")
            if(not key in myObject):
                myObject[key] = value
            else:
                print("Duplicate assignment of key '%s'" % key)

    return myObject