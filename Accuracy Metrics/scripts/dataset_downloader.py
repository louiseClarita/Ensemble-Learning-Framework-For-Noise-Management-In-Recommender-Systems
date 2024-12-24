import os
import requests


""" Loads a dataset from an online link into ./datasets/ """
def initialize_dataset(dataset):

    if dataset == 'ml-100k':
        os.makedirs(os.path.join('datasets', dataset), exist_ok=True)
        urls = ['https://files.grouplens.org/datasets/movielens/ml-100k/u.user',
                'https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                'https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                'https://files.grouplens.org/datasets/movielens/ml-100k/u.genre',
                'https://files.grouplens.org/datasets/movielens/ml-100k/u.occupation']

        for url in urls:
            file_name = os.path.basename(url)
            r = requests.get(url, allow_redirects=True)
            open(os.path.join('datasets', dataset, file_name), 'wb').write(r.content)
