import os
import codecs
import pandas as pd

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.evaluation.python_evaluation import user_serendipity


# top k items to recommend
TOP_K = 10

# Model parameters
EPOCHS = 50
BATCH_SIZE = 1024
SEED = DEFAULT_SEED  # Set None for non-deterministic results
yaml_file = "./models/lightgcn/config/lightgcn.yaml"

hparams = prepare_hparams(
    yaml_file,
    n_layers=3,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=0.005,
    eval_epoch=5,
    top_k=TOP_K,
)

train_df = pd.read_csv('output/exp-3-epinions/train.csv').rename(columns={'Unnamed: 0': 'index'})
test_df = pd.read_csv('output/exp-3-epinions/test.csv').rename(columns={'Unnamed: 0': 'index'})
train = train_df.set_index('index')
test = test_df.set_index('index')

def train_on_groups(clusters):
    user_serendipity_dfs = []

    train_clusters = train_df.reset_index().merge(clusters, on='userID')
    total_groups = list(set(clusters.group_clusters.to_list()))
    print("Total groups:", len(total_groups))

    # Loop over all possible group clusters
    for target_group in total_groups:

        # Train data
        target_group_df = train_clusters[train_clusters['group_clusters'] == target_group]
        train = target_group_df[['userID', 'itemID', 'rating']]

        # Test data - select only ratings that can be predicted
        users_in_train = list(set(train.userID.to_list()))
        test = test_df[test_df.userID.isin(users_in_train)]

        # Train model
        data = ImplicitCF(train=train, test=test, seed=SEED)
        model = LightGCN(hparams, data, seed=SEED)
        model.fit()
        topk_scores = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

        # Ealuate model
        eval_serendipity = user_serendipity(train, topk_scores)
        user_serendipity_dfs.append(eval_serendipity)

    all_user_serendipity = pd.concat(user_serendipity_dfs)

    return all_user_serendipity
