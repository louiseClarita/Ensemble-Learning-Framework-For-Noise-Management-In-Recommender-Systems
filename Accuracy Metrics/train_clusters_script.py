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

# Select MovieLens data size: 100k, 1m, 10m, or 20m
dataset = 'ml-100k'

# Model parameters
EPOCHS = 50
BATCH_SIZE = 1024
SEED = DEFAULT_SEED  # Set None for non-deterministic results

yaml_file = "./models/lightgcn/config/lightgcn.yaml"
user_file = "./models/lightgcn/output/tests/user_embeddings.csv"
item_file = "./models/lightgcn/output/item_embeddings.csv"

hparams = prepare_hparams(
    yaml_file,
    n_layers=3,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=0.005,
    eval_epoch=5,
    top_k=TOP_K,
)

dataset_path = os.path.join('datasets', dataset)
ratings_path = os.path.join(dataset_path, 'u.data')
ratings_file = codecs.open(ratings_path, 'rU', 'UTF-8')
df = pd.read_csv(ratings_file, sep='\t', names=('userID', 'itemID', 'rating', 'timestamp'))

# Normal train/test split (random portion)
train_df = pd.read_csv('output/exp-3/train.csv')
test_df = pd.read_csv('output/exp-3/test.csv')

def train_on_groups(clusters):
    user_serendipity_dfs = []

    train_clusters = train_df.reset_index().merge(clusters, left_on='userID', right_on='user_id').drop(columns=['user_id'])
    total_groups = list(set(clusters.group_clusters.to_list()))
    print("Total groups:", len(total_groups))

    # Loop over all possible group clusters
    for target_group in total_groups:

        # Train data
        target_group_df = train_clusters[train_clusters['group_clusters'] == target_group]
        train = target_group_df[['userID', 'itemID', 'rating', 'timestamp']]

        # Test data - choose only ratings that can be predicted
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
    # all_user_serendipity.to_csv('test-2-loop-model.csv')
