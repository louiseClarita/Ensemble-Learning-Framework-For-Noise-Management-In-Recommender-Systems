## Import from custom scripts
from helpers.dataset import get_config_data, load_ratings
from datetime import datetime as dt
##

dataset_path = get_config_data()['dataset']
raw_ratings_df = load_ratings(dataset_path)[['userId','movieId','rating','timestamp']].rename({'movieId': 'itemId'}, axis=1)
raw_ratings_df['date'] = raw_ratings_df['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())

# group users per rating per day
ratings_per_day = raw_ratings_df.groupby(['userId', 'date']).size().reset_index(name="ratings-per-day").sort_values(by=['ratings-per-day'])
days_per_user = ratings_per_day.groupby(['userId']).size().reset_index(name="days-per-user").sort_values(by=['days-per-user'])
stats_df = ratings_per_day.merge(days_per_user, how='inner', on='userId')

stats_df.to_csv('dataset-stats.csv', index=False)