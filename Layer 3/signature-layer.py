'''

This is to be used if signature is inside the second layer (Data set that entered the layer 2 will onky be classified here)
'''

import os
import sys
print(os.getcwd())
sys.path.append(r'C:\Users\clari\Desktop\M2 - Thesis\Research\Dr Jacques Bou Abdo\Recommender System\7 - Signature\Obfuscation_Application\pythonProject')

from Obfuscation.obfuscation import Optout
from Obfuscation.helpers import Helpers



import pandas as pd
# We will set the folder of the algorithm with the highest algorithm
Winning_Algorithm = 'ml_25m_subset(4)_#2.4'
path = 'ml-25m-subset(4)-#2.4'
OTP = 'ml-25m-subset(4)-#2.4'
dataset = 'ml-25m'
ELNB = '1'

Output_Path = r"C:\Users\clari\Desktop\M2 - Thesis\Research\Dr Jacques Bou Abdo\Recommender System\7 - Signature\Obfuscation_Application\pythonProject\Input\ml-25m-subset(3)finaloutput(#2.4).csv"
#ratings_path = Output_Path + '\\' + path + '\\' + path + '_el' + ELNB + '.csv'
if '#8.4' in Winning_Algorithm:
    ratings_path =  Output_Path + '\\' + path + '\\' + 'ratings.csv'

optout = Optout()
s_helpers = Helpers()


# load the ratings csv
ratings_df = pd.read_csv(Output_Path).rename({'movieId': 'itemId'}, axis=1)
# call the first noise algo to get the dataset with natural noise
ratings_Noisy = ratings_df[(ratings_df['isNoisy'] == 1)].copy()

opt_out_users = optout.get_opt_out_users(ratings_Noisy)
optout_df = pd.DataFrame(list(opt_out_users.items()),columns=['userId', 'OptOut'])
optout_df.to_csv('OptOut03112024('+Winning_Algorithm+').csv',index=False)

#optout_df.to_csv('../Output/' + OTP  + '/OptOut.csv',index=False)

ratings_df = pd.merge(ratings_df, optout_df, on='userId',how='left')
ratings_df['OptOut'].fillna(0, inplace=True)
ratings_df.to_csv('ratings_with_optout_'+Winning_Algorithm+'.csv',index=False)


#ratings_df.to_csv('../Output/' + OTP  + '/ratings_with_optout2.csv',index=False)
