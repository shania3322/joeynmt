# author @shiya wang
# Calculating differences by TER scores, etc.

import os
import pandas as pd
import numpy as np
import pyter
from show_table import readcsv_to_df, export_to_csv, clear_pad


'''
Example one:

ref = u'SAUDI ARABIA denied THIS WEEK information published in the AMERICAN new york times'.split()
hyp = u'THIS WEEK THE SAUDIS denied information published in the new york times'.split()
print('%.3f' % pyter.ter(hyp, ref))

Example two:
ref = list(u"Pythonは、より素早く、効果的にシステムとの統合が可能なプログラミング言語です。")
hyp = list(u"Pythonは、より迅速に動作するとより効果的にシステムを統合できるプログラミング言語です。")
print('%.3f' % pyter.ter(hyp, ref))
'''


file_pth = r'C:\Users\shwa01\nmtproject\github\joeynmt\models\sy_transformer_wmt17_ende_groundhog_3\test_50.n_csv.dev'
n_best = 50
num_pred = 1000  # default = 100

predictions = readcsv_to_df(file_pth, num_pred)
hyps_data = clear_pad(predictions)   # <class 'pandas.core.frame.DataFrame'>


#TODO :
# Step 1: For one sequence,  take the top prediction as the reference, the rest predictions as hypothese to be compared.
# Calculating TER scores

ter_scores = [[] for _ in range(len(hyps_data.index))]
list_of_hyps = hyps_data["Predictions"].to_numpy(dtype = str)
ref = list_of_hyps[0]

for i, hyp in enumerate(list_of_hyps):
    ter_scores[i] = pyter.ter(hyp, ref)

hyps_data["TER scores"] = ter_scores


# TODO:
#  Step 2: Consider both scores for quality and TER scores as selection criteria
#  Scores: the higher, the better quality better
#  TER scores: the higher, the larger difference
#  Function (simple version): z_scores = beta * Scores + alpha * (TER scores),
#  beta can possibly be 0 when quality scores do not play a role in selections

# alpha = 1, beta = 1
hyps_data["Z_scores"] = hyps_data["Scores"] + hyps_data["TER scores"]

# sort data for every n_best number of sequences
sorted_data = pd.DataFrame(columns = hyps_data.columns)

for i in range(0, len(hyps_data.index), n_best):
    sorted_data = sorted_data.append(hyps_data.iloc[i:i + n_best].sort_values(by=['Z_scores'], ascending=False))


print("length of hyps:", len(hyps_data.index))
print("length of sorted data:", len(sorted_data))

# Write data to .csv file
hyps_data.to_csv(r'C:\Users\shwa01\nmtproject\github\joeynmt\models\sy_transformer_wmt17_ende_groundhog_3\hyps_50.csv')
sorted_data.to_csv(r'C:\Users\shwa01\nmtproject\github\joeynmt\models\sy_transformer_wmt17_ende_groundhog_3\hyps_sorted_50.csv')
print("hypotheses:")
print(hyps_data.head())
print("Sort with z_scores:")
print(sorted_data.head())


# TODO:
#  Step3: consider sentence length
#  3-1: length penalty for TER scores as well as the total z_scores
#  3-2: length filtering


# TODO: selecting selected_num of items as a subset from n_best list
