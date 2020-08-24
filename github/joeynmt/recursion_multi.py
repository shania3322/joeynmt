import pandas as pd
import numpy as np
import pyter
from vizseq.scorers.ter import TERScorer
from vizseq.scorers.bleu import BLEUScorer
import time


"""
Given all hypotheses from one sequence, finding n hypotheses that satisfy both quality and diversity criteria.
Quality scores are represented as "Scores" in DataFrame, which are confidence scores generating from an nmt model.
Diversity scores are represented as averaged TER scores.
Calculating a new score with weighted quality score and the weighted diversity score, which is described as:
z_score = alpha * quality scores + beta * diversity score

Pseudocode of seaching n hypothese(non recursion):
    if n == 1:
        selected_index = argmax(hypothesis.Scores)
        selected_list.append(selected_index)
        remove selected_index from remaining_list
    else if n > 1:
        Calculate z_score for all hypotheses in remaining_list:
        z_scores = alpha * hypothesis.Scores + beta * 1/(n-1) * Sum ( TER(remaining hypotheses, hyp_1st), TER(remaining hypotheses, hyp_2nd), ..., TER(remaining hypotheses, hyp_nth)) 
        selected_index = argmax(z_scores)  
        selected_list.append(selected_index)
        remove selected_index from remaining_list  
        
"""

"""
Recursion and memorizing are used for actual implementation.

Input:
    data: all generated predictions corresponding quality scores for one sequence
    n: the number of predictions to be selected according to criteria of quality and diversity
    alpha: weight for quality scores
    beta: weight for diversity scores
    
Return:
    selected_list: a list of indexes of n selected predictions. 
    remaining_list: a list with value False for selected indexes. True for unselected indexes
    memo_ter: a list of TER score lists. Used for memorizing previously calculated TER scores          
"""

def find_seq_list(data: pd.DataFrame, n: int, scorer, alpha=1.0, beta=1.0):
    memo_ter = [[] for _ in range(n - 1)]
    selected_list = []
    remaining_list = [bool(1) for _ in range(len(data.index))] # Initialize remaining_list with lists of True

    def cal_seq(data: pd.DataFrame, selected_list: list, remaining_list: list, n: int, memo_ter: list, alpha: float,
                beta: float, scorer):

        #assert n > 0, "Number of selected predictions has to be a positive integer."

        # Select the top score in data["Scores"] as the first selected index
        if n == 1:
            selected_idx = np.argmax(data["Scores"].to_numpy())
            selected_list.append(selected_idx)
            remaining_list[selected_idx] = False
            return selected_list, remaining_list, memo_ter

        if n > 1:
            selected_list, remaining_list, memo_ter = cal_seq(data, selected_list, remaining_list, n - 1, memo_ter,
                                                              alpha, beta, scorer)

            #print(n - 2)

            '''Code modified for multi processing'''
            #ter_list = [[] for _ in range(len(data.index))] # ter_list stores TER scores for n
            ref = data["Predictions"][selected_list[n - 2]] # Take the latest selected index
            list_ref = np.repeat(ref,len(data.index),axis=0).reshape(-1,1)
            list_hyp = data["Predictions"].astype(str).values.tolist()
            list_ref = [[''.join(x) for x in list_ref]]
            #test_group_tag = np.arange(len(data.index)).tolist()
            scores = scorer.score(list_hyp, list_ref)
            ter_list = scores.sent_scores

            for iter_i in range(len(data.index)):
                if remaining_list[iter_i] == 0:
                    ter_list[iter_i] = 0.0
                    # Setting False for already selected indexes in remaining_list to exclude them from calculating TER scores
            memo_ter[n - 2] = ter_list  # Save TER socres to memo_ter
            #print("second")

            sum_ter = np.zeros((len(data.index), 1))
            z_scores = np.ones(sum_ter.shape) * (-np.inf) # Initialize z_scores with negative infinite values
            for j in range(n - 1):
                ter = np.array(memo_ter[j], dtype=np.float64).reshape(50, 1)
                sum_ter = sum_ter + ter
            sum_ter = sum_ter / (n - 1) # Calculate diversity scores by averaging TER scores
            z_scores[remaining_list] = alpha * np.array(data["Scores"][remaining_list]).reshape(-1, 1) + beta * sum_ter[
                remaining_list] # Update z_scores for remaining hypotheses
            selected_idx = np.argmax(z_scores)
            selected_list.append(selected_idx)
            remaining_list[selected_idx] = False

            return selected_list, remaining_list, memo_ter

    selected_ls, remaining_ls, ter_ls = cal_seq(data, selected_list, remaining_list, n, memo_ter, alpha, beta, scorer)
    #print("third")
    #print(len(ter_ls))
    return selected_ls, remaining_ls, ter_ls


#debug

def main():
    # Read fist 100 hypotheses from .csv file. Take all 50 hypotheses for 2nd sequence

    scorer = TERScorer(corpus_level=False, sent_level=True, n_workers=2, verbose=True, extra_args=None)

    path = "models\sy_transformer_wmt17_ende_groundhog_3\hyps_50.csv"
    num_rows = 100
    r_data = pd.read_csv(path, nrows=num_rows, index_col=[0])
    r_data = r_data.drop(['TER scores','Z_scores'], axis = 1)
    data = r_data[50:100]
    data = data.reset_index(drop=True)
    data.style.set_properties(**{'text-align': 'left'})
    pd.set_option('display.max_colwidth', 120)
    #print(data.head())

    #selected_ls, remaining_ls, ter_ls = find_seq_list(data,3)

    selected_ls, remaining_ls, ter_ls = find_seq_list(data, 4, scorer)


    print(selected_ls)
    print(remaining_ls)
    print(len(ter_ls))

    sub_df = data[["Predictions","Scores"]]
    print(sub_df.iloc[selected_ls])



if __name__=='__main__':
    main()