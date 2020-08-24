from vizseq.scorers.ter import TERScorer
import pyter
import pandas as pd
import numpy as np


if __name__=='__main__':
    # Test 1 : test vizseq Scorer for 3 hypotheses
    
    scorer = TERScorer(corpus_level=True, sent_level=True, n_workers=2, verbose=False, extra_args=None)
    ref = [['Der neue indische Premierminister Narendra Modi trifft seinen japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .',
            'Der neue indische Premierminister Narendra Modi trifft seinen japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .',
            'Der neue indische Premierminister Narendra Modi trifft seinen japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .']]
    hypo = ['Indiens neuer Premierminister , Narendra Modi , trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit der Wahl im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .',
            'Der neue indische Premierminister Narendra Modi trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um über die wirtschaftlichen und sicherheitspolitischen',
            'Der neue indische Ministerpräsident Narendra Modi trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um die Wirtschafts- und Sicherheitsbeziehungen zu erörtern']
    tags = [['Test Group 1', 'Test Group 2']]
    #scores = scorer.score(hypo, ref, tags=tags)
    scores = scorer.score(hypo, ref)

    #print('Corpus-level ter:{:f}'.format(scores.corpus_score))
    print(f'Sentence-level ter:{scores.sent_scores}') # Sentence-level ter:[0.3125, 0.28125, 0.1875]
    for results in scores.sent_scores:
        print('Sentence-level ter:{:0.12f}'.format(results))
    #print('Sentence-level ter:{}'.format(scores.sent_scores))
    #print(f'Group ter: {scores.group_scores}')
    

#--------------------------------------------------------------------------------------------------------------------
    # Test 2: Test pyter.ter for 3 hypotheses
    '''
    ref = 'Der neue indische Premierminister Narendra Modi trifft seinen japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .'.split()
    hyp = 'Indiens neuer Premierminister , Narendra Modi , trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit der Wahl im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .'.split()
    print('{:f}'.format(pyter.ter(hyp, ref))) #0.3125
    
    ref = 'Der neue indische Premierminister Narendra Modi trifft seinen japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .'.split()
    hyp = 'Der neue indische Premierminister Narendra Modi trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um über die wirtschaftlichen und sicherheitspolitischen'.split()
    print('{:f}'.format(pyter.ter(hyp, ref))) #0.28125

    ref = 'Der neue indische Premierminister Narendra Modi trifft seinen japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .'.split()
    hyp = 'Der neue indische Ministerpräsident Narendra Modi trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um die Wirtschafts- und Sicherheitsbeziehungen zu erörtern'.split()
    print('{:f}'.format(pyter.ter(hyp, ref))) #0.1875
    
    #
    '''

#----------------------------------------------------------------------------------------------------------------------
    # Test 3 : test ter score calculation in recursion_multi.py for both vizseq Scorer and pyter.ter
    scorer = TERScorer(corpus_level=False, sent_level=True, n_workers=2, verbose=True, extra_args=None)
    d_list = ['Der neue indische Premierminister Narendra Modi trifft seinen japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .',
              'Indiens neuer Premierminister , Narendra Modi , trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit der Wahl im Mai , um Wirtschafts- und Sicherheitsbeziehungen zu erörtern .',
              'Der neue indische Premierminister Narendra Modi trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um über die wirtschaftlichen und sicherheitspolitischen',
              'Der neue indische Ministerpräsident Narendra Modi trifft sich mit seinem japanischen Amtskollegen Shinzo Abe in Tokio anlässlich seines ersten großen Auslandsbesuchs seit dem Wahlsieg im Mai , um die Wirtschafts- und Sicherheitsbeziehungen zu erörtern']
    data = pd.DataFrame(data=d_list,columns=['Predictions'])
    selected_idx = 0
    remaining_list = [0,1,1,1]

    # Test vizseq Scorer
    '''
    ref = data["Predictions"][selected_idx] # Take the latest selected index
    list_ref = np.repeat(ref,len(data.index),axis=0).reshape(-1,1)
    list_hyp = data["Predictions"].astype(str).values.tolist()
    list_ref = [[''.join(x) for x in list_ref]]
    #test_group_tag = np.arange(len(data.index)).tolist()
    scores = scorer.score(list_hyp, list_ref)
    ter_list = scores.sent_scores
    for iter_i in range(len(data.index)):
        if remaining_list[iter_i] == 0:
            ter_list[iter_i] = 0.0
    print(ter_list) # [0.0, 0.3125, 0.28125, 0.1875] 

    '''
    '''
    # Test pyter.ter
    ter_list = [[] for _ in range(len(data.index))]
    ref = data["Predictions"][selected_idx]
    print(f'Initial ter_list: {ter_list}')
    print(f'Initial ref:{ref}')
    for iter_i in range(len(data.index)):
        if remaining_list[iter_i] == 0:
            # exclude elements in selected_list as hypotheses to be compared to ref
            ter_list[iter_i] = 0.0
        else:
            ter_list[iter_i] = pyter.ter(data["Predictions"][iter_i], ref)
        print(f'iter_i: {iter_i}')
        print(f'hypo: {data["Predictions"][iter_i]}')
        print(f'ref: {ref}')
        print(f'ter_list[iter_i]:{ter_list[iter_i]}')
    print(f'Full ter_list:{ter_list}') #[0.0, 0.10084033613445378, 0.18487394957983194, 0.1092436974789916]
    '''
