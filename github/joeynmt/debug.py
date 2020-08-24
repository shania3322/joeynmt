import pandas as pd
import cProfile
import recursion
import pstats



#python -m cProfile -o recursionmulti_6.stats recursion_multi.py
#cProfile.run('recursion.main()', 'recurtionstats')

p=pstats.Stats('recursionmulti_6.stats')
p.sort_stats('time','cumtime').print_stats(.5)