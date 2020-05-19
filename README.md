## NMT-programming
Thesis: Multiple Machine Translation Proposals in Post-editing

## Followed tutorial
1. JoeyNMT
tutorial: https://joeynmt.readthedocs.io/en/latest/tutorial.html   
github: https://github.com/joeynmt/joeynmt   
paper: https://arxiv.org/pdf/1907.12484.pdf  
preprocessing following paper: https://arxiv.org/pdf/1712.05690.pdf   


2. Open-NMT python
github: https://github.com/OpenNMT/OpenNMT-py   
tutorial: https://opennmt.net/OpenNMT-py/quickstart.html#step-1-preprocess-the-data  


## Other tutorial 
Build Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html  
Explaining Tokenization: https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html  


## Data preprecessing
Followed by Sockeye :   
https://github.com/shania3322/sockeye/tree/arxiv_1217/arxiv/code  

Training:  
```
wget -nc http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz  
wget -nc http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz  
wget -nc http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz  
wget -nc http://data.statmt.org/wmt17/translation-task/rapid2016.tgz  
```

Dev and Test:  
```
wget -nc http://data.statmt.org/wmt17/translation-task/dev.tgz  
wget -nc http://data.statmt.org/wmt17/translation-task/test.tgz  
```

Extract files:  
```
tar -xzvf training-parallel-europarl-v7.tgz  
...  
```

## Other datasets:  
Multi30k https://github.com/multi30k/dataset   
  

