import gzip
import os

filename='corpus.tc.en.gz'
with gzip.open(filename, 'rb') as f:
    data=f.read()


fb = open("corpus.tc.en","wb" )
fb.write(data)
fb.close()
