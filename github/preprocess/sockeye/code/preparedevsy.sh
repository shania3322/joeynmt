#!/bin/bash

. env.sh

bpe_model=$PWD/data/en-de/bpe.model
lang=de
prefix=$PWD/data/en-de

cat $prefix/wmt17-de-en.$lang | $MOSES/scripts/tokenizer/normalize-punctuation.perl -l $lang \
    | $MOSES/scripts/tokenizer/remove-non-printing-char.perl \
    | $MOSES/scripts/tokenizer/tokenizer.perl -q -no-escape -protected  $MOSES/scripts/tokenizer/basic-protected-patterns -l $lang \
    | tee $prefix/wmt17-de-en.tok.$lang \
    | $BPE/apply_bpe.py -c $bpe_model \
    > $prefix/wmt17-de-en.bpe.$lang

exit

#####alternative- type following commands one by one

#python -m sacrebleu -t wmt15 -l en-de --echo src >wmt15-de-en.src
#python -m sacrebleu -t wmt17 -l en-de --echo src >wmt17-de-en.src

#python -m sacrebleu -t wmt15 -l en-de --echo ref >wmt15-de-en.ref
#python -m sacrebleu -t wmt17 -l en-de --echo ref >wmt17-de-en.ref


#python -m sacrebleu -t wmt15 -l en-de --echo src >wmt15-de-en.en
#python -m sacrebleu -t wmt17 -l en-de --echo src >wmt17-de-en.en

#python -m sacrebleu -t wmt15 -l en-de --echo ref >wmt15-de-en.de
#python -m sacrebleu -t wmt17 -l en-de --echo ref >wmt17-de-en.de


