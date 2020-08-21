#!/bin/bash

. env.sh

if [[ -z $4 ]]; then
    echo "Usage: cat RAW_FILE | prepare.sh BPE_MODEL LANG PREFIX"
    echo "  where BPE_MODEL is the path to the joint BPE model"
    echo "        LANG is the ISO 639-1 two-character language code"
    echo "        PREFIX is an output prefix"
    echo "  Generates PREFIX.tok.LANG and PREFIX.bpe.LANG"
    #exit
fi

bpe_model=$PWD/data/en-de/bpe.model
lang=en
prefix=$PWD/data/en-de

echo $bpe_model
echo $lang
echo $prefix
echo $MOSES
echo $BPE

cat $prefix/wmt15-de-en.$lang | $MOSES/scripts/tokenizer/normalize-punctuation.perl -l $lang \
    | $MOSES/scripts/tokenizer/remove-non-printing-char.perl \
    | $MOSES/scripts/tokenizer/tokenizer.perl -q -no-escape -protected $MOSES/scripts/tokenizer/basic-protected-patterns -l $lang \
    | tee $prefix/wmt15-de-en.tok.$lang \
    | $BPE/apply_bpe.py -c $bpe_model \
    > $prefix/wmt15-de-en.bpe.$lang

wait

cat $prefix/wmt17-de-en.$lang | $MOSES/scripts/tokenizer/normalize-punctuation.perl -l $lang \
    | $MOSES/scripts/tokenizer/remove-non-printing-char.perl \
    | $MOSES/scripts/tokenizer/tokenizer.perl -q -no-escape -protected  $MOSES/scripts/tokenizer/basic-protected-patterns -l $lang \
    | tee $prefix/wmt17-de-en.tok.$lang \
    | $BPE/apply_bpe.py -c $bpe_model \
    > $prefix/wmt17-de-en.bpe.$lang
