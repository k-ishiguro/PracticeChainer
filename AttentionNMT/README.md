# Attentional NMT
An implementation of an attention-based End-to-End Neural Machine Translation on Chainer. 

# Architecture

## network
- basedon Luong model
- Encoder: simple LSTM stacks
  - I will work on to implement bi-directional LSTM
- Decoder: simple LSTM stacks
- Attention: Luong type
- Generator (output): softmax

## training condition
- optimizer: SGD
- max. epoch: 13

## translation architecture
- search: 1-best greedy search
  - I will implemet a simple beam search. 


# How to use

## preprocess the training dataset
Feeding the training and validation corpus to preprocess.py. 
Then you will have a pickle-seriazlied data (and two human-readable dictionary files). 

## train the network
Given the pickled training dataset, you can run train.py as follows: 


### translate


# Reference

Softwares
- OpenNMT (opennmt.org)
- OpenNMT-py

Academic papers
- Luong
- Bahdanau
