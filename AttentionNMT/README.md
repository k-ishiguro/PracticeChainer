# Attentional NMT
An implementation of an attention-based End-to-End Neural Machine Translation on Chainer. 

# Architecture

## network
- based on [Luong15] model
- Encoder: simple LSTM stacks(default) or bi-directional LSTM
- Decoder: simple LSTM stacks
- Attention: based on [Luong15] attention (slightly different from [Bahadanu15] (a.k.a. Groundhog))
- Generator (output): softmax

## training condition
- optimizer: SGD
- max. epoch: 20
- learning_rate: 1.0, decaying with a factor of 0.9 after 10-th epoch. 

## translation architecture
- search: Left-to-right beam search(default) or 1-best greedy search


# How to use

## preprocess the training dataset
Feeding the training and validation corpus to preprocess.py. 
Then you will have a pickle-seriazlied data (and two human-readable dictionary files). 

```
python preprocess.py -train_src <training source corpus> -train_tgt <training target corpus> -val_src <validation source corpus> -val_tgt <validation target corpus> -out_prefix <prefix for output pickle file>
```
- Mandatory arguments
  - -train_src: file path of the source-side corpus for training
  - -train_tgt: file path of the target-side corpus for training
  - -val_src: file path of the source-side corpus for validation
  - -val_tgt: file path of the target-side corpus for validation
  - -out_prefix: prefix for output pickled data and dictionaries. 
- Optional arguments
  - --src_vocab_size: maximum vocabulary size for source side (default=50000)
  - --tgt_vocab_size: maximum vocabulary size for target side (default=50000)
  - --max_sentence_length: maximum length for source/target sentenes. If exceeds, corresponding parallel pairs are removed. 
- Outputs
  - <out_prefix>.traindata.pckl: pickled training data for train.py
  - <out_prefix>.{src, tgt}.vocab: source/target side vocabulary dictionary

## train the network
Given the pickled training dataset, you can run train.py as follows: 
```
python train.py -data <prepared .traindata.pckl> -out_prefix <prefix for output files>
```
- Mandatory arguments
  - -data: file path for the pickled traindata.pckl (generated by preprocess.py)
  - -out_prefix: prefix for output file
- Optional arguments
  - training options
    - --batchsize: number of parallel pairs for training mini-batch (default=100)
    - --epoch: maximum number of the epoch (a full sweep of a training corpus) (default=20)
    - --learning_rate: initial learning rate value (default=1.0)
    - --learning_rate_decay: discount factor of learning rate (default=0.9)
    - --learning_rate_decay_start: epoch of starting learning rate decaying trick (default=10)
    - --dropout: dropout ratio throughout the network (default=0.1)
    - --max_tgt_len: maximum number of tokens for translation (used in validation) (defaut=50)
  - network structure
    - --n_layers: number of LSTM stack layers in encoder and decoder (shared between enc. and dec.) (default=2)
    - --w_vec_dim: dimension of word embedding (shared between enc. and dec.) (default=500)
    - --lstm_dim: dimension of LSTM hidden state vectors (shared between enc. and dec.) (default=500)
    - --encoder_type: choice of {'rnn', 'brnn'}. Choose 'rnn' for uni-directional LSTM encoder. Choose 'brnn' for bi-directional LSTM encoder (default=rnn)
  - computing issues
    - --gpu: GPU id. set -1 for CPU trainig. set 0 or greater for GPU training, with specified GPU core. (default=-1)
    - --resume: resume the training from snapshot (NOT YET IMPLEMENTED)
    - --frequency: frequency for taking a snopshot (NOT YET IMPLEMENTED)
    - --noplot: disable plot reportng extension (NOT YET IMPLEMENTED)
- Outputs
  - <out_prefix>_ep<epoch_number>.model.npz: epoch-wise chainer-serialized binary for all network parameters
  - <out_prefix>_ep<epoch_number>.model.spec: wpoch-wise pickled list variable for specifying the network structure

### translate
Given the query input and the trained model.npz and model.spec (generated by train.py), you can run translate.py as follows: 
```
python translate.py -modelname <prefix for trained .npz and .spec> -src <query source side corpus> -out_name <translation result file name>
```
- Mandatory arguments
  - -modelname: prefix for trained .npz and .spec files. 
    - If you have specified <out_prefix> as "test", then the train.py will output test_ep[1, 2, 3, ...].model.npz and test_ep[1, 2, 3, ...].model.spec. 
    - Then you would specify: -modelname test_ep[1,2,3,...].model
  - -src: file path for the query source side corpus
  - -out_namae: file path for output the translation results. 
- Optional arguments
  - translation options
    - --beam_size: beam size. if 1 is set, then turn on the greedy search (default=5)
    - --max_tgt_len: maximum number of tokens for translation (used in validation) (defaut=50)
  - computing issues
    - --detailed_output: if enabled, log likelihood is printed in output file. (default=False)
    - --gpu: GPU id. set -1 for CPU translation. set 0 or greater for GPU translation, with specified GPU core. (default=-1)
    - --quiet: if enabled, no print to stdout. 
    - --noplot: disable plot reportng extension (NOT YET IMPLEMENTED)
- Outputs
  - <out_name>: a text file with translation results


# Reference

Softwares
- OpenNMT (opennmt.org)
- OpenNMT-py

Academic papers
- [Luong15]: Luong+, "Effective Approaches to Attention-based Neural Machine Translation", EMNLP 2015. 
- [Bahdanau15]: Bahdanau+, "Neural Machine Translation by Jointly Learning to Align and Translate", ICLR 2015. 
