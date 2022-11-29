#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

# define the dataset file
DATA_FNAME = "fra.txt"

# define the batch size
BATCH_SIZE = 512

# define the vocab size for the source and the target
# text vectorization layers
SOURCE_VOCAB_SIZE = 15_000
TARGET_VOCAB_SIZE = 15_000

# define the maximum positions in the source and target dataset
MAX_POS_ENCODING = 2048

# define the number of layers for the encoder and the decoder
ENCODER_NUM_LAYERS = 6
DECODER_NUM_LAYERS = 6

# define the dimensions of the model
D_MODEL = 512

# define the units of the point wise feed forward network
DFF = 2048

# define the number of heads and dropout rate
NUM_HEADS = 8
DROP_RATE = 0.1

# define the number of epochs to train the transformer model
EPOCHS = 25

# define the output directory
OUTPUT_DIR = "output"
