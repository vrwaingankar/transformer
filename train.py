#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

import sys
import tensorflow as tf
from Vinit.loss_accuracy import masked_accuracy, masked_loss
from Vinit.translate import Translator
tf.keras.utils.set_random_seed(42)
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.optimizers import Adam
from Vinit import config
from Vinit.dataset import (
    load_data,
    make_dataset,
    splitting_dataset,
    tf_lower_and_split_punct,
)
from Vinit.rate_schedule import CustomSchedule
from Vinit.transformer import Transformer

# load data from disk
print(f"[INFO] loading data from {config.DATA_FNAME}...")
(source, target) = load_data(fname=config.DATA_FNAME)

# split the data into training, validation, and test set
print("[INFO] splitting the dataset into train, val, and test...")
(train, val, test) = splitting_dataset(source=source, target=target)

# create source text processing layer and adapt on the training
# source sentences
print("[INFO] adapting the source text processor on the source dataset...")
sourceTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=config.SOURCE_VOCAB_SIZE
)
sourceTextProcessor.adapt(train[0])

# create target text processing layer and adapt on the training
# target sentences
print("[INFO] adapting the target text processor on the target dataset...")
targetTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=config.TARGET_VOCAB_SIZE
)
targetTextProcessor.adapt(train[1])

# build the TensorFlow data datasets of the respective data splits
print("[INFO] building TensorFlow Data input pipeline...")
trainDs = make_dataset(
    splits=train,
    batchSize=config.BATCH_SIZE,
    train=True,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)
valDs = make_dataset(
    splits=val,
    batchSize=config.BATCH_SIZE,
    train=False,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)
testDs = make_dataset(
    splits=test,
    batchSize=config.BATCH_SIZE,
    train=False,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)

# build the transformer model
print("[INFO] building the transformer model...")
transformerModel = Transformer(
    encNumLayers=config.ENCODER_NUM_LAYERS,
    decNumLayers=config.DECODER_NUM_LAYERS,
    dModel=config.D_MODEL,
    numHeads=config.NUM_HEADS,
    dff=config.DFF,
    sourceVocabSize=config.SOURCE_VOCAB_SIZE,
    targetVocabSize=config.TARGET_VOCAB_SIZE,
    maximumPositionEncoding=config.MAX_POS_ENCODING,
    dropOutRate=config.DROP_RATE,
)

# compile the model
print("[INFO] compiling the transformer model...")
learningRate = CustomSchedule(dModel=config.D_MODEL)
optimizer = Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformerModel.compile(
    loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
)

# fit the model on the training dataset
transformerModel.fit(
    trainDs,
    epochs=config.EPOCHS,
    validation_data=valDs,
)

# infer on a sentence
translator = Translator(
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
    transformer=transformerModel,
    maxLength=50,
)
# serialize and save the translator
print("[INFO] serialize the inference translator to disk...")
tf.saved_model.save(
    obj=translator,
    export_dir="translator",
)