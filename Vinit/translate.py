#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup

class Translator(tf.Module):
    def __init__(
        self,
        sourceTextProcessor,
        targetTextProcessor,
        transformer,
        maxLength
    ):
        # initialize the source text processor
        self.sourceTextProcessor = sourceTextProcessor
        # initialize the target text processor and a string from
        # index string lookup layer for the target ids
        self.targetTextProcessor = targetTextProcessor
        self.targetStringFromIndex = StringLookup(
            vocabulary=targetTextProcessor.get_vocabulary(),
            mask_token="",
            invert=True
        )
        # initialize the pre-trained transformer model
        self.transformer = transformer
        self.maxLength = maxLength
    
    def tokens_to_text(self, resultTokens):
        # decode the token from index to string
        resultTextTokens = self.targetStringFromIndex(resultTokens)
        # format the result text into a human readable format
        resultText = tf.strings.reduce_join(
            inputs=resultTextTokens, axis=1, separator=" "
        )
        resultText = tf.strings.strip(resultText)
        # return the result text
        return resultText
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        # the input sentence is a string of source language
        # apply the source text processor on the list of source sentences
        sentence = self.sourceTextProcessor(sentence[tf.newaxis])
        encoderInput = sentence
        # apply the target text processor on an empty sentence
        # this will create the start and end tokens
        startEnd = self.targetTextProcessor([""])[0] # 0 index is to index the only batch
        # grab the start and end tokens individually
        startToken = startEnd[0][tf.newaxis]
        endToken = startEnd[1][tf.newaxis]
        # build the output array
        outputArray = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        outputArray = outputArray.write(index=0, value=startToken)
        # iterate over the maximum length and get the output ids
        for i in tf.range(self.maxLength):
            # transpose the output array stack
            output = tf.transpose(outputArray.stack())
            # get the predictions from the transformer and
            # grab the last predicted token
            predictions = self.transformer([encoderInput, output], training=False)
            predictions = predictions[:, -1:, :] # (bsz, 1, vocabSize)
            # get the predicted id from the predictions using argmax and
            # write the predicted id into the output array
            predictedId = tf.argmax(predictions, axis=-1)
            outputArray = outputArray.write(i+1, predictedId[0])
            # if the predicted id is the end token stop iteration
            if predictedId == endToken:
                break
        output = tf.transpose(outputArray.stack())
        text = self.tokens_to_text(output)
        return text
