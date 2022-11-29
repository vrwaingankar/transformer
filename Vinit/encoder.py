#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer
from .attention import GlobalSelfAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEmbedding

class EncoderLayer(Layer):
    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):
        """
        Args:
            dModel: The dimension of the transformer module
            numHeads: Number of heads of the multi head attention module in the encoder layer
            dff: The intermediate dimension size in the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)
        # define the Global Self Attention layer
        self.globalSelfAttention = GlobalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate,
        )
        # initialize the pointwise feed forward sublayer
        self.ffn = FeedForward(dff=dff, dModel=dModel, dropoutRate=dropOutRate)
    def call(self, x):
        # apply global self attention to the inputs
        x = self.globalSelfAttention(x)
        # apply feed forward network and return the outputs
        x = self.ffn(x)
        return x

class Encoder(Layer):
    def __init__(
        self,
        numLayers,
        dModel,
        numHeads,
        sourceVocabSize,
        maximumPositionEncoding,
        dff,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Args:
            numLayers: The number of encoder layers in the encoder
            dModel: The dimension of the transformer module
            numHeads: Number of heads of multihead attention layer in each encoder layer
            sourceVocabSize: The source vocabulary size
            maximumPositionEncoding: The maximum number of tokens in a sentence in the source dataset
            dff: The intermediate dimension of the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)
        # define the dimension of the model and the number of encoder layers
        self.dModel = dModel
        self.numLayers = numLayers
        # initialize the positional embedding layer
        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=sourceVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding,
        )
        # define a stack of encoder layers
        self.encoderLayers = [
            EncoderLayer(
                dModel=dModel, dff=dff, numHeads=numHeads, dropOutRate=dropOutRate
            )
            for _ in range(numLayers)
        ]
        # initialize a dropout layer
        self.dropout = Dropout(rate=dropOutRate)
    def call(self, x):
        # apply positional embedding to the source token ids
        x = self.positionalEmbedding(x)
        # apply dropout to the embedded inputs
        x = self.dropout(x)
        # iterate over the stacks of encoder layer
        for encoderLayer in self.encoderLayers:
            x = encoderLayer(x=x)
        # return the output of the encoder
        return x
