#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer
from Vinit.attention import CausalSelfAttention, CrossAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEmbedding

class DecoderLayer(Layer):
    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):
        """
        Args:
            dModel: The dimension of the transformer module
            numHeads: Number of heads of the multi head attention module in the encoder layer
            dff: The intermediate dimension size in the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)
        # initialize the causal attention module
        self.causalSelfAttention = CausalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate,
        )
        # initialize the cross attention module
        self.crossAttention = CrossAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate,
        )
        # initialize a feed forward network
        self.ffn = FeedForward(
            dff=dff,
            dModel=dModel,
            dropoutRate=dropOutRate,
        )
    def call(self, x, context):
        x = self.causalSelfAttention(x=x)
        x = self.crossAttention(x=x, context=context)
        # get the attention scores for plotting later
        self.lastAttentionScores = self.crossAttention.lastAttentionScores
        # apply feedforward network to the outputs and return it
        x = self.ffn(x)
        return x

class Decoder(Layer):
    def __init__(
        self,
        numLayers,
        dModel,
        numHeads,
        targetVocabSize,
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
            targetVocabSize: The target vocabulary size
            maximumPositionEncoding: The maximum number of tokens in a sentence in the source dataset
            dff: The intermediate dimension of the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)
        # define the dimension of the model and the number of decoder layers
        self.dModel = dModel
        self.numLayers = numLayers
        # initialize the positional embedding layer
        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=targetVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding,
        )
        # define a stack of decoder layers
        self.decoderLayers = [
            DecoderLayer(
                dModel=dModel, dff=dff, numHeads=numHeads, dropOutRate=dropOutRate
            )
            for _ in range(numLayers)
        ]
        # initialize a dropout layer
        self.dropout = Dropout(rate=dropOutRate)
    def call(self, x, context):
        # apply positional embedding to the target token ids
        x = self.positionalEmbedding(x)
        # apply dropout to the embedded targets
        x = self.dropout(x)
        # iterate over the stacks of decoder layer
        for decoderLayer in self.decoderLayers:
            x = decoderLayer(x=x, context=context)
        # get the attention scores and cache it
        self.lastAttentionScores = self.decoderLayers[-1].lastAttentionScores
        # return the output of the decoder
        return x
