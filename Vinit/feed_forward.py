#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

# import the necessary packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Add, Dense, Dropout, Layer, LayerNormalization

class FeedForward(Layer):
    def __init__(self, dff, dModel, dropoutRate=0.1, **kwargs):
        """
        Args:
            dff: Intermediate dimension for the feed forward network
            dModel: The dimension of the transformer model
            dropOutRate: Rate for dropout layer
        """
        super().__init__(**kwargs)
        # initialize the sequential model of dense layers
        self.seq = Sequential(
            [
                Dense(units=dff, activation="relu"),
                Dense(units=dModel),
                Dropout(rate=dropoutRate),
            ]
        )
        # initialize the addition layer and layer normalization layer
        self.add = Add()
        self.layernorm = LayerNormalization()
    def call(self, x):
        # add the processed input and original input
        x = self.add([x, self.seq(x)])
        # apply layer norm on the residual and return it
        x = self.layernorm(x)
        return x
