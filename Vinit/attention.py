#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Add, Layer, LayerNormalization, MultiHeadAttention

class BaseAttention(Layer):
    """
    The base attention module. All the other attention modules will
    be subclassed from this module.
    """
    def __init__(self, **kwargs):
        # Note the use of kwargs here, it is used to initialize the
        # MultiHeadAttention layer for all the subclassed modules
        super().__init__()
        # initialize a multihead attention layer, layer normalization layer, and
        # an addition layer
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        # apply multihead attention to the query and the context inputs
        (attentionOutputs, attentionScores) = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True,
        )
        # store the attention scores that will be later visualized
        self.lastAttentionScores = attentionScores
        # apply residual connection and layer norm
        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)
        # return the processed query
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        # apply self multihead attention
        attentionOutputs = self.mha(
            query=x,
            key=x,
            value=x,
        )
        # apply residual connection and layer norm
        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)
        # return the processed query
        return x

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        # apply self multi head attention with causal masking (look-ahead-mask)
        attentionOutputs = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True,
        )
        # apply residual connection and layer norm
        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)
        # return the processed query
        return x
