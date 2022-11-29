#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

# import the necessary packages
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Mean
from Vinit.decoder import Decoder
from Vinit.encoder import Encoder

class Transformer(Model):
    def __init__(
        self,
        encNumLayers,
        decNumLayers,
        dModel,
        numHeads,
        dff,
        sourceVocabSize,
        targetVocabSize,
        maximumPositionEncoding,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Args:
            encNumLayers: The number of encoder layers
            decNumLayers: The number of decoder layers
            dModel: The dimension of the transformer model
            numHeads: The number of multi head attention module for the encoder and decoder layers
            dff: The intermediate dimension of the feed forward network
            sourceVocabSize: The source vocabulary size
            targetVocabSize: The target vocabulary size
            maximumPositionEncoding: The maximum token length in the dataset
            dropOutRate: The rate of dropout layers
        """
        super().__init__(**kwargs)
        # initialize the encoder and the decoder layers
        self.encoder = Encoder(
            numLayers=encNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            sourceVocabSize=sourceVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate,
        )
        self.decoder = Decoder(
            numLayers=decNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            targetVocabSize=targetVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate,
        )
        # define the final layer of the transformer
        self.finalLayer = Dense(units=targetVocabSize)
    def call(self, inputs):
        # get the source and the target from the inputs
        (source, target) = inputs
        # get the encoded representation from the source inputs and the
        # decoded representation from the encoder outputs and target inputs
        encoderOutput = self.encoder(x=source)
        decoderOutput = self.decoder(x=target, context=encoderOutput)
        # apply a dense layer to the decoder output to formulate the logits
        logits = self.finalLayer(decoderOutput)
        # drop the keras mask, so it doesn't scale the losses/metrics.
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        # return the final logits
        return logits
