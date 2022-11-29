#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def masked_loss(label, prediction):
    # mask positions where the label is not equal to 0
    mask = label != 0
    # build the loss object and apply it to the labels
    lossObject = SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = lossObject(label, prediction)
    # mask the loss
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    # average the loss over the batch and return it
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_accuracy(label, prediction):
    # mask positions where the label is not equal to 0
    mask = label != 0
    # get the argmax from the logits
    prediction = tf.argmax(prediction, axis=2)
    # cast the label into the prediction datatype
    label = tf.cast(label, dtype=prediction.dtype)
    # calculate the matches
    match = label == prediction
    match = match & mask
    # cast the match and masks
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # average the match over the batch and return it
    match = tf.reduce_sum(match) / tf.reduce_sum(mask)
    return match
