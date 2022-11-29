#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:36:23 2022

@author: vinit
"""

#!pip install tensorflow_text

# USAGE
# python inference.py -s "input sentence"
# import the necessary packages
import tensorflow_text as tf_text # this is a no op import important for op registry
import tensorflow as tf
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sentence", required=True, help="input english sentence")
args = vars(ap.parse_args())

# convert the input english sentence to a constant tensor
sourceText = tf.constant(args["sentence"])
# load the translator model from disk
print("[INFO] loading the translator model from disk...")
translator = tf.saved_model.load("translator")
# perform inference and display the result
print("[INFO] translating english sentence to french...")
result = translator(sentence=sourceText)
translatedText = result.numpy()[0].decode()
print("[INFO] english sentence: {}".format(args["sentence"]))
print("[INFO] french translation: {}".format(translatedText))
